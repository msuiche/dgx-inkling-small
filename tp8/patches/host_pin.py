# SPDX-License-Identifier: Apache-2.0
"""DGX Spark (GB10) Big-Core CPU pinning & worker GC hygiene for Inkling-2-Small TP8.

GB10 CPU topology (verified via `lscpu -e` and `/proc/cpuinfo` on spark-one):
  * L3 Cluster 0: CPUs 0..4  = Cortex-A725 (LITTLE, 2808 MHz, part 0xd87)
                  CPUs 5..9  = Cortex-X925 (BIG,    3900 MHz, part 0xd85)
  * L3 Cluster 1: CPUs 10..14 = Cortex-A725 (LITTLE, 2808 MHz, part 0xd87)
                  CPUs 15..19 = Cortex-X925 (BIG,    3900 MHz, part 0xd85)

Default optimal allocation (pair with `docker run --cpuset-cpus=5-9,15-19`,
`RDMA_AR_PROXY_CORE=9`, `RDMA_AR_PROXY_CORE2=19`, `OMP_NUM_THREADS=1`):
  * CPU 9      : RDMA_AR_PROXY_CORE (exclusive BIG X925 in L3 Cluster 0; SCHED_FIFO!)
  * CPU 19     : RDMA_AR_PROXY_CORE2 (exclusive BIG X925 in L3 Cluster 1; SCHED_FIFO!)
  * CPU 5      : VLLM::Worker_TP* main thread (same L3 Cluster 0 as RDMA proxy -> shared L3!)
  * CPUs 6..8  : VLLM::Worker_TP* helper threads (AsyncOutput, cuda-EvtHandlr, ZMQ, gloo)
  * CPU 15     : VLLM::EngineCore (rank 0, isolated in L3 Cluster 1, nice -5)
  * CPUs 16..18: vllm API server + uvicorn SSE + detokenizer + ZMQ I/O (rank 0, L3 Cluster 1,
                 strictly excluding CPU 19!)

Env:
  INKLING_HOST_PIN=1              Enable worker pinning + nice -10 (default off)
  INKLING_WORKER_CORE=5           Main thread BIG core for Worker_TP* (default 5)
  INKLING_HELPER_CORES=6,7,8      Helper threads BIG cores for Worker_TP* (default 6,7,8)
  INKLING_WORKER_NICE=-10         Nice value for Worker_TP* (default -10, requires CAP_SYS_NICE)
  INKLING_WORKER_GC_FREEZE=1      Call gc.collect(); gc.freeze() after graph capture (default 1 when HOST_PIN=1)
  INKLING_PIN_ENGINE_API=1        Opt-in: auto-pin VLLM::EngineCore & parent APIServer (default off)
  INKLING_ENGINECORE_CORE=15      Exclusive BIG core for VLLM::EngineCore (default 15)
  INKLING_ENGINECORE_NICE=-5      Nice value for VLLM::EngineCore (default -5)
  INKLING_APISERVER_CORES=16,17,18 Exclusive BIG cores for vllm APIServer (default 16,17,18)
"""
from __future__ import annotations

import gc
import os

_PINNED = False
_FROZEN = False
_ENGINE_API_PINNED = False


def parse_cpuset(spec: str) -> set[int]:
    out: set[int] = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def big_cores_from_sysfs() -> list[int]:
    """Detect BIG cores (>3.2 GHz) from cpufreq sysfs; fallback to GB10 [5..9, 15..19]."""
    big = []
    for cpu in range(64):
        p = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"
        if not os.path.exists(p):
            if cpu >= 20:
                break
            continue
        try:
            khz = int(open(p).read().strip())
            if khz >= 3200000:
                big.append(cpu)
        except Exception:
            pass
    return big or [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]


def compute_worker_pin_plan(
    allowed: set[int] | None = None,
    proxy_core: int | None = None,
    worker_core: int | None = None,
    helper_cores: set[int] | None = None,
) -> tuple[int | None, set[int]]:
    """Return (main_tid_core, helper_tids_cores), strictly avoiding proxy_core and LITTLE cores."""
    if allowed is None:
        try:
            allowed = set(os.sched_getaffinity(0))
        except Exception:
            allowed = set(range(20))
    if proxy_core is None:
        proxy_core = int(os.environ.get("RDMA_AR_PROXY_CORE", "9"))
    proxy_core2 = int(os.environ.get("RDMA_AR_PROXY_CORE2", "-1"))
    if worker_core is None:
        worker_core = int(os.environ.get("INKLING_WORKER_CORE", "5"))
    if helper_cores is None:
        helper_cores = parse_cpuset(os.environ.get("INKLING_HELPER_CORES", "6,7,8"))

    # Never collide with either RDMA proxy core.
    reserved = {c for c in (proxy_core, proxy_core2) if c >= 0}
    avail = allowed - reserved
    if not avail:
        return None, set()
    main_core = worker_core if worker_core in avail else min(avail)
    helpers = (helper_cores & avail) - {main_core}
    if not helpers:
        helpers = (avail - {main_core}) or {main_core}
    return main_core, helpers


def pin_worker(logger=None) -> bool:
    """Pin Worker_TP* main thread to INKLING_WORKER_CORE and sibling threads to INKLING_HELPER_CORES."""
    global _PINNED
    if _PINNED:
        return False
    if os.environ.get("INKLING_HOST_PIN", "0").strip().lower() not in ("1", "on", "true", "yes"):
        return False
    _PINNED = True
    try:
        main_core, helper_cores = compute_worker_pin_plan()
        if main_core is None:
            return False
        pid = os.getpid()
        proxy_core = int(os.environ.get("RDMA_AR_PROXY_CORE", "-1"))
        proxy_core2 = int(os.environ.get("RDMA_AR_PROXY_CORE2", "-1"))
        proxy_cores = {c for c in (proxy_core, proxy_core2) if c >= 0}
        # Pin all existing threads (except main thread, RDMA proxy cores, and SCHED_FIFO threads)
        task_dir = f"/proc/{pid}/task"
        n_helpers = 0
        if os.path.isdir(task_dir):
            for entry in os.listdir(task_dir):
                try:
                    tid = int(entry)
                except ValueError:
                    continue
                if tid == pid:
                    continue
                try:
                    cur = os.sched_getaffinity(tid)
                    if len(cur) == 1 and next(iter(cur)) in proxy_cores:
                        continue  # Leave RDMA proxy threads exclusively on their cores!
                    if os.sched_getscheduler(tid) in (os.SCHED_FIFO, os.SCHED_RR):
                        continue  # Leave real-time RDMA proxy threads untouched!
                    os.sched_setaffinity(tid, helper_cores)
                    n_helpers += 1
                except Exception:
                    pass
        # Pin the worker main thread last to its dedicated BIG core
        os.sched_setaffinity(pid, {main_core})
        nice_val = int(os.environ.get("INKLING_WORKER_NICE", "-10"))
        if nice_val < 0:
            try:
                os.setpriority(os.PRIO_PROCESS, pid, nice_val)
            except Exception:
                pass
        if logger is not None:
            logger.info(
                "INKLING_HOST_PIN: Worker pid=%d pinned main->cpu=%d, %d helpers->cpus=%s, "
                "RDMA_AR_PROXY_CORE=%d, nice=%d",
                pid,
                main_core,
                n_helpers,
                sorted(helper_cores),
                proxy_core,
                nice_val,
            )
        return True
    except Exception as e:  # noqa: BLE001
        if logger is not None:
            logger.warning("INKLING_HOST_PIN failed: %r", e)
        return False


def freeze_worker_gc(logger=None) -> bool:
    """Freeze long-lived model/graph objects out of Gen-2 GC after capture completes."""
    global _FROZEN
    if _FROZEN:
        return False
    if os.environ.get("INKLING_HOST_PIN", "0").strip().lower() not in ("1", "on", "true", "yes"):
        return False
    if os.environ.get("INKLING_WORKER_GC_FREEZE", "1").strip().lower() in ("0", "off", "false", "no"):
        return False
    _FROZEN = True
    gc.collect()
    gc.freeze()
    if logger is not None:
        logger.info("INKLING_HOST_PIN: worker gc.freeze() completed (%d frozen objects)", gc.get_freeze_count())
    return True


def compute_engine_api_pin_plan(
    allowed: set[int] | None = None,
    proxy_cores: set[int] | None = None,
    worker_core: int | None = None,
    helper_cores: set[int] | None = None,
    engine_core: int | None = None,
    api_cores: set[int] | None = None,
) -> tuple[int | None, set[int]]:
    """Return (engine_core, api_cores) on rank 0, strictly disjoint from both RDMA proxies
    (SCHED_FIFO busy-spinners on CPUs 9 & 19) and Worker_TP0 (CPUs 5, 6-8)."""
    if allowed is None:
        try:
            allowed = set(os.sched_getaffinity(0))
        except Exception:
            allowed = set(range(20))
    if proxy_cores is None:
        p1 = int(os.environ.get("RDMA_AR_PROXY_CORE", "9"))
        p2 = int(os.environ.get("RDMA_AR_PROXY_CORE2", "-1"))
        proxy_cores = {c for c in (p1, p2) if c >= 0}
    if worker_core is None:
        worker_core = int(os.environ.get("INKLING_WORKER_CORE", "5"))
    if helper_cores is None:
        helper_cores = parse_cpuset(os.environ.get("INKLING_HELPER_CORES", "6,7,8"))
    if engine_core is None:
        engine_core = int(os.environ.get("INKLING_ENGINECORE_CORE", "15"))
    if api_cores is None:
        api_cores = parse_cpuset(os.environ.get("INKLING_APISERVER_CORES", "16,17,18"))

    # Hard exclusion: never collide with SCHED_FIFO RDMA proxy cores; prefer excluding Worker cores too.
    non_proxy = allowed - proxy_cores
    if not non_proxy:
        return None, set()
    worker_set = ({worker_core} if worker_core >= 0 else set()) | helper_cores
    preferred_pool = non_proxy - worker_set
    pool = preferred_pool if preferred_pool else non_proxy

    eng = engine_core if engine_core in pool else min(pool)
    api = (api_cores & pool) - {eng}
    if not api:
        api = (pool - {eng}) or {eng}
    return eng, api


def _pin_process_tree(pid: int, cpus: set[int], proxy_cores: set[int]) -> int:
    """Pin all non-RT, non-proxy threads of `pid` plus `pid` itself to `cpus`."""
    pinned = 0
    task_dir = f"/proc/{pid}/task"
    if os.path.isdir(task_dir):
        for entry in os.listdir(task_dir):
            try:
                tid = int(entry)
            except ValueError:
                continue
            try:
                cur = os.sched_getaffinity(tid)
                if len(cur) == 1 and next(iter(cur)) in proxy_cores:
                    continue
                if os.sched_getscheduler(tid) in (os.SCHED_FIFO, os.SCHED_RR):
                    continue
                os.sched_setaffinity(tid, cpus)
                pinned += 1
            except Exception:
                pass
    try:
        os.sched_setaffinity(pid, cpus)
        if pinned == 0:
            pinned = 1
    except Exception:
        pass
    return pinned


def _is_vllm_apiserver_pid(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except Exception:
        return False
    cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
    if "EngineCore" in cmd or "Worker" in cmd:
        return False
    return ("vllm" in cmd) or ("api_server" in cmd)


def pin_engine_and_api(logger=None) -> bool:
    """Opt-in (`INKLING_HOST_PIN=1` + `INKLING_PIN_ENGINE_API=1`): pin VLLM::EngineCore to
    exclusive BIG core 15 (nice -5) and its parent vllm APIServer to exclusive BIG cores 16-18,
    strictly avoiding SCHED_FIFO RDMA proxy cores (9, 19) and Worker_TP0 (5, 6-8)."""
    global _ENGINE_API_PINNED
    if _ENGINE_API_PINNED:
        return False
    if os.environ.get("INKLING_HOST_PIN", "0").strip().lower() not in ("1", "on", "true", "yes"):
        return False
    if os.environ.get("INKLING_PIN_ENGINE_API", "0").strip().lower() not in ("1", "on", "true", "yes"):
        return False
    _ENGINE_API_PINNED = True
    try:
        p1 = int(os.environ.get("RDMA_AR_PROXY_CORE", "9"))
        p2 = int(os.environ.get("RDMA_AR_PROXY_CORE2", "-1"))
        proxy_cores = {c for c in (p1, p2) if c >= 0}
        eng_core, api_cores = compute_engine_api_pin_plan(proxy_cores=proxy_cores)
        if eng_core is None:
            return False

        pid = os.getpid()
        n_eng = _pin_process_tree(pid, {eng_core}, proxy_cores)
        nice_val = int(os.environ.get("INKLING_ENGINECORE_NICE", "-5"))
        if nice_val < 0:
            try:
                os.setpriority(os.PRIO_PROCESS, pid, nice_val)
            except Exception:
                pass

        ppid = os.getppid()
        n_api = 0
        if api_cores and _is_vllm_apiserver_pid(ppid):
            n_api = _pin_process_tree(ppid, api_cores, proxy_cores)

        if logger is not None:
            logger.info(
                "INKLING_PIN_ENGINE_API: EngineCore pid=%d (%d tids)->cpu=%d (nice=%d), "
                "APIServer ppid=%d (%d tids)->cpus=%s (excluded proxies=%s)",
                pid,
                n_eng,
                eng_core,
                nice_val,
                ppid,
                n_api,
                sorted(api_cores) if n_api else [],
                sorted(proxy_cores),
            )
        return True
    except Exception as e:  # noqa: BLE001
        if logger is not None:
            logger.warning("INKLING_PIN_ENGINE_API failed: %r", e)
        return False
