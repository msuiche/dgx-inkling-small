#!/usr/bin/env bash
# node-facts.sh — everything that can make one "identical" Spark load slower
# than another, as sorted key=value lines. Read-only; no sudo needed.
#
#   for h in $RANK0_HOST $RANK1_HOST $RANK2_HOST $RANK3_HOST; do
#     ssh "$h" 'bash -s' < scripts/node-facts.sh > "facts-$h.txt"
#   done
#   diff facts-<good>.txt facts-<bad>.txt
#
# Run it idle and again while a rank sits in the weight load: memory.* then
# shows the headroom the loader is working with and activity.5s.* whether it
# is still reading.
set -u

kv() { printf '%s=%s\n' "$1" "$(tr -s '[:space:]' ' ' | sed 's/^ //; s/ $//')"; }
fact() { if [ -r "$2" ]; then kv "$1" < "$2"; else echo "$1=<unreadable>"; fi; }

{
  uname -r | kv kernel.release
  getconf PAGESIZE | kv kernel.page_size
  fact kernel.cmdline /proc/cmdline
  fact driver.version /sys/module/nvidia/version
  # Unified-memory mode of the driver (driver vs numa) and any registry keys.
  if [ -r /proc/driver/nvidia/params ]; then
    grep -iE 'Coherent|RegistryDwords|Numa|EnableUserNUMA' /proc/driver/nvidia/params \
      | while IFS=: read -r k v; do echo "$v" | kv "driver.param.$k"; done
  fi
  cat /etc/modprobe.d/*nvidia* 2>/dev/null | grep -v '^#' | sort | kv driver.modprobe

  for k in MemTotal SwapTotal HugePages_Total Hugepagesize CmaTotal; do
    awk -v k="$k:" '$1 == k {print $2, $3}' /proc/meminfo | kv "memory.static.$k"
  done
  # Moves with load: compare these between nodes at the same point in a boot.
  for k in MemAvailable MemFree Cached SwapFree Shmem Unevictable Mlocked AnonPages \
           KReclaimable SUnreclaim PageTables; do
    awk -v k="$k:" '$1 == k {print int($2 / 1024) " MiB"}' /proc/meminfo | kv "memory.now.$k"
  done
  df -m --output=used /dev/shm 2>/dev/null | tail -1 | kv memory.now.dev_shm_used_mib
  swapon --noheadings --show=NAME,TYPE,SIZE,PRIO 2>/dev/null | kv swap.devices

  for k in swappiness min_free_kbytes watermark_scale_factor watermark_boost_factor \
           vfs_cache_pressure zone_reclaim_mode overcommit_memory compaction_proactiveness \
           dirty_ratio dirty_background_ratio page-cluster; do
    fact "vm.$k" "/proc/sys/vm/$k"
  done
  fact kernel.numa_balancing /proc/sys/kernel/numa_balancing
  for k in enabled defrag shmem_enabled khugepaged/defrag; do
    fact "thp.$k" "/sys/kernel/mm/transparent_hugepage/$k"
  done

  # The disk and filesystem behind the checkpoint.
  cache="${HF_CACHE:-$HOME/.cache/huggingface}"
  findmnt -no SOURCE,FSTYPE,OPTIONS -T "$cache" 2>/dev/null | kv checkpoint.mount
  df -BG --output=avail "$cache" 2>/dev/null | tail -1 | kv checkpoint.avail
  dev=$(findmnt -no SOURCE -T "$cache" 2>/dev/null)
  disk=$(lsblk -no PKNAME "$dev" 2>/dev/null | head -1)
  disk="${disk:-$(basename "${dev:-none}")}"
  for k in read_ahead_kb scheduler max_sectors_kb nr_requests; do
    fact "checkpoint.disk.$k" "/sys/block/$disk/queue/$k"
  done

  # Five seconds of activity. During a load this separates "stopped" from
  # "slow" (mmap reads never show up in strace) and shows reclaim thrash.
  snap() {
    grep -E '^(pgmajfault|pgscan_kswapd|pgscan_direct|pgsteal_kswapd|pgsteal_direct|allocstall_normal|allocstall_movable|compact_stall|pswpin|pswpout) ' /proc/vmstat
    awk -v d="$disk" '$3 == d {print "checkpoint_disk_read_mib", int($6 / 2048)}' /proc/diskstats
  }
  before=$(snap); sleep 5; after=$(snap)
  paste <(echo "$before") <(echo "$after") | awk '{print "activity.5s." $1 "=" $4 - $2}'

  # Who else holds memory on this node.
  nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>/dev/null \
    | sort | kv gpu.compute_apps
  docker ps --format '{{.Names}} {{.Image}}' 2>/dev/null | sort | kv docker.running
  ps -eo rss=,comm= --sort=-rss 2>/dev/null | head -8 \
    | awk '{printf "%s:%dMiB ", $2, $1 / 1024}' | kv memory.now.top_rss
} | sort
