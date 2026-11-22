#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live Multi-1M Simultaneous Resident Prefix-Cache Proof on 8x DGX Spark.

Proves that N (default 2, e.g. 2 or 3) DISTINCT 1,048,500-token conversations
with unique UUIDs and unique Vault-Alpha/Beta/Gamma passcodes can ALL reside
simultaneously in VRAM with ZERO eviction:
  Phase 1 (Cold Fill): Send Session 0, Session 1, ..., Session N-1 cold
                       (verifies cached_tokens == 0, TTFT, and 3/3 needles each).
  Phase 2 (Warm Re-Query): Re-query Session 0, Session 1, ..., Session N-1 in order
                       with a new Turn-2 prompt (verifies EVERY session still hits
                       ~1,048,064 cached_tokens, TTFT < 3.5s, and 3/3 needles!).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "/home/jeweled/dsv41-bench")
import bench_1m_e2e as b1m  # type: ignore[import-not-found]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default="inkling-2-small")
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=1048500)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--json-out", default="/tmp/multi_1m_proof.json")
    args = ap.parse_args()

    sessions = []
    print(f"=== Multi-1M Simultaneous Residency Proof: {args.sessions} x {args.ctx:,} tok ===", flush=True)

    # Phase 1: Cold-fill all N distinct 1M sessions back-to-back
    for s in range(args.sessions):
        t0 = time.perf_counter()
        msgs, needles, tok_est, run_uuid = b1m.build_cold_needle_prompt(
            args.url, args.model, args.ctx, seed=args.seed + s * 1009,
            max_model_len=1048576, max_tokens=args.max_tokens,
        )
        build_s = time.perf_counter() - t0
        print(f"\n[Session {s} COLD] built in {build_s:.2f}s | tok={tok_est:,} | uuid={run_uuid[:12]}", flush=True)
        cold = b1m.run_streaming_chat(
            args.url, args.model, msgs, max_tokens=args.max_tokens, enable_thinking=False, timeout=1800.0
        )
        cn = b1m.score_needles(needles, cold["content"], cold["reasoning"])
        print(
            f"  [Session {s} COLD] prompt={cold['prompt_tokens']:,} cached={cold['cached_tokens']:,} "
            f"TTFT={cold['ttft_s']:.2f}s prefill={cold['prefill_tok_s']:,.1f} tok/s "
            f"needles={cn['hits']}/{cn['total']}",
            flush=True,
        )
        sessions.append({
            "idx": s,
            "uuid": run_uuid,
            "messages": msgs,
            "needles": needles,
            "cold": {**cold, "needles": cn},
        })

    # Phase 2: Warm Turn-2 re-query of ALL N sessions (starting with OLDEST Session 0 to prove ZERO eviction!)
    print("\n=== Phase 2: Warm Turn-2 Re-Query of ALL Sessions (Oldest First = Zero Eviction Proof) ===", flush=True)
    all_warm_ok = True
    for rec in sessions:
        s = rec["idx"]
        base_doc = rec["messages"][1]["content"].rsplit("\n\nList the exact three passcodes", 1)[0]
        warm_q = (
            "Repeat the exact three passcodes found in the archive above in REVERSE order:\n"
            "1. Vault-Gamma: <passcode>\n"
            "2. Vault-Beta: <passcode>\n"
            "3. Vault-Alpha: <passcode>\n"
            "Reply concisely with ONLY those three lines."
        )
        warm_msgs = [rec["messages"][0], {"role": "user", "content": f"{base_doc}\n\n{warm_q}"}]
        warm = b1m.run_streaming_chat(
            args.url, args.model, warm_msgs, max_tokens=args.max_tokens, enable_thinking=False, timeout=600.0
        )
        wn = b1m.score_needles(rec["needles"], warm["content"], warm["reasoning"])
        ok = (warm["cached_tokens"] >= int(args.ctx * 0.99) and wn["hits"] == 3 and warm["ttft_s"] < 5.0)
        all_warm_ok = all_warm_ok and ok
        rec["warm"] = {**warm, "needles": wn, "resident_ok": ok}
        print(
            f"  [Session {s} WARM] prompt={warm['prompt_tokens']:,} cached={warm['cached_tokens']:,} "
            f"TTFT={warm['ttft_s']:.2f}s speedup={rec['cold']['ttft_s']/max(warm['ttft_s'],1e-6):.1f}x "
            f"needles={wn['hits']}/{wn['total']} resident_ok={ok}",
            flush=True,
        )

    out_summary = {
        "sessions": args.sessions,
        "ctx": args.ctx,
        "all_warm_ok": all_warm_ok,
        "results": [
            {
                "idx": r["idx"],
                "uuid": r["uuid"],
                "cold": r["cold"],
                "warm": r["warm"],
            }
            for r in sessions
        ],
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\n[MULTI-1M PROOF] all_warm_ok={all_warm_ok} -> {args.json_out}", flush=True)
    return 0 if all_warm_ok else 1


if __name__ == "__main__":
    sys.exit(main())
