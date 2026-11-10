#!/usr/bin/env bash
# Round-robin A/B so both libraries see the same load.
#   ONE  : the serial Python-thread version (no size gate)
#   TWO  : the new library, default threshold (jaccard gated off, it lost)
set -u
cd "$(dirname "$0")/.."
A=${ONE:-/tmp/lib_old.so}
B=${TWO:-/tmp/lib_jac_off.so}
for r in 1 2; do
  LIB_AB_SO="$A" timeout 1200 pixi run python bench/parallel_ab.py >>/tmp/rr_a.log 2>&1
  LIB_AB_SO="$B" timeout 1200 pixi run python bench/parallel_ab.py >>/tmp/rr_b.log 2>&1
done
echo done
