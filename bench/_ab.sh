#!/usr/bin/env bash
# Per-worker-count A/B: old library then new, identical shapes.
set -u
cd "$(dirname "$0")/.."
LIB_AB_SO=/tmp/lib_old.so timeout 600 pixi run python bench/parallel_ab.py >/tmp/ab_old.log 2>&1
LIB_AB_SO=/tmp/lib_par_probe.so timeout 600 pixi run python bench/parallel_ab.py >/tmp/ab_new.log 2>&1
echo done
