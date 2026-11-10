#!/usr/bin/env bash
# Sequential before/after runs: old library, then the parallel one.
set -u
cd "$(dirname "$0")/.."
LIB_AB_SO=/tmp/lib_old.so timeout 600 pixi run python bench/parallel_before_after.py >/tmp/ba_old.log 2>&1
LIB_AB_SO=/tmp/lib_par_probe.so timeout 600 pixi run python bench/parallel_before_after.py >/tmp/ba_new.log 2>&1
echo done
