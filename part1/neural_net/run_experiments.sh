#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

R=results
mkdir -p "$R"
LOG=$R/run.raw
./bin/main > "$LOG"

awk '
  BEGIN {print "variant,ms"}
  /Basic \(ijk\):/     && /ms$/ {print "Basic,"      $(NF-1)}
  /Reordered \(ikj\):/ && /ms$/ {print "Reordered,"  $(NF-1)}
  /^ *Unrolled:/       && /ms$/ {print "Unrolled,"   $(NF-1)}
  /Tiled\/Blocked:/    && /ms$/ {print "Tiled,"      $(NF-1)}
  /SIMD Vectorized:/   && /ms$/ {print "Vectorized," $(NF-1)}
' "$LOG" > "$R/matmul.csv"

awk '
  BEGIN {print "variant,ms"}
  /Time:/ {
    for (i = 1; i <= NF; i++) {
      if ($i == "Time:") { print $1 "," $(i+1); break }
    }
  }
' "$LOG" > "$R/nn_training.csv"

echo "wrote $R/matmul.csv and $R/nn_training.csv"
cat "$R/matmul.csv"
echo "---"
cat "$R/nn_training.csv"
