#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

R=results
mkdir -p "$R"

SIZES=(128 256 512 1024)
TILES=(16 32 64 128)
REPS=3

get_ms() { grep -oE '[0-9]+ ms' | head -1 | awk '{print $1}'; }
median()  { sort -n | awk 'BEGIN{c=0} {a[c++]=$1} END{if(c==0) print "NA"; else if(c%2) print a[int(c/2)]; else print (a[c/2]+a[c/2-1])/2}'; }

echo "variant,size,tile,ms" > "$R/exec_time.csv"
for N in "${SIZES[@]}"; do
    for r in $(seq 1 $REPS); do echo "$(./bin/naive $N | get_ms)"; done | median | awk -v n=$N '{print "naive,"n",,"$1}' >> "$R/exec_time.csv"
    for r in $(seq 1 $REPS); do echo "$(./bin/loop  $N | get_ms)"; done | median | awk -v n=$N '{print "loop," n",,"$1}' >> "$R/exec_time.csv"
    for r in $(seq 1 $REPS); do echo "$(./bin/simd  $N | get_ms)"; done | median | awk -v n=$N '{print "simd," n",,"$1}' >> "$R/exec_time.csv"
    for T in "${TILES[@]}"; do
        for r in $(seq 1 $REPS); do echo "$(./bin/tiling_$T $N | get_ms)"; done | median | awk -v n=$N -v t=$T '{print "tiling,"n","t","$1}' >> "$R/exec_time.csv"
        for r in $(seq 1 $REPS); do echo "$(./bin/combination_$T $N | get_ms)"; done | median | awk -v n=$N -v t=$T '{print "combination,"n","t","$1}' >> "$R/exec_time.csv"
    done
    echo "size $N done"
done

echo "variant,size,tile,instructions,L1D_load_misses,L1D_loads,MPKI" > "$R/perf.csv"

perf_one() {
    local bin="$1"; local n="$2"
    local out
    out=$(perf stat -x, -e instructions,L1-dcache-load-misses,L1-dcache-loads "$bin" "$n" 2>&1 || true)
    local instr=$(echo "$out" | awk -F, '$3=="instructions"{print $1}')
    local lmiss=$(echo "$out" | awk -F, '$3=="L1-dcache-load-misses"{print $1}')
    local lload=$(echo "$out" | awk -F, '$3=="L1-dcache-loads"{print $1}')
    local mpki="NA"
    if [[ -n "$instr" && "$instr" != "0" ]]; then
        mpki=$(echo "$lmiss $instr" | awk '{if($2>0) printf "%.6f", $1*1000/$2; else print "NA"}')
    fi
    echo "$instr,$lmiss,$lload,$mpki"
}

for N in "${SIZES[@]}"; do
    echo "naive,$N,,$(perf_one ./bin/naive $N)"        >> "$R/perf.csv"
    echo "loop,$N,,$(perf_one ./bin/loop $N)"           >> "$R/perf.csv"
    echo "simd,$N,,$(perf_one ./bin/simd $N)"           >> "$R/perf.csv"
    for T in "${TILES[@]}"; do
        echo "tiling,$N,$T,$(perf_one ./bin/tiling_$T $N)"          >> "$R/perf.csv"
        echo "combination,$N,$T,$(perf_one ./bin/combination_$T $N)" >> "$R/perf.csv"
    done
    echo "perf size $N done"
done

echo "done. see $R/exec_time.csv and $R/perf.csv"
