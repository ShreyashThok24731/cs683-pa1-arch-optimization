#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

R=results
mkdir -p "$R"

SIZES=(256 512 1024 2048)
TILES=(16 32 64 128)
REPS=3

get_ms() { grep -oE '[0-9]+ ms' | head -1 | awk '{print $1}'; }
median()  { sort -n | awk 'BEGIN{c=0} {a[c++]=$1} END{if(c==0) print "NA"; else if(c%2) print a[int(c/2)]; else print (a[c/2]+a[c/2-1])/2}'; }

time_one() {
    local bin="$1" n="$2" name="$3" tile="$4"
    local vals=()
    for r in $(seq 1 $REPS); do vals+=("$($bin $n | get_ms)"); done
    printf '%s\n' "${vals[@]}" | median | awk -v v="$name" -v n="$n" -v t="$tile" '{print v","n","t","$1}'
}

perf_one() {
    local bin="$1" n="$2"
    local out instr lmiss lload mpki
    out=$(perf stat -x, -e instructions,L1-dcache-load-misses,L1-dcache-loads "$bin" "$n" 2>&1 || true)
    instr=$(echo "$out" | awk -F, '$3=="instructions"{print $1}')
    lmiss=$(echo "$out" | awk -F, '$3=="L1-dcache-load-misses"{print $1}')
    lload=$(echo "$out" | awk -F, '$3=="L1-dcache-loads"{print $1}')
    mpki="NA"
    if [[ -n "$instr" && "$instr" != "0" ]]; then
        mpki=$(echo "$lmiss $instr" | awk '{if($2>0) printf "%.6f", $1*1000/$2; else print "NA"}')
    fi
    echo "$instr,$lmiss,$lload,$mpki"
}

echo "variant,size,tile,ms" > "$R/exec_time.csv"
for N in "${SIZES[@]}"; do
    time_one ./bin/naive    $N naive    "" >> "$R/exec_time.csv"
    time_one ./bin/reorder  $N reorder  "" >> "$R/exec_time.csv"
    time_one ./bin/unroll   $N unroll   "" >> "$R/exec_time.csv"
    time_one ./bin/loop     $N loop     "" >> "$R/exec_time.csv"
    time_one ./bin/simd_128 $N simd128  "" >> "$R/exec_time.csv"
    time_one ./bin/simd_256 $N simd256  "" >> "$R/exec_time.csv"
    time_one ./bin/simd_512 $N simd512  "" >> "$R/exec_time.csv"
    for T in "${TILES[@]}"; do
        time_one ./bin/tiling_$T $N tiling $T >> "$R/exec_time.csv"
        for W in 128 256 512; do
            time_one ./bin/combination${W}_$T $N combination$W $T >> "$R/exec_time.csv"
        done
    done
    echo "exec_time: size $N done"
done

echo "variant,size,tile,instructions,L1D_load_misses,L1D_loads,MPKI" > "$R/perf.csv"
for N in "${SIZES[@]}"; do
    echo "naive,$N,,$(perf_one ./bin/naive $N)"       >> "$R/perf.csv"
    echo "reorder,$N,,$(perf_one ./bin/reorder $N)"   >> "$R/perf.csv"
    echo "unroll,$N,,$(perf_one ./bin/unroll $N)"     >> "$R/perf.csv"
    echo "loop,$N,,$(perf_one ./bin/loop $N)"         >> "$R/perf.csv"
    echo "simd128,$N,,$(perf_one ./bin/simd_128 $N)"  >> "$R/perf.csv"
    echo "simd256,$N,,$(perf_one ./bin/simd_256 $N)"  >> "$R/perf.csv"
    echo "simd512,$N,,$(perf_one ./bin/simd_512 $N)"  >> "$R/perf.csv"
    for T in "${TILES[@]}"; do
        echo "tiling,$N,$T,$(perf_one ./bin/tiling_$T $N)" >> "$R/perf.csv"
        for W in 128 256 512; do
            echo "combination$W,$N,$T,$(perf_one ./bin/combination${W}_$T $N)" >> "$R/perf.csv"
        done
    done
    echo "perf: size $N done"
done

echo "done. see $R/exec_time.csv and $R/perf.csv"
