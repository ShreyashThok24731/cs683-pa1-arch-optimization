#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

R=results
mkdir -p "$R"

REPS=3
BIN=./bin/emb

median() { sort -n | awk 'BEGIN{c=0}{a[c++]=$1}END{if(c==0)print"NA";else if(c%2)print a[int(c/2)];else print (a[c/2]+a[c/2-1])/2}'; }

run_us() {
    local run=$1; shift
    local out
    out=$("$@" $BIN)
    echo "$out" | awk -F, -v r=$run '$1==r{print $2}'
}

echo "table_size,dim,run,us" > "$R/size_sweep.csv"
for TS in 200000 500000 1000000 2000000 4000000; do
    for RUN in naive prefetch simd prefetch_simd; do
        vals=()
        for r in $(seq 1 $REPS); do
            v=$(run_us $RUN env EMB_TABLE_SIZE=$TS EMB_DIM=128 EMB_INPUT_SIZE=2048 EMB_PREFETCH_DIST=4 EMB_HINT=0 EMB_SIMD_WIDTH=256)
            vals+=("$v")
        done
        med=$(printf '%s\n' "${vals[@]}" | median)
        echo "$TS,128,$RUN,$med" >> "$R/size_sweep.csv"
    done
    echo "table_size $TS done"
done

echo "pd,run,us" > "$R/prefetch_distance.csv"
for PD in 1 2 4 8 16 32 64; do
    for RUN in prefetch prefetch_simd; do
        vals=()
        for r in $(seq 1 $REPS); do
            v=$(run_us $RUN env EMB_TABLE_SIZE=2000000 EMB_DIM=128 EMB_INPUT_SIZE=2048 EMB_PREFETCH_DIST=$PD EMB_HINT=0 EMB_SIMD_WIDTH=256)
            vals+=("$v")
        done
        med=$(printf '%s\n' "${vals[@]}" | median)
        echo "$PD,$RUN,$med" >> "$R/prefetch_distance.csv"
    done
done

echo "hint,run,us" > "$R/hint.csv"
for H in 0 1 2 3; do
    for RUN in prefetch prefetch_simd; do
        vals=()
        for r in $(seq 1 $REPS); do
            v=$(run_us $RUN env EMB_TABLE_SIZE=2000000 EMB_DIM=128 EMB_INPUT_SIZE=2048 EMB_PREFETCH_DIST=4 EMB_HINT=$H EMB_SIMD_WIDTH=256)
            vals+=("$v")
        done
        med=$(printf '%s\n' "${vals[@]}" | median)
        echo "$H,$RUN,$med" >> "$R/hint.csv"
    done
done

echo "simd_width,dim,run,us" > "$R/simd_width.csv"
for DIM in 64 128 256 512; do
    for W in 128 256 512; do
        for RUN in simd prefetch_simd; do
            vals=()
            for r in $(seq 1 $REPS); do
                v=$(run_us $RUN env EMB_TABLE_SIZE=1000000 EMB_DIM=$DIM EMB_INPUT_SIZE=2048 EMB_PREFETCH_DIST=4 EMB_HINT=0 EMB_SIMD_WIDTH=$W)
                vals+=("$v")
            done
            med=$(printf '%s\n' "${vals[@]}" | median)
            echo "$W,$DIM,$RUN,$med" >> "$R/simd_width.csv"
        done
        vals=()
        for r in $(seq 1 $REPS); do
            v=$(run_us naive env EMB_TABLE_SIZE=1000000 EMB_DIM=$DIM EMB_INPUT_SIZE=2048)
            vals+=("$v")
        done
        med=$(printf '%s\n' "${vals[@]}" | median)
        echo "$W,$DIM,naive,$med" >> "$R/simd_width.csv"
    done
done

echo "run,pd,L1D_load_misses,L2_misses,LLC_misses,sw_prefetch_access,instructions" > "$R/perf.csv"

perf_one() {
    local run=$1; local pd=$2
    local out
    out=$(EMB_TABLE_SIZE=2000000 EMB_DIM=128 EMB_INPUT_SIZE=2048 EMB_PREFETCH_DIST=$pd EMB_HINT=0 EMB_SIMD_WIDTH=256 EMB_ONLY=$3 \
        perf stat -x, -e instructions,L1-dcache-load-misses,LLC-load-misses,sw_prefetch_access.t0,sw_prefetch_access.t1_t2,sw_prefetch_access.nta $BIN 2>&1 || true)
    local instr=$(echo "$out" | awk -F, '$3=="instructions"{print $1}')
    local l1m=$(echo "$out"   | awk -F, '$3=="L1-dcache-load-misses"{print $1}')
    local llcm=$(echo "$out"  | awk -F, '$3=="LLC-load-misses"{print $1}')
    local pt0=$(echo "$out"   | awk -F, '$3=="sw_prefetch_access.t0"{print $1}')
    local pt12=$(echo "$out"  | awk -F, '$3=="sw_prefetch_access.t1_t2"{print $1}')
    local pnta=$(echo "$out"  | awk -F, '$3=="sw_prefetch_access.nta"{print $1}')
    local sw=$(echo "$pt0 $pt12 $pnta" | awk '{print $1+$2+$3}')
    echo "$run,$pd,$l1m,,$llcm,$sw,$instr"
}
perf_one naive         0 1 >> "$R/perf.csv"
perf_one prefetch      4 2 >> "$R/perf.csv"
perf_one simd          0 3 >> "$R/perf.csv"
perf_one prefetch_simd 4 4 >> "$R/perf.csv"

echo "done. see results/"
