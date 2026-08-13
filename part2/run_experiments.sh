#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

R=results
mkdir -p "$R"

REPS=3
INPUT=2048
BIN=./bin/emb

median() { sort -n | awk 'BEGIN{c=0}{a[c++]=$1}END{if(c==0)print"NA";else if(c%2)print a[int(c/2)];else print (a[c/2]+a[c/2-1])/2}'; }

run_us() {
    local run=$1; shift
    local out
    out=$("$@" $BIN)
    echo "$out" | awk -F, -v r=$run '$1==r{print $2}'
}

only_of() {
    case $1 in
        naive)         echo 1;;
        prefetch)      echo 2;;
        simd)          echo 3;;
        prefetch_simd) echo 4;;
    esac
}

# Counters are gated to the kernel region via perf's --control fifo, so the
# ~1 GB embedding-table initialisation is excluded from every count.
perf_row() {
    local sweep=$1 run=$2 ts=$3 dim=$4 pd=$5 hint=$6 w=$7
    local only ctl ack out1 out2 us instr l1 l2 llc t0 t12 nta sw

    only=$(only_of $run)

    ctl=$(mktemp -u /tmp/embctl.XXXXXX); ack=$(mktemp -u /tmp/emback.XXXXXX)
    mkfifo "$ctl" "$ack"
    out1=$(EMB_TABLE_SIZE=$ts EMB_DIM=$dim EMB_INPUT_SIZE=$INPUT EMB_PREFETCH_DIST=$pd \
           EMB_HINT=$hint EMB_SIMD_WIDTH=$w EMB_ONLY=$only \
           EMB_CTL_FIFO=$ctl EMB_ACK_FIFO=$ack \
           perf stat -x, --delay=-1 --control fifo:$ctl,$ack \
           -e instructions,L1-dcache-load-misses,l2_rqsts.demand_data_rd_miss,LLC-load-misses \
           $BIN 2>&1 || true)
    rm -f "$ctl" "$ack"

    ctl=$(mktemp -u /tmp/embctl.XXXXXX); ack=$(mktemp -u /tmp/emback.XXXXXX)
    mkfifo "$ctl" "$ack"
    out2=$(EMB_TABLE_SIZE=$ts EMB_DIM=$dim EMB_INPUT_SIZE=$INPUT EMB_PREFETCH_DIST=$pd \
           EMB_HINT=$hint EMB_SIMD_WIDTH=$w EMB_ONLY=$only \
           EMB_CTL_FIFO=$ctl EMB_ACK_FIFO=$ack \
           perf stat -x, --delay=-1 --control fifo:$ctl,$ack \
           -e sw_prefetch_access.t0,sw_prefetch_access.t1_t2,sw_prefetch_access.nta \
           $BIN 2>&1 || true)
    rm -f "$ctl" "$ack"

    us=$(echo    "$out1" | awk -F, -v r=$run '$1==r{print $2}')
    instr=$(echo "$out1" | awk -F, '$3=="instructions"{print $1}')
    l1=$(echo    "$out1" | awk -F, '$3=="L1-dcache-load-misses"{print $1}')
    l2=$(echo    "$out1" | awk -F, '$3=="l2_rqsts.demand_data_rd_miss"{print $1}')
    llc=$(echo   "$out1" | awk -F, '$3=="LLC-load-misses"{print $1}')
    t0=$(echo    "$out2" | awk -F, '$3=="sw_prefetch_access.t0"{print $1}')
    t12=$(echo   "$out2" | awk -F, '$3=="sw_prefetch_access.t1_t2"{print $1}')
    nta=$(echo   "$out2" | awk -F, '$3=="sw_prefetch_access.nta"{print $1}')
    sw=$(echo "${t0:-0} ${t12:-0} ${nta:-0}" | awk '{print $1+$2+$3}')

    echo "$sweep,$run,$ts,$dim,$pd,$hint,$w,$instr,$l1,$l2,$llc,$sw,$us"
}

echo "table_size,dim,run,us" > "$R/size_sweep.csv"
for TS in 200000 500000 1000000 2000000 4000000; do
    for RUN in naive prefetch simd prefetch_simd; do
        vals=()
        for r in $(seq 1 $REPS); do
            v=$(run_us $RUN env EMB_TABLE_SIZE=$TS EMB_DIM=128 EMB_INPUT_SIZE=$INPUT EMB_PREFETCH_DIST=4 EMB_HINT=0 EMB_SIMD_WIDTH=256)
            vals+=("$v")
        done
        med=$(printf '%s\n' "${vals[@]}" | median)
        echo "$TS,128,$RUN,$med" >> "$R/size_sweep.csv"
    done
    echo "size_sweep: table_size $TS done"
done

echo "pd,run,us" > "$R/prefetch_distance.csv"
for PD in 1 2 4 8 16 32 64; do
    for RUN in prefetch prefetch_simd; do
        vals=()
        for r in $(seq 1 $REPS); do
            v=$(run_us $RUN env EMB_TABLE_SIZE=2000000 EMB_DIM=128 EMB_INPUT_SIZE=$INPUT EMB_PREFETCH_DIST=$PD EMB_HINT=0 EMB_SIMD_WIDTH=256)
            vals+=("$v")
        done
        med=$(printf '%s\n' "${vals[@]}" | median)
        echo "$PD,$RUN,$med" >> "$R/prefetch_distance.csv"
    done
done
echo "prefetch_distance sweep done"

echo "hint,run,us" > "$R/hint.csv"
for H in 0 1 2 3; do
    for RUN in prefetch prefetch_simd; do
        vals=()
        for r in $(seq 1 $REPS); do
            v=$(run_us $RUN env EMB_TABLE_SIZE=2000000 EMB_DIM=128 EMB_INPUT_SIZE=$INPUT EMB_PREFETCH_DIST=4 EMB_HINT=$H EMB_SIMD_WIDTH=256)
            vals+=("$v")
        done
        med=$(printf '%s\n' "${vals[@]}" | median)
        echo "$H,$RUN,$med" >> "$R/hint.csv"
    done
done
echo "hint sweep done"

# Dim sweep uses a 200k-row table so that dim=512 stays at ~400 MB.
echo "simd_width,dim,run,us" > "$R/simd_width.csv"
for DIM in 64 128 256 512; do
    for W in 128 256 512; do
        for RUN in simd prefetch_simd; do
            vals=()
            for r in $(seq 1 $REPS); do
                v=$(run_us $RUN env EMB_TABLE_SIZE=200000 EMB_DIM=$DIM EMB_INPUT_SIZE=$INPUT EMB_PREFETCH_DIST=4 EMB_HINT=0 EMB_SIMD_WIDTH=$W)
                vals+=("$v")
            done
            med=$(printf '%s\n' "${vals[@]}" | median)
            echo "$W,$DIM,$RUN,$med" >> "$R/simd_width.csv"
        done
        vals=()
        for r in $(seq 1 $REPS); do
            v=$(run_us naive env EMB_TABLE_SIZE=200000 EMB_DIM=$DIM EMB_INPUT_SIZE=$INPUT)
            vals+=("$v")
        done
        med=$(printf '%s\n' "${vals[@]}" | median)
        echo "$W,$DIM,naive,$med" >> "$R/simd_width.csv"
    done
    echo "simd_width sweep: dim $DIM done"
done

echo "sweep,run,table_size,dim,pd,hint,simd_width,instructions,L1D_misses,L2_misses,LLC_misses,sw_prefetch,us" > "$R/perf.csv"

for TS in 200000 1000000 2000000; do
    for RUN in naive prefetch simd prefetch_simd; do
        perf_row table_size $RUN $TS 128 4 0 256 >> "$R/perf.csv"
    done
    echo "perf table_size $TS done"
done

for PD in 1 4 16 64; do
    for RUN in prefetch prefetch_simd; do
        perf_row pd $RUN 1000000 128 $PD 0 256 >> "$R/perf.csv"
    done
done
echo "perf pd sweep done"

for H in 0 1 2 3; do
    for RUN in prefetch prefetch_simd; do
        perf_row hint $RUN 1000000 128 4 $H 256 >> "$R/perf.csv"
    done
done
echo "perf hint sweep done"

for DIM in 64 128 256 512; do
    perf_row dim_width naive 200000 $DIM 4 0 256 >> "$R/perf.csv"
    for W in 128 256 512; do
        for RUN in simd prefetch_simd; do
            perf_row dim_width $RUN 200000 $DIM 4 0 $W >> "$R/perf.csv"
        done
    done
    echo "perf dim_width dim $DIM done"
done

echo "done. see $R/"
