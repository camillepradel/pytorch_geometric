#!/bin/sh

# CPU SPECS - PHYSICAL CORES ONLY
CORES=$(lscpu | grep 'Core(s)' | cut -f2 -d':')
SOCKETS=$(lscpu | grep 'Socket(s)' | cut -f2 -d':')
TOTAL_CORES=$((SOCKETS * CORES))
echo "TOTAL_CORES:" $TOTAL_CORES
mkdir -p logs

# loop variables
declare -a HT=(0 1)
declare -a AFFINITY=(0 1)
declare -a MODELS=('gcn' 'gat' 'rgcn')
declare -a NUM_WORKERS=(0 1 2 3 4 8 12 16 20)

# inputs for the script
BATCH_SIZE=512
NUM_HIDDEN_CHANNELS=256
NUM_LAYERS=3
HETERO_NEIGHBORS=5
WARMUP=1

# for each model run benchmark in 4 configs: NO_HT+NO_AFF, NO_HT+AFF, HT+NO_AFF, HT+AFF
for nr_workers in ${NUM_WORKERS[@]}; do
    # do the math
    for ht in ${HT[@]}; do
        if [ $ht = 1 ]; then
            echo on > /sys/devices/system/cpu/smt/control
        else
            echo off > /sys/devices/system/cpu/smt/control
        fi
        for aff in ${AFFINITY[@]}; do
            for model in ${MODELS[@]}; do
                if [ $aff = 1 ] && [ $nr_workers = 0 ]; then
                    continue
                fi
                if [ $aff = 1 ]; then
                    lower=$nr_workers
                    upper=$((TOTAL_CORES - 1))
                    #export OMP_SCHEDULE=STATIC
                    #export OMP_PROC_BIND=CLOSE
                    export GOMP_CPU_AFFINITY="$(echo $lower-$upper)"
                else 
                    unset GOMP_CPU_AFFINITY
                fi

                OMP_NUM_THREADS=$((TOTAL_CORES - nr_workers))
                if [ OMP_NUM_THREADS > 64 ]; then
                    export NUMEXPR_MAX_THREADS=$TOTAL_CORES
                fi
                export OMP_NUM_THREADS=$OMP_NUM_THREADS
                
                log="logs/${model}_W${nr_workers}HT${ht}A${aff}.log"

                echo "HYPERTHREADING:" $(cat /sys/devices/system/cpu/smt/active)
                echo "AFFINITY:" $aff
                echo "GOMP_CPU_AFFINITY: " $(echo $GOMP_CPU_AFFINITY)
                echo "OMP_NUM_THREADS: " $(echo $OMP_NUM_THREADS)
                echo "NR_WORKERS: " $nr_workers
                echo "MODEL: " $model  
                echo "LOG: " $log
                
                python=$(which python)

                $python -u inference_benchmark.py --models $model --num-workers $nr_workers --eval-batch-sizes $BATCH_SIZE --num-layers $NUM_LAYERS --num-hidden-channels $NUM_HIDDEN_CHANNELS --hetero-num-neighbors $HETERO_NEIGHBORS --warmup $WARMUP --cpu_affinity $aff --use-sparse-tensor | tee $log
            done
        done
    done
done

