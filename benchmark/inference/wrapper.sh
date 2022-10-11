#!/bin/sh
# CPU SPECS - PHYSICAL CORES ONLY
SOCKETS=2
CORES=56
TOTAL_CORES=$SOCKETS*$CORES

# loop variables
HT=(0 1)
AFFINITY=(0 1)
MODELS=('gcn' 'gat' 'rgcn')
NUM_WORKERS=(0 1 2 3 4 8 12 16 20 24)

# inputs for the script
BATCH_SIZE=256
NUM_HIDDEN_CHANNELS=16
NUM_LAYERS=2
HETERO_NEIGHBORS=3
WARMUP=0
# for each model run benchmark in 4 configs: NO_HT+NO_AFF, NO_HT+AFF, HT+NO_AFF, HT+AFF
for nr_workers in ${NUM_WORKERS[@]}; do
    # do the math
    for model in ${MODELS[@]}; do
        for ht in ${HT[@]}; do
            if [ $ht = 1 ]; then
                echo on > /sys/devices/system/cpu/smt/control
            else
                echo off > /sys/devices/system/cpu/smt/control
            fi
            echo "HYPERTHREADING:" $(cat /sys/devices/system/cpu/smt/active)
            for aff in ${AFFINITY[@]}; do
            echo "AFFINITY:" $aff
                if [ $aff = 1 ] && [ $nr_workers = 0 ]; then
                    echo "skip"
                    continue
                fi
                if [ $aff = 1 ]; then
                    lower=$nr_workers-1
                    upper=$TOTAL_CORES-1

                    export OMP_SCHEDULE=STATIC
                    export OMP_PROC_BIND=CLOSE
                    export GOMP_CPU_AFFINITY="${lower}-${upper}"
                    echo "GOMP_CPU_AFFINITY: " $GOMP_CPU_AFFINITY
                fi
                omp=$TOTAL_CORES-$nr_workers
                export OMP_NUM_THREADS=$omp
                log="${model}_HT${ht}A${aff}W${nr_workers}.log"

                echo "OMP_NUM_THREADS: " $OMP_NUM_THREADS
                echo "NR_WORKERS: " $model
                echo "MODEL: " $model  
                echo "LOG: " $log
                
                /home/sdp/miniconda3/envs/pyg/bin/python inference_benchmark.py --models $model --num-workers $nr_workers --eval-batch-sizes $BATCH_SIZE --num-layers $NUM_LAYERS --num-hidden-channels $NUM_HIDDEN_CHANNELS --hetero-num-neighbors $HETERO_NEIGHBORS --warmup $WARMUP --cpu_affinity $aff --use-sparse-tensor | tee $log

            done
        done
    done
done

