#!/bin/bash
# Note: Due to the scripts are highly related to intracompany platform, 
#we only demonstrate the functionality and show the pseudocode of the 
#related scripts (e.g., run.sh, prepare_env.sh). Please adjust to your 
#platform if you would like to execute the testbed experiment.

if [ $# -lt 33 ]; then
	echo "usage: model_0, batch-size_0, num_workers0, prefetch_factor0, train_dir0, iter0, job_id0, job_counter0, model_1, batch-size_1, num_workers1, prefetch_factor1, train_dir1, iter1, job_id1, job_counter1, model_2, batch-size_2, num_workers2, prefetch_factor2, train_dir2, iter2, job_id2, job_counter2, model_3, batch-size_3, num_workers3, prefetch_factor3, train_dir3, iter3, job_id3, job_counter3, num_gpu, other_params"
	exit -1;
fi
 
export MODEL0=$1
shift
export BS0=$1
shift
export NUM_WORKERS0=$1
shift
export PREFETCH_FACTOR0=$1
shift
export TRAIN_DIR0=$1
shift
export ITER0=$1
shift
export JOB_ID0=$1
shift
export JOB_COUNTER0=$1
shift
export MODEL1=$1
shift
export BS1=$1
shift
export NUM_WORKERS1=$1
shift
export PREFETCH_FACTOR1=$1
shift
export TRAIN_DIR1=$1
shift
export ITER1=$1
shift
export JOB_ID1=$1
shift
export JOB_COUNTER1=$1
shift
export MODEL2=$1
shift
export BS2=$1
shift
export NUM_WORKERS2=$1
shift
export PREFETCH_FACTOR2=$1
shift
export TRAIN_DIR2=$1
shift
export ITER2=$1
shift
export JOB_ID2=$1
shift
export JOB_COUNTER2=$1
shift
export MODEL3=$1
shift
export BS3=$1
shift
export NUM_WORKERS3=$1
shift
export PREFETCH_FACTOR3=$1
shift
export TRAIN_DIR3=$1
shift
export ITER3=$1
shift
export JOB_ID3=$1
shift
export JOB_COUNTER3=$1
shift
export NUM_GPU=$1
shift

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $THIS_DIR

#get real datasets -- imagenet-1k
judge_path="$THIS_DIR/datasets/imagenet"

#get nlp datasets - wikitext
TRAIN_FILE=$THIS_DIR/datasets/wikitext-2-raw/wiki.train.raw

arg="$@"
echo $arg

if [[ "$MODEL1" == "-1" ]]; then
    MODEL1=$MODEL0
fi
if [ $NUM_WORKERS1 -eq -1 ]; then
    NUM_WORKERS1=$NUM_WORKERS0
fi
if [ $PREFETCH_FACTOR1 -eq -1 ]; then
    PREFETCH_FACTOR1=$PREFETCH_FACTOR0
fi
if [ $BS1 -eq -1 ]; then
    BS1=$BS0
fi

if [[ "$MODEL2" == "-1" ]]; then
    MODEL2=$MODEL0
fi
if [ $NUM_WORKERS2 -eq -1 ]; then
    NUM_WORKERS2=$NUM_WORKERS0
fi
if [ $PREFETCH_FACTOR2 -eq -1 ]; then
    PREFETCH_FACTOR2=$PREFETCH_FACTOR0
fi
if [ $BS2 -eq -1 ]; then
    BS2=$BS0
fi

if [[ "$MODEL3" == "-1" ]]; then
    MODEL3=$MODEL0
fi
if [ $NUM_WORKERS3 -eq -1 ]; then
    NUM_WORKERS3=$NUM_WORKERS0
fi
if [ $PREFETCH_FACTOR3 -eq -1 ]; then
    PREFETCH_FACTOR3=$PREFETCH_FACTOR0
fi
if [ $BS3 -eq -1 ]; then
    BS3=$BS0
fi

# train data path
if [[ "$TRAIN_DIR0" == "-1" ]]; then
    if [[ "$MODEL0" == "dqn" ]] || [[ "$MODEL0" == "a2c" ]]; then
        TRAIN_DIR0="./"
    elif [[ "$MODEL0" == "bert" ]] || [[ "$MODEL0" == "gpt2" ]]; then
        TRAIN_DIR0=$TRAIN_FILE
    else
        TRAIN_DIR0=$judge_path
    fi
fi
if [[ "$TRAIN_DIR1" == "-1" ]]; then
    if [[ "$MODEL1" == "dqn" ]] || [[ "$MODEL1" == "a2c" ]]; then
        TRAIN_DIR1="./"
    elif [[ "$MODEL1" == "bert" ]] || [[ "$MODEL1" == "gpt2" ]]; then
        TRAIN_DIR1=$TRAIN_FILE
    else
        TRAIN_DIR1=$judge_path
    fi
fi
if [[ "$TRAIN_DIR2" == "-1" ]]; then
    if [[ "$MODEL2" == "dqn" ]] || [[ "$MODEL2" == "a2c" ]]; then
        TRAIN_DIR2="./"
    elif [[ "$MODEL2" == "bert" ]] || [[ "$MODEL2" == "gpt2" ]]; then
        TRAIN_DIR2=$TRAIN_FILE
    else
        TRAIN_DIR2=$judge_path
    fi
fi
if [[ "$TRAIN_DIR3" == "-1" ]]; then
    if [[ "$MODEL3" == "dqn" ]] || [[ "$MODEL3" == "a2c" ]]; then
        TRAIN_DIR3="./"
    elif [[ "$MODEL3" == "bert" ]] || [[ "$MODEL3" == "gpt2" ]]; then
        TRAIN_DIR3=$TRAIN_FILE
    else
        TRAIN_DIR3=$judge_path
    fi
fi

hostfile=$THIS_DIR/hostfiles/hostfile-[${JOB_ID0}-${JOB_ID1}-${JOB_ID2}-${JOB_ID3}]-[${JOB_COUNTER0}-${JOB_COUNTER1}-${JOB_COUNTER2}-${JOB_COUNTER3}]
echo $hostfile

# set common command for mpirun
COMMON_CMD=""

if [ $NUM_GPU -ge 8 ]; then
    GPU_PERNODE=8
else
    GPU_PERNODE=$NUM_GPU
fi

echo $NUM_GPU $GPU_PERNODE $COMMON_CMD

echo "-------------------------------"
echo $MODEL0 $BS0 $MODEL1 $BS1 $MODEL2 $BS2 $MODEL3 $BS3

exec mpirun -n $NUM_GPU --npernode $GPU_PERNODE ${COMMON_CMD} \
    python3 $THIS_DIR/main_real_preenv.py --model0 $MODEL0 --batch-size0 $BS0 --train-dir0 $TRAIN_DIR0 --num-workers0 ${NUM_WORKERS0} --prefetch-factor0 ${PREFETCH_FACTOR0} --iters0 $ITER0 --job-id0 $JOB_ID0 --model1 $MODEL1 --batch-size1 $BS1 --train-dir1 $TRAIN_DIR1 --num-workers1 ${NUM_WORKERS1} --prefetch-factor1 ${PREFETCH_FACTOR1} --iters1 $ITER1 --job-id1 $JOB_ID1 --model2 $MODEL2 --batch-size2 $BS2 --train-dir2 $TRAIN_DIR2 --num-workers2 ${NUM_WORKERS2} --prefetch-factor2 ${PREFETCH_FACTOR2} --iters2 $ITER2 --job-id2 $JOB_ID2 --model3 $MODEL3 --batch-size3 $BS3 --train-dir3 $TRAIN_DIR3 --num-workers3 ${NUM_WORKERS3} --prefetch-factor3 ${PREFETCH_FACTOR3} --iters3 $ITER3 --job-id3 $JOB_ID3 --this-dir $THIS_DIR $arg >$THIS_DIR/test.txt 

