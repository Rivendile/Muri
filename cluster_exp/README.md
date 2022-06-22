# Testbed experiments
Note: 
- Due to the execution scripts are highly related to intracompany platform, we only demonstrate the functionality and show the pseudocode of the related scripts (e.g., run.sh, prepare_env.sh). Please adjust the scripts to your platform if you would like to execute the testbed experiment.
- Our testbed experiments were performed on 8 nodes with 8 V100 GPUs per node. For other cluster settings, please change ```setups``` in ```run.sh```.

# 0. Content
- **cluster_exp/** contains code for real-cluster experiment.
  - **cluster_spec/** contains configuration files for cluster, e.g., the number of nodes, the number of GPU per node.
  - **runtime/** contains gRPC runtime of scheduler, trainer, master, and worker.
  - **trace-data/** contains traces for testbed evaluation.
  - **workloads/** contains the implementations of DL workloads used in our evaluation.
  - **calc.py** computes metrics, e.g., avg. JCT, Makespan, and 99th JCT.
  - **cluster.py**, **switch.py**, and **node.py** contain implementations of the cluster.
  - **jobs.py** and **model.py** contain information of the jobs.
  - **flags.py** contains the argument definition method.
  - **log.py** and **utils.py** contain auxiliary functions.
  - **matching.py** contains the implementation of the matching algorithm for Muri.
  - **run.py** contains the implementation of different scheduling policies.
  - **controller.py**, **scheduler.py**, **trainer.py**, **worker.py**, and **task.py** contain the implementation of scheduler components and scheduling tasks.
  - **Makefile** prepares gRPC

# 1. Environment config
### Step 1: interconnect each node

### Step 2: create conda environment
```
# create conda env
conda create -n muri python=3.8
conda activate muri
```

### Step 3: install Open MPI
[Install Open MPI](https://www.open-mpi.org/faq/?category=building#easy-build) or other MPI implementation.

### Step 4: install python dependencies
```
# gRPC
python -m pip install grpcio
python -m pip install grpcio-tools

# prepare rpc
cd <repo>/cluster_exp
make rpc

# other dependencies
conda install numpy
conda install -c conda-forge cvxpy
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
HOROVOD_GPU_OPERATIONS=NCCL python -m pip install horovod

# dependencies for workloads
# NLP
conda install -c huggingface transformers
# RL
python -m pip install -r <repo>/cluster_exp/workloads/requirements.txt
```

### Step 5: prepare datasets (for testbed experiment)
- [Imagenet-1k](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2) for CV models.
- [Wikitext](https://huggingface.co/datasets/wikitext) for NLP models.
Store these datsets in ```<repo>/cluster_exp/datasets/```

# 2. Reproduce testbed results (for SIGCOMM'22 artifact evaluation)
- ```cd <repo>/cluster_exp```
- Table 3&4, Figure 8: ```bash run.sh <scheduler>```, ```<scheduler>``` can be set to
  - ```shortest```: SRTF
  - ```shortest-gpu```: SRSF
  - ```multi-resource-blossom-same-gpu```: Muri-S
  - ```dlas-gpu```: Tiresias
  - ```themis```: Themis
  - ```multi-resource-blossom-same-gpu-unaware```: Muri-L
- Each test takes about 1 day.

Note: We list the detailed log and evaluation results in ```<repo>/cluster_exp/results```. You can use ```python3 draw.py``` to get the figures shown in our paper.