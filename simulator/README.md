# Simulator

# 0. Content
- **simulator/** contains code for simulation and is adapted from [Tiresias](https://github.com/SymbioticLab/Tiresias).
  - **cluster_spec/** contains configuration files for cluster, e.g., the number of nodes, the number of GPU per node.
  - **trace-data/** contains traces for simulation evaluation.
  - **calc.py** computes metrics, e.g., avg. JCT, Makespan, and 99th JCT.
  - **cluster.py**, **switch.py**, and **node.py** contain implementations of the cluster.
  - **jobs.py** and **model.py** contain information of the jobs.
  - **flags.py** contains the argument definition method.
  - **log.py** and **utils.py** contain auxiliary functions.
  - **matching.py** contains the implementation of the matching algorithm for Muri.
  - **run_sim.py** contains the implementation of different scheduling policies.

# 1. Environment config
### Step 1: create conda environment
```
# create conda env
conda create -n muri python=3.8
conda activate muri
```

### Step 2: install python dependencies
```
conda install numpy
conda install -c conda-forge cvxpy
```

# 2. Reproduce simulation results (for SIGCOMM'22 artifact evaluation)
- ```cd <repo>/simulator```
- Figure 9: ```bash sim_fig9.sh```
- Figure 10: ```bash sim_fig10.sh```
- Figure 11: ```bash sim_fig11.sh```
- Figure 12: ```bash sim_fig12.sh```
- Figure 13: ```bash sim_fig13.sh```
- Note: The detailed results will be stored in ```<repo>/simulator/results/```.

