#!/bin/bash
placement=("yarn") 
#schedule=("fifo" "fjf" "sjf" "shortest" "shortest-gpu" "dlas" "dlas-gpu")
#schedule=("dlas" "dlas-gpu" "dlas-gpu-100" "dlas-gpu-8" "dlas-gpu-4" "dlas-gpu-2" "dlas-gpu-1" "dlas-gpu-05")
# schedule=("dlas-gpu")

# schedule=("shortest" "shortest-gpu" "multi-resource-blossom-same-gpu" "dlas-gpu" "antman" "multi-resource-blossom-same-gpu-unaware" )
# schedule=("multi-resource-blossom-same-gpu-unaware" "multi-resource-blossom-same-gpu-unaware-worstordering" "multi-resource-gpu-unaware")
# schedule=("shortest" "multi-resource-blossom-same-gpu" "dlas-gpu" "multi-resource-blossom-same-gpu-unaware")
schedule=("multi-resource-blossom-same-gpu-unaware-dlas" "dlas-gpu")
# schedule=("dlas-gpu")
# schedule=("dlas" "dlas-gpu" "multi-resource-same-unaware" "multi-resource-same-gpu-unaware" )
# schedule=("shortest-gpu")
# schedule=("dlas" "dlas-gpu")
# schedule=("dlas-gpu-05")
# schedule=("dlas-gpu-1" "dlas-gpu-2" "dlas-gpu-4" "dlas-gpu-8" "dlas-gpu-10" "dlas-gpu-100" "dlas-gpu-1000")
# schedule=("dlas-gpu" "fifo")

# philly trace
# jobs=("new_philly_traces_7f04ca" "new_philly_traces_7f04ca_0" "new_philly_traces_ee9e8c" "new_philly_traces_ee9e8c_0" "new_philly_traces_2869ce" "new_philly_traces_2869ce_0" "new_philly_traces_0e4a51" "new_philly_traces_0e4a51_0")
# jobs=("new_philly_traces_7f04ca" "new_philly_traces_7f04ca_per-1-0" "new_philly_traces_7f04ca_per-1-1" "new_philly_traces_7f04ca_per-1-2" "new_philly_traces_7f04ca_per-1-3" "new_philly_traces_7f04ca_per-2-0" "new_philly_traces_7f04ca_per-2-1" "new_philly_traces_7f04ca_per-2-2" "new_philly_traces_7f04ca_per-2-3" "new_philly_traces_7f04ca_per-2-4" "new_philly_traces_7f04ca_per-2-5" "new_philly_traces_7f04ca_per-3-0" "new_philly_traces_7f04ca_per-3-1" "new_philly_traces_7f04ca_per-3-2" "new_philly_traces_7f04ca_per-3-3")
# jobs=("new_philly_traces_2869ce_per-1-0" "new_philly_traces_2869ce_per-1-1" "new_philly_traces_2869ce_per-1-2" "new_philly_traces_2869ce_per-1-3" "new_philly_traces_2869ce_per-2-0" "new_philly_traces_2869ce_per-2-1" "new_philly_traces_2869ce_per-2-2" "new_philly_traces_2869ce_per-2-3" "new_philly_traces_2869ce_per-2-4" "new_philly_traces_2869ce_per-2-5" "new_philly_traces_2869ce_per-3-0" "new_philly_traces_2869ce_per-3-1" "new_philly_traces_2869ce_per-3-2" "new_philly_traces_2869ce_per-3-3")
jobs=("new_philly_traces_2869ce_per-1-2" "new_philly_traces_2869ce")
# jobs=("philly_traces_b436b2")
# jobs=("60_job_fix_modified")
#philly trace - submit at time 0
#jobs=("philly_traces_7f04ca_0" "philly_traces_6214e9_0" "philly_traces_ee9e8c_0" "philly_traces_b436b2_0" "philly_traces_ed69ec_0" "philly_traces_e13805_0" "philly_traces_103959_0" "philly_traces_6c71a0_0" "philly_traces_2869ce_0" "philly_traces_11cb48_0" "philly_traces_0e4a51_0" )
# jobs=("philly_traces_7f04ca_0" "philly_traces_ed69ec_0" "philly_traces_e13805_0" "philly_traces_2869ce_0")
# cluster experiments
# jobs=("cluster_philly_traces_7f04ca" "cluster_philly_traces_6214e9" "cluster_philly_traces_ee9e8c" "cluster_philly_traces_b436b2" "cluster_philly_traces_ed69ec" "cluster_philly_traces_e13805" "cluster_philly_traces_103959" "cluster_philly_traces_6c71a0" "cluster_philly_traces_2869ce" "cluster_philly_traces_11cb48" "cluster_philly_traces_0e4a51")
# jobs=("cluster_philly_traces_2869ce")
# jobs=("60_job_fix")
setups=("n8g8")
multi_resource=4
packing_num=(2 4)

mkdir results
echo "running..."
for setup in ${setups[@]};do
    cluster_spec="cluster_spec/${setup}.csv"
    for job in ${jobs[@]};do
        job_file="trace-data/${job}.csv"
        log_folder="results/${setup}j${job}"
        mkdir ${log_folder}
        # echo ${job}
        for p in ${placement[@]};do
            for s in ${schedule[@]};do
                if [ "$s" == "multi-resource-blossom-same-gpu-unaware" ] || [ "$s" == "multi-resource-blossom-same-gpu" ]; then
                    for pn in ${packing_num[@]};do
                        log_name="${log_folder}/${s}-${p}-${pn}"
                        mkdir $log_name
                        cmd="python3 run_sim.py --cluster_spec=${cluster_spec} --print --scheme=${p} --trace_file=${job_file} --schedule=${s} --log_path=${log_name} --multi_resource ${multi_resource} --packing_num ${pn} >tmp.out"
                        # echo ${cmd} 
                        python3 run_sim.py --cluster_spec=${cluster_spec} --print --scheme=${p} --trace_file=${job_file} --schedule=${s} --log_path=${log_name} --multi_resource ${multi_resource} --packing_num ${pn} >${log_name}/tmp.out &
                    done
                else
                    log_name="${log_folder}/${s}-${p}"
                    mkdir $log_name
                    cmd="python3 run_sim.py --cluster_spec=${cluster_spec} --print --scheme=${p} --trace_file=${job_file} --schedule=${s} --log_path=${log_name} --multi_resource ${multi_resource} >tmp.out"
                    # echo ${cmd} 
                    python3 run_sim.py --cluster_spec=${cluster_spec} --print --scheme=${p} --trace_file=${job_file} --schedule=${s} --log_path=${log_name} --multi_resource ${multi_resource} >${log_name}/tmp.out &
                fi
            done
        done
    done
done

wait
echo "calc..."
for setup in ${setups[@]};do
    cluster_spec="cluster_spec/${setup}.csv"
    for job in ${jobs[@]};do
        job_file="trace-data/${job}.csv"
        log_folder="results/${setup}j${job}"
        echo ${job}
        for p in ${placement[@]};do
            for s in ${schedule[@]};do
                echo $s
                if [ "$s" == "multi-resource-blossom-same-gpu-unaware" ] || [ "$s" == "multi-resource-blossom-same-gpu" ]; then
                    for pn in ${packing_num[@]};do
                        log_name="${log_folder}/${s}-${p}-${pn}"
                        echo ${s}-${p}-${pn}
                        python3 calc.py ${log_name}
                    done
                else
                    log_name="${log_folder}/${s}-${p}"
                    echo ${s}-${p}
                    python3 calc.py ${log_name}
                fi
            done
        done
    done
done
