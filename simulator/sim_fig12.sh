#!/bin/bash
placement=("yarn") 

schedule=("antman" "multi-resource-blossom-same-gpu-unaware")

# philly trace
jobs=("trace1_pr" "trace2_pr" "trace3_pr" "trace4_pr")

setups=("n8g8")
multi_resource=4
packing_num=(2 3 4)

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
                if [ "$s" == "multi-resource-blossom-same-gpu-unaware" ]; then
                    for pn in ${packing_num[@]};do
                        log_name="${log_folder}/${s}-${p}-${pn}"
                        mkdir $log_name
                        cmd="python3 run_sim.py --cluster_spec=${cluster_spec} --print --scheme=${p} --trace_file=${job_file} --schedule=${s} --log_path=${log_name} --multi_resource ${multi_resource} --packing_num ${pn} >tmp.out"
                        # echo ${cmd} 
                        python run_sim.py --cluster_spec=${cluster_spec} --print --scheme=${p} --trace_file=${job_file} --schedule=${s} --log_path=${log_name} --multi_resource ${multi_resource} --packing_num ${pn} >${log_name}/tmp.out &
                    done
                else
                    log_name="${log_folder}/${s}-${p}"
                    mkdir $log_name
                    cmd="python3 run_sim.py --cluster_spec=${cluster_spec} --print --scheme=${p} --trace_file=${job_file} --schedule=${s} --log_path=${log_name} --multi_resource ${multi_resource} >tmp.out"
                    # echo ${cmd} 
                    python run_sim.py --cluster_spec=${cluster_spec} --print --scheme=${p} --trace_file=${job_file} --schedule=${s} --log_path=${log_name} --multi_resource ${multi_resource} >${log_name}/tmp.out &
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
                if [ "$s" == "multi-resource-blossom-same-gpu-unaware" ]; then
                    for pn in ${packing_num[@]};do
                        log_name="${log_folder}/${s}-${p}-${pn}"
                        echo ${s}-${p}-${pn}
                        python calc.py ${log_name}
                    done
                else
                    log_name="${log_folder}/${s}-${p}"
                    echo ${s}-${p}
                    python calc.py ${log_name}
                fi
            done
        done
    done
done
