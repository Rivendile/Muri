#!/bin/bash
placement=("yarn") 
# shortest -- SRTF; multi-resource-blossom-same-gpu -- Muri-S;
# dlas-gpu -- Tiresias; multi-resource-blossom-same-gpu-unaware -- Muri-L
schedule=("shortest" "multi-resource-blossom-same-gpu" "dlas-gpu" "multi-resource-blossom-same-gpu-unaware")

# philly trace
jobs=("job_type_1" "job_type_2" "job_type_3" "job_type_4")

setups=("n8g8")
multi_resource=4

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
                log_name="${log_folder}/${s}-${p}"
                mkdir $log_name
                cmd="python3 run_sim.py --cluster_spec=${cluster_spec} --print --scheme=${p} --trace_file=${job_file} --schedule=${s} --log_path=${log_name} --multi_resource ${multi_resource} >tmp.out"
                # echo ${cmd} 
                python run_sim.py --cluster_spec=${cluster_spec} --print --scheme=${p} --trace_file=${job_file} --schedule=${s} --log_path=${log_name} --multi_resource ${multi_resource}  >${log_name}/tmp.out &
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
                log_name="${log_folder}/${s}-${p}"
                python calc.py ${log_name}
            done
        done
    done
done
