import torch
from options import parser
import torch.profiler
# from torch.utils.tensorboard import SummaryWriter
import horovod.torch as hvd
import os
# from tqdm import tqdm
import time
import threading
import models
import torch.backends.cudnn as cudnn
# add for cluster experiment
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trainer import Trainer
import utils
import subprocess
import math
import signal

trainer = None
def get_time(sync=False):
    if sync:
        torch.cuda.synchronize()
    return time.time()

def get_sargs():
    sargs0 = {"train_dir":args.train_dir0, "model_name":args.model0, "batch_size":args.batch_size0, "iters":args.iters0}
    sargs0["num_workers"] = args.num_workers0
    sargs0["prefetch_factor"] = args.prefetch_factor0
     
    sargs1 = {"train_dir":args.train_dir1, "model_name":args.model1, "batch_size":args.batch_size1, "iters":args.iters1}
    sargs1["num_workers"] = args.num_workers1
    sargs1["prefetch_factor"] = args.prefetch_factor1
    
    sargs2 = {"train_dir":args.train_dir2, "model_name":args.model2, "batch_size":args.batch_size2, "iters":args.iters2}
    sargs2["num_workers"] = args.num_workers2
    sargs2["prefetch_factor"] = args.prefetch_factor2

    sargs3 = {"train_dir":args.train_dir3, "model_name":args.model3, "batch_size":args.batch_size3, "iters":args.iters3}
    sargs3["num_workers"] = args.num_workers3
    sargs3["prefetch_factor"] = args.prefetch_factor3
    
    return sargs0, sargs1, sargs2, sargs3

def get_model(idx, args, sargs):
    if sargs["model_name"] == 'dqn':
        return models.DQNModel(idx, args, sargs)
    elif sargs["model_name"] == 'a2c':
        return models.A2CModel(idx, args, sargs)
    elif sargs["model_name"] == 'bert' or sargs["model_name"] == 'gpt2':
        return models.NLPModel(idx, args, sargs)
    else:
        return models.CVModel(idx, args, sargs)

class myThread(threading.Thread):
    def __init__(self, id):
        threading.Thread.__init__(self)
        self.id = id
    def run(self):
        torch.cuda.set_device(hvd.local_rank())
        self.data = get_data(self.id)
    def get_result(self):
        return self.data

def get_data(id):
    if id == 0: 
        data = model0.get_data()
    elif id==1:
        data = model1.get_data()
    elif id==2:
        data = model2.get_data()
    elif id==3:
        data = model3.get_data()
    else:
        print("Error: wrong dataloader ID.")
        exit(1)
    return data

# overlap pattern:
# D GN
# ND G
# GND 
#  GND
def train():
    time_st = time.time()
    time_all = 0.0
    cur_iter = 0
    iter_list = [sargs0['iters'], sargs1['iters'], sargs2['iters'], sargs3['iters']]
    iter_list = list(set(iter_list))
    iter_list.sort()
    if iter_list[0] == 0:
        del iter_list[0]
    print(iter_list)
    args.iters = iter_list[-1]
    tmp_iter = 0
    itertime_list = []
    last_iter = 10
    
    if hvd.rank()==0:
        secs = 50

        # start subprocess
        # gpu
        visible_device_str = os.getenv('CUDA_VISIBLE_DEVICES')
        if visible_device_str!=None:
            split_ch = ','
            visible_devices = visible_device_str.split(split_ch)
            filename = args.this_dir+"/profiling"+ visible_devices[hvd.local_rank()] +".xml"
            command = "exec nvidia-smi -q -i " + visible_devices[hvd.local_rank()] + " -x -l 1 -f " + filename
            gpu_process=subprocess.Popen(command, shell=True)
        # cpu
        cur_pid = os.getpid()
        cpu_command = "exec top -d 0.2 -bn " + str(secs) + " -p "+ str(cur_pid) +" | grep Cpu > "+ args.this_dir + "/profiling_cpu_"+str(cur_pid) +".out"
        cpu_process = subprocess.Popen(cpu_command, shell=True)

    if num_job == 1:
        while_up = args.iters
    else:
        while_up = args.iters+1
    time_io_st = time.time()
    while cur_iter < while_up:

        # ------- stage 1 -------
        # data 0
        if cur_iter<sargs0['iters']:
            thread0 = myThread(0)
            thread0.start()

        if cur_iter>0:
            # fp/bp 2
            if cur_iter-1<sargs2['iters']:
                model2.forward_backward(thread2)

            # comm 1
            if cur_iter-1<sargs1['iters']:
                model1.comm()
        

        # ------- stage 2 -------
        # data 1
        if cur_iter<sargs1['iters']:
            thread1 = myThread(1)
            thread1.start()

        if cur_iter>0:
            # fp/bp 3
            if cur_iter-1<sargs3['iters']:
                model3.forward_backward(thread3)
            
            # comm 2
            if cur_iter-1<sargs2['iters']:
                model2.comm()


        # ------- stage 3 -------
        # data 2
        if cur_iter<sargs2['iters']:
            thread2 = myThread(2)
            thread2.start()

        # fp/bp 0
        if cur_iter<sargs0['iters']:
            model0.forward_backward(thread0)

        # comm 3
        if cur_iter>0 and cur_iter-1<sargs3['iters']:
            model3.comm()


        # ------- stage 4 -------
        # data 3
        if cur_iter<sargs3['iters']:
            thread3 = myThread(3)
            thread3.start()
        
        # fp/bp 1
        if cur_iter<sargs1['iters']:
            model1.forward_backward(thread1)

        # comm 0
        if cur_iter<sargs0['iters']:
            model0.comm()

        # update iter and time
        time_end = time.time()
        cur_iter += 1
        if hvd.rank()==0:
            trainer.record(time_end-time_st)
            if cur_iter > last_iter:
                time_all += time_end-time_st
                if cur_iter >= iter_list[tmp_iter]:
                    itertime_list.append(time_all/(cur_iter - last_iter))
                    last_iter = iter_list[tmp_iter]+5
                    tmp_iter += 1
                    time_all = 0
            print(cur_iter, " time: ", time_end-time_st)
        time_st = time.time()
    time_io = time.time()-time_st

    if hvd.rank()==0:
        print("Model Info:")
        data_size = 0
        if sargs0['iters']!=0:
            model0.print_info()
            data_size += model0.data_size()
        if sargs1['iters']!=0:
            model1.print_info()
            data_size += model1.data_size()
        if sargs2['iters']!=0:
            model2.print_info()
            data_size += model2.data_size()
        if sargs3['iters']!=0:
            model3.print_info()
            data_size += model3.data_size()
        
        if num_job == 1 and len(itertime_list)==0:
            itertime_list = [0]
        print('itertime: ', itertime_list)

        # handle gpu
        if visible_device_str!=None:
            gpu_process.send_signal(signal.SIGINT)
            gpu_process.terminate()
            gpu_process.wait()

            filename = args.this_dir + "/profiling" + visible_devices[hvd.local_rank()] + ".xml"
            memory_usage, utilization = utils.parse_xml(filename)
            for i in range(len(memory_usage)):
                memory_usage[i] = int(memory_usage[i].split(' ')[0])
                utilization[i] = int(utilization[i].split(' ')[0])
            sorted_memory_usages = sorted(memory_usage)
            gpu_util_device = 0
            gpu_util_cnt = 0
            for i in range(len(memory_usage)):
                if math.isclose(memory_usage[i], sorted_memory_usages[-2], rel_tol=1e-1):
                    gpu_util_device += utilization[i]
                    gpu_util_cnt += 1
            gpu_util = gpu_util_device/gpu_util_cnt
            os.system("rm -rf "+filename)
        else:
            gpu_util = 0

        # handle cpu
        cpu_process.wait()
        cpu_util_list = []
        print(cur_pid)
        util_str_list = open(args.this_dir + "/profiling_cpu_"+ str(cur_pid) +".out", "r").read().split('\n')
        for i in range(secs):
            idle = float(util_str_list[i].split(',')[3].split()[-2])
            cpu_util_list.append(round(100.0 -idle, 3))
        start_point = int(len(cpu_util_list)*0.2)
        end_point = int(len(cpu_util_list)*0.8)
        cpu_util = sum(cpu_util_list[start_point:end_point])/(end_point-start_point)
        os.system("rm -rf "+args.this_dir+"/profiling_cpu_"+str(cur_pid)+".out")
        # print(args.this_dir+"/profiling_cpu_"+str(cur_pid)+".out")

        if itertime_list[0]!=0:
            io_read = data_size / itertime_list[0]
        else:
            io_read = data_size*while_up / time_io


        if visible_device_str!=None:
            print("gpu: ", memory_usage, utilization, gpu_util)
        print("cpu: ", cpu_util_list, cpu_util)
        print("io: ", io_read, 'kb/s')
        trainer.report_itertime(itertime_list, [gpu_util, cpu_util, io_read])


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
    if hvd.rank()==0:
        trainer = Trainer(args.scheduler_ip, args.scheduler_port, utils.get_host_ip(), args.trainer_port, [args.job_id0, args.job_id1, args.job_id2, args.job_id3])

    cudnn.benchmark = True

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(2)

    # deal with specific args
    sargs0, sargs1, sargs2, sargs3 = get_sargs()
    num_job = 0
    # time0 = time.time()
    if sargs0['iters']!=0:
        model0 = get_model(0, args, sargs0)
        model0.prepare(hvd)
        num_job += 1
    # time1 = time.time()
    if sargs1['iters']!=0:
        model1 = get_model(1, args, sargs1)
        model1.prepare(hvd)
        num_job += 1
    # time2 = time.time()
    if sargs2['iters']!=0:
        model2 = get_model(2, args, sargs2)
        model2.prepare(hvd)
        num_job += 1
    # time3 = time.time()
    if sargs3['iters']!=0:
        model3 = get_model(3, args, sargs3)
        model3.prepare(hvd)
        num_job += 1
    # time4 = time.time()
    # print(time1-time0, time2-time1, time3-time2, time4-time3)

    # if args.job_id0==0:
    #     exit(1)
    # elif args.job_id0==2:
    #     time.sleep(100)

    train()
