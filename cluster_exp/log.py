from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import csv
import math

import utils
import flags 
import cluster 
import jobs
import time

FLAGS = flags.FLAGS
CLUSTER = cluster.CLUSTER
JOBS = jobs.JOBS



class _Log(object):

    def __init__(self):
        self.log_path = ''
        self.log_file = ''
        self.log_cpu = ''
        self.log_gpu = ''
        self.log_network = ''
        self.log_mem = ''
        self.log_job = ''
        self.log_list = list()
        self.cpu_list = list()
        self.gpu_list = list()
        self.network_list = list()
        self.job_list = list()
        self.mem_list = list()
        self._start_time = 0
        self._gpu_util = 0
        self._cpu_util = 0
        self._io_speed = 0

    def init_log(self):
        self.log_path = FLAGS.log_path
        if self.log_path[-1] == '/':
            self.log_path = self.log_path[:-1]
        utils.print_fn(self.log_path)
        utils.print_fn(' ')

        #prepare folder
        cmd = 'mkdir -p ' + self.log_path
        ''' python 2.7
        status, output = commands.getstatusoutput(cmd)
        '''
        #python 2.7 & 3
        ret = subprocess.check_output(cmd, shell=True)

        self.log_file = self.log_path + '/cluster.csv'
        self.log_job = self.log_path + '/job.csv'
        if FLAGS.scheme != 'count':
            self.log_cpu = self.log_path + '/cpu.csv'
            self.log_gpu = self.log_path + '/gpu.csv'
            self.log_network = self.log_path + '/network.csv'
            self.log_mem = self.log_path + '/memory.csv'

        fd = open(self.log_file, 'w+')
        log_writer = csv.writer(fd)  
        if FLAGS.scheme == 'gandiva':
            log_writer.writerow(['time', 'idle_node', 'busy_node', 'full_node', 'fra_gpu', 'busy_gpu', 'pending_job', 'running_job', 'completed_job', 'len_g1', 'len_g2', 'len_g4', 'len_g8', 'len_g16', 'len_g32', 'len_g64'])
        else:
            log_writer.writerow(['time', 'queue_length', 'blocking_index', 'gpu_util', 'cpu_util', 'io_read_speed'])
        fd.close()
    

        if FLAGS.scheme != 'count':
            fd = open(self.log_cpu, 'w+')
            log_writer = csv.writer(fd)  
            log_writer.writerow(['time'] + ['cpu'+str(i) for i in range(CLUSTER.num_node)])
            ''''if combine all the info together
            log_writer.writerow(['cpu'+str(i) for i in range(CLUSTER.num_node)] 
                                + ['gpu'+str(i) for i in range(CLUSTER.num_node)] 
                                + ['net'+str(i) for i in range(CLUSTER.num_node)])
            '''
            fd.close()
            fd = open(self.log_gpu, 'w+')
            log_writer = csv.writer(fd)  
            log_writer.writerow(['time'] + ['gpu'+str(i) for i in range(CLUSTER.num_node)])
            fd.close()
            fd = open(self.log_network, 'w+')
            log_writer = csv.writer(fd)  
            title_list = list()
            title_list.append('time')
            for i in range(CLUSTER.num_node):
                title_list.append('in'+str(i))
                title_list.append('out'+str(i))
            log_writer.writerow(title_list)
            # log_writer.writerow(['net'+str(i) for i in range(CLUSTER.num_node)])
            fd.close()

            fd = open(self.log_mem, 'w+')
            log_writer = csv.writer(fd)  
            # log_writer.writerow(['time'] + ['mem'+str(i) for i in range(CLUSTER.num_node)])
            log_writer.writerow(['time', 'max', '99th', '95th', 'med'])
            fd.close()
            
        fd = open(self.log_job, 'w+')
        log_writer = csv.writer(fd)  
        if FLAGS.schedule == 'gpu-demands':
            log_writer.writerow(['time', '1-GPU', '2-GPU', '4-GPU', '8-GPU', '12-GPU', '16-GPU', '24-GPU', '32-GPU'])
        else:
            if FLAGS.scheme == 'count':
                log_writer.writerow(['time', 'job_id', 'num_gpu', 'submit_time', 'start_time', 'end_time', 'executed_time', 'real_executed_time', 'JCT', 'duration', 'pending_time', 'preempt', 'resume', 'promote'])
            else:
                log_writer.writerow(['time', 'job_id', 'num_gpu', 'submit_time', 'start_time', 'end_time', 'executed_time', 'real_executed_time', 'JCT', 'duration', 'pending_time', 'job_counter', 'promote'])
        fd.close()

        self._start_time = time.time()


    def dump_all_logs(self):
        fd = open(self.log_file, 'a+')
        log_writer = csv.writer(fd)  
        for log in self.log_list:
            log_writer.writerow(log)
        fd.close()
        del self.log_list[:]

        if FLAGS.scheme != 'count':
            fd = open(self.log_cpu, 'a+')
            log_writer = csv.writer(fd)  
            for log in self.cpu_list:
                log_writer.writerow(log)
            fd.close()
            del self.cpu_list[:]

            fd = open(self.log_gpu, 'a+')
            log_writer = csv.writer(fd)  
            for log in self.gpu_list:
                log_writer.writerow(log)
            fd.close()
            del self.gpu_list[:]

            fd = open(self.log_network, 'a+')
            log_writer = csv.writer(fd)  
            for log in self.network_list:
                log_writer.writerow(log)
            fd.close()
            del self.network_list[:]

            fd = open(self.log_mem, 'a+')
            log_writer = csv.writer(fd)  
            for log in self.mem_list:
                log_writer.writerow(log)
            fd.close()
            del self.mem_list[:]

    def gandiva_checkpoint(self, event_time, idle_node, busy_gpu, frag_gpu, pending_job, running_job, len_g1, len_g2, len_g4, len_g8, len_g16, len_g32, len_g64):
        busy_node = CLUSTER.num_node - idle_node
        full_node = 0
        idle_gpu = frag_gpu
        completed_job = len(JOBS.completed_jobs)
        self.log_list.append([event_time, idle_node, busy_node, full_node, idle_gpu, busy_gpu, pending_job, running_job, completed_job, len_g1, len_g2, len_g4, len_g8, len_g16, len_g32, len_g64])
        if len(self.log_list) >= 1:
            self.dump_all_logs()

    def checkpoint(self, event_time, scheduler, new_util=False, secs=20):
        '''
        Record cluster, and job information, including:
        time
        queue length 
        blocking index
        gpu util
        cpu util
        io read speed
        '''
        queue_length = 0
        blocking_index = 0
        for rjob in JOBS.runnable_jobs:
            if rjob['status'] == 'PENDING':
                queue_length += 1
                blocking_index += rjob['pending_time']/(rjob['remaining_iterations']*rjob['iteration_time'])
        if queue_length>0:
            blocking_index /= queue_length
        if new_util:
            self._gpu_util, self._cpu_util, self._io_speed = scheduler._controller.get_util(secs)
        self.log_list.append([event_time, queue_length, blocking_index, self._gpu_util, self._cpu_util, self._io_speed])
        if len(self.log_list) >= 1:
            self.dump_all_logs()

    def checkpoint_utils(self, event_time, scheduler):
        '''
        Record cluster, and job information, including:
        time
        queue length 
        blocking index
        gpu util
        cpu util
        io read speed
        '''
        queue_length = 0
        blocking_index = 0
        for rjob in JOBS.runnable_jobs:
            if rjob['status'] == 'PENDING':
                queue_length += 1
                blocking_index += rjob['pending_time']/(rjob['remaining_iterations']*rjob['iteration_time'])
        if queue_length>0:
            blocking_index /= queue_length
        self.log_list.append([event_time, queue_length, blocking_index, scheduler._src_utils[0]/CLUSTER.num_gpu, scheduler._src_utils[1]/CLUSTER.num_node, scheduler._src_utils[2]/CLUSTER.num_node])
        if len(self.log_list) >= 1:
            self.dump_all_logs()


    def checkpoint_multi_dlas_gpu(self, event_time):
        '''
        Record cluster, and job information, including:
        time
        idle_node
        busy_node: gpu running
        full_node: all gpus are running
        idle_gpu
        busy_gpu
        pending_job
        running_job
        completed_job
        '''
        idle_node = 0
        busy_node = 0
        full_node = 0
        idle_gpu = 0
        busy_gpu = 0
        pending_job = 0
        running_job = 0
        completed_job = 0

        if FLAGS.schedule != 'multi-dlas-gpu':
            utils.print_fn("Error, not multi-dlas-gpu in checkpoint")
            exit()

        for num_gpu, gjob in JOBS.gpu_job.items():
            idle_gpu += gjob.free_gpu

        busy_gpu = CLUSTER.num_gpu - idle_gpu

        busy_node = int(math.ceil(busy_gpu / CLUSTER.num_gpu_p_node))
        full_node = busy_node
        idle_node = int(CLUSTER.num_node - busy_node)

        for job in JOBS.job_list:
            if job['status'] == 'RUNNING':
                running_job += 1
            elif job['status'] == 'PENDING':
                pending_job += 1
            elif job['status'] == 'END':
                completed_job += 1

        #add log
        self.log_list.append([event_time, int(idle_node), int(busy_node), int(full_node), int(idle_gpu), int(busy_gpu), int(pending_job), int(running_job), int(completed_job)])
        if len(self.log_list) >= 1:
            self.dump_all_logs()

    def dump_job_logs(self):
        fd = open(self.log_job, 'a+')
        log_writer = csv.writer(fd)  
        for log in self.job_list:
            log_writer.writerow(log)
        fd.close()
        del self.job_list[:]

    def job_complete(self, job, event_time):
        '''
        ['even_time', 'job_id', 'num_gpu', 'submit_time', 'start_time', 'end_time', 'executed time', 'real executed time', 'jct', 'duration', 'pending_time', 'job_counter', 'promote']
        '''
        job['end_time'] = event_time
        executed_time = job['end_time'] - job['start_time']
        real_executed_time = job['total_executed_time']
        jct = job['end_time'] - job['submit_time']
        if FLAGS.scheme == 'count':
            self.job_list.append([event_time, job['job_id'], job['num_gpu'], job['submit_time'], job['start_time'], job['end_time'], executed_time, real_executed_time, jct, job['duration'], job['pending_time'], job['preempt'], job['resume'], job['promote']])
        else:
            self.job_list.append([event_time, job['job_id'], job['num_gpu'], job['submit_time'], job['start_time'], job['end_time'], executed_time, real_executed_time, jct, job['duration'], job['pending_time'], job['job_counter'], job['promote']])


        if len(self.job_list) >= 1:
            self.dump_job_logs()

    def checkpoint_gpu_demands(self, event_time):
        '''        
        1-GPU, 2-GPU, 4-GPU, 8-GPU, 12-GPU, 16-GPU, 24-GPU, 32-GPU
        '''
        log_list = [event_time]
        gpu_list = [1,2,4,8,12,16,24,32]
        for num_gpu in gpu_list:
            total_gpu_job = 0
            if num_gpu in JOBS.gpu_job:
                total_gpu_job = num_gpu * JOBS.gpu_job[num_gpu]

            log_list.append(total_gpu_job)

        self.job_list.append(log_list)
        if len(self.job_list) >= 1:
            self.dump_job_logs()


LOG = _Log()

_allowed_symbols = [
    'LOG'
]