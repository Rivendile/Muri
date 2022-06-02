from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils

'''
TODO: add cpu and network load support in class _Node
'''
class _Node(object):
    def __init__(self, id, num_gpu=0, num_cpu=0, mem=0):
        self.id = id
        self.num_cpu = num_cpu
        self.free_cpus = num_cpu
        self.num_gpu = num_gpu       
        self.free_gpus = num_gpu
        #network load: can be bw, or the amount of traffic
        # in and out should be the same
        self.network_in = 0
        self.network_out = 0

        self.mem = mem
        self.free_mem = mem

        #node class for gandiva
        self.job_gpu = 0
        self.num_jobs = 0
        self.gpu_job_list = [{0:[], 1:[]} for i in range(self.num_gpu)]
        self.gpu_util_list = [0.0 for i in range(self.num_gpu)]

        utils.print_fn('    Node[%d] has %d gpus, %d cpus, %d G memory' % (id, num_gpu, num_cpu, mem))
    
    def init_node(self, num_gpu=0, num_cpu=0, mem=0):
        if num_gpu != 0:
            self.num_gpu = num_gpu
            self.free_gpus = num_gpu
        if num_cpu != 0:
            self.num_cpu = num_cpu
            self.free_cpus = num_cpu
        if mem != 0:
            self.mem = mem
            self.free_mem = mem 
        self.gpu_job_list = [{0:[], 1:[]} for i in range(self.num_gpu)]
        self.gpu_util_list = [0.0 for i in range(self.num_gpu)]

        self.add_gpus(self.num_gpu)        
        self.add_cpus(self.num_gpu)        


    ''' GPU  '''
    def add_gpus(self, num_gpu=0):
        pass

    def check_free_gpus(self):
        return self.free_gpus

    def get_free_gpus(self, priority):
        avail_gpu_list = []
        if priority==0:
            for i in range(self.num_gpu):
                if len(self.gpu_job_list[i][0])==0:
                    avail_gpu_list.append(i)
        else:
            for i in range(self.num_gpu):
                if len(self.gpu_job_list[i][1])<=1:
                    avail_gpu_list.append(i)
        return avail_gpu_list
            


    def alloc_gpus(self, num_gpu=0, priority=-1, avail_gpu_list=None, job_idx=-1, gpu_util=0.5):
        '''
        If enough free gpus, allocate gpus
        Return: True, for success;
                False, for failure
        '''
        if num_gpu > self.free_gpus:
            return False
        else:
            self.free_gpus -= num_gpu
            if priority>=0:
                for avail_gpu in avail_gpu_list:
                    self.gpu_job_list[avail_gpu][priority].append(job_idx)
                    self.gpu_util_list[avail_gpu] += gpu_util
            return True

    def release_gpus(self, num_gpu=0, priority=-1, avail_gpu_list=None, job_idx=-1, gpu_util=0.5):
        '''
        release using gpus back to free list
        '''
        if priority>=0:
            for avail_gpu in avail_gpu_list:
                if job_idx in self.gpu_job_list[avail_gpu][priority]:
                    assert job_idx in self.gpu_job_list[avail_gpu][priority]
                    self.gpu_job_list[avail_gpu][priority].remove(job_idx)
                    self.gpu_util_list[avail_gpu] -= gpu_util
        if priority!=1:
            if self.free_gpus + num_gpu > self.num_gpu:
                self.free_gpus = self.num_gpu
                return False
            else:
                self.free_gpus += num_gpu
                return True
        else:
            return True


    ''' CPU '''

    def add_cpus(self, num_cpu=0):
        pass

    def check_free_cpus(self):
        return self.free_cpus

    def alloc_cpus(self, num_cpu=0):
        '''
        If enough free cpus, allocate gpus
        Return: True, for success;
                False, for failure
        '''
        if num_cpu > self.free_cpus:
            return False
        else:
            self.free_cpus -= num_cpu
            return True

    def release_cpus(self, num_cpu=0):
        '''
        release using cpus back to free list
        '''
        if self.free_cpus + num_cpu > self.num_cpu:
            self.free_cpus = self.num_cpu
            return False
        else:
            self.free_cpus += num_cpu
            return True 


    '''network'''

    def add_network_load(self, in_load=0, out_load=0):
        self.network_in += in_load
        self.network_out += out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)


    def release_network_load(self, in_load=0, out_load=0):
        self.network_in -= in_load
        self.network_out -= out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)

    def set_network_load(self, in_load=0, out_load=0):
        self.network_in = in_load
        self.network_out = out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)


    def alloc_job_res(self, num_gpu=0, num_cpu=0, priority=-1, avail_gpu_list=None, job_idx=-1, gpu_util=0.5):
        '''
        alloc job resource
        '''
        gpu = self.alloc_gpus(num_gpu, priority, avail_gpu_list, job_idx, gpu_util)
        cpu = self.alloc_cpus(num_cpu)

        # print(job_idx, gpu, cpu)

        if cpu == False or gpu == False:
            self.release_gpus(num_gpu, priority, avail_gpu_list, job_idx, gpu_util)
            self.release_cpus(num_cpu)
            return False

        return True 

    def find_gpu_util(self, gpu_util_upper):
        gpu_list = []
        for i in range(self.num_gpu):
            if self.gpu_util_list[i]<gpu_util_upper:
                gpu_list.append({'node':self.id, 'gpu':i})
                # print(self.id, i, self.gpu_util_list[i])
        return gpu_list

    def release_job_res(self, node_dict, priority=-1, avail_gpu_list=[], job_idx=-1, gpu_util=0.5):
        '''
        input is node_dict from placement
        {'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'network': xxxx, 'tasks': [w2, ps2]}
        '''
        self.release_network_load(node_dict['network'], node_dict['network'])
        cpu = True
        if priority!=1:
            cpu = self.release_cpus(node_dict['num_cpu'])
        gpu = self.release_gpus(node_dict['num_gpu'], priority, avail_gpu_list, job_idx, gpu_util=gpu_util)

        self.free_mem = self.free_mem + node_dict['mem']

        # print(job_idx, cpu, gpu)

        return (cpu and gpu)

    def release_job_gpu_cpu(self, num_gpu, num_cpu):
        '''
        input is gpu and cpu
        '''
        cpu = self.release_cpus(num_cpu)
        gpu = self.release_gpus(num_gpu)

        return (cpu and gpu)
