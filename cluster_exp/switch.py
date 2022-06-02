from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from node import _Node
import flags 
import utils
import jobs
import math
import copy

FLAGS = flags.FLAGS
JOBS = jobs.JOBS


class _Switch(object):

    def __init__(self, id, num_node=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_node=0):
        self.num_node = num_node
        self.num_gpu_p_node = num_gpu_p_node
        self.num_cpu_p_node = num_cpu_p_node
        self.mem_p_node = mem_p_node
        self.id = id
        self.node_list = list()
        utils.print_fn('  Switch[%d] has %d nodes' % (id, num_node))

    def add_nodes(self, num_node=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_node=0):
        if num_node != 0 and num_gpu_p_node != 0 and num_cpu_p_node != 0 and mem_p_node != 0:
            self.num_node = num_node
            self.num_gpu_p_node = num_gpu_p_node
            self.num_cpu_p_node = num_cpu_p_node
            self.mem_p_node = mem_p_node
        
        for n in range(0, self.num_node):
            tmp_n = _Node(n, self.num_gpu_p_node, self.num_cpu_p_node, self.mem_p_node)
            self.node_list.append(tmp_n)



    def alloc_gpus(self, job):
        '''
        alloc gpus to job
        '''
        pass 

    def try_cross_node_alloc(self, job, not_place=False):
        '''
        used in MS_YARN placement
        try get gpus from multiple nodes
            [need_gpu / gpu_p_node] nodes, and one node with [need_gpu % gpu_p_node]
        if can't find , give up, and return False
        '''
        need_gpu = job['num_gpu']
        num_full_nodes = math.floor(need_gpu / self.num_gpu_p_node)
        last_node_gpu =  need_gpu % self.num_gpu_p_node
        last_node_cpu = int(last_node_gpu * 6)
        last_node = None
        idle_node_cpu = int(self.num_gpu_p_node * 6) #w:2, ps:4

        model_size = job['model']['total_size']

        ps_mem = JOBS.ps_mem + need_gpu * JOBS.p_w_mem
        ps_w_mem = ps_mem + JOBS.worker_mem 

        full_node_list = list()
        for node in self.node_list:
            if node.check_free_gpus() == node.num_gpu and node.check_free_cpus() >= idle_node_cpu and node.free_mem >= (ps_w_mem * node.num_gpu):
                #get idle node
                full_node_list.append(node)
                if len(full_node_list) == num_full_nodes:
                    #enough full nodes
                    break
        if len(full_node_list) < num_full_nodes:
            return False 

        if last_node_gpu != 0:
            for node in self.node_list: 
                if node not in full_node_list:
                    if node.check_free_gpus() >= last_node_gpu and node.check_free_cpus() >= last_node_cpu and node.free_mem >= (ps_w_mem * last_node_gpu):
                        #get last node
                        last_node = node
                        break
            if last_node == None:
                return False


        ''' can allocate, do resource counting and record job placement '''
        node_list = list()
        idx = 0
        for node in full_node_list:
            if not_place==False:
                gpu_list = [gpu_id for gpu_id in range(node.num_gpu)]
                node.alloc_job_res(node.num_gpu, idle_node_cpu, 0, gpu_list, job['job_idx'], 1.0)  
            else:
                node.alloc_job_res(node.num_gpu, idle_node_cpu)
            node.free_mem -= ps_w_mem * node.num_gpu
            node_dict = dict()
            node_dict['id'] = node.id
            node_dict['num_gpu'] = node.num_gpu
            node_dict['num_cpu'] = idle_node_cpu
            node_dict['mem'] = ps_w_mem * node.num_gpu
            if not_place==False:
                node_dict['gpu_list'] = gpu_list

            # traffic = round(model_size * node.num_gpu, 1)
            # for i in range(0, node.num_gpu):
            #     traffic += traffic + job['ps_network'][idx]
            #     traffic = round(traffic, 1)
            #     idx += 1

            #worker traffic
            traffic = round(model_size * node.num_gpu, 1)
            #ps traffic
            for i in range(0, node.num_gpu):
                #add ps traffic
                traffic += job['ps_network'][idx] * (need_gpu - node.num_gpu) #send to (need - local_gpu) workers, no need for local PS-to-worker
                #remove co-locate worker traffic
                traffic -= job['ps_network'][idx] * node.num_gpu #no need for local worker-to-PS
                traffic = round(traffic, 1)
                idx += 1
            node_dict['network'] = traffic
            node.add_network_load(traffic, traffic)

            node_dict['tasks'] = list()
            node_list.append(node_dict)

        if last_node_gpu != 0:
            if not_place==False:
                gpu_list = last_node.get_free_gpus(-1)[:last_node_gpu]
                last_node.alloc_job_res(last_node_gpu, last_node_cpu, 0, gpu_list, job['job_idx'], 1.0)
            else:
                last_node.alloc_job_res(last_node_gpu, last_node_cpu)
            last_node.free_mem -= ps_w_mem * last_node_gpu 
            node_dict = dict()
            node_dict['id'] = last_node.id
            node_dict['num_gpu'] = last_node_gpu
            node_dict['num_cpu'] = last_node_cpu
            node_dict['mem'] = ps_w_mem * last_node_gpu
            if not_place==False:
                node_dict['gpu_list'] = gpu_list

            traffic = round(model_size * last_node_gpu, 1)
            # for i in range(0, last_node_gpu):
            #     traffic += job['ps_network'][idx]
            #     traffic = round(traffic, 1)
            #     idx += 1
            for i in range(0, last_node_gpu):
                traffic += job['ps_network'][idx] * (need_gpu - last_node_gpu) #send to (need-last_gpu), no need for local PS-to-worker
                traffic -= job['ps_network'][idx] * last_node_gpu #no need for local worker-to-PS
                traffic = round(traffic, 1)
                idx += 1
            node_dict['network'] = traffic
            last_node.add_network_load(traffic, traffic)

            node_dict['tasks'] = list()
            node_list.append(node_dict)

        if not_place==False:
            JOBS.create_multi_nodes_placement(job, self.id, node_list)
        return True


    def try_single_node_alloc(self, job, not_place=False):
        '''
        used in MS_YARN placement
        try get gpus from a single node
        if can't find a node, give up, and return False
        '''
        need_gpu = job['num_gpu']
        if len(job['ps_network']) == 0 and job['num_gpu'] == 1:
            need_cpu = int(need_gpu * 2) # worker:2
        else:
            need_cpu = int(need_gpu * 6) # worker:2, ps:4

        # print("try_single_node_alloc: ", need_gpu, need_cpu, JOBS.worker_mem)

        for node in self.node_list:
            # print(node.id, node.check_free_gpus(), node.check_free_cpus(), node.free_mem)
            if (node.check_free_gpus() >= need_gpu) and (node.check_free_cpus() >= need_cpu) and (node.free_mem >= JOBS.worker_mem):
                # if node.alloc_gpus(need_gpu) == False:
                if not_place==False:
                    gpu_list = node.get_free_gpus(0)[:need_gpu]
                    if node.alloc_job_res(need_gpu, need_cpu, 0, gpu_list, job['job_idx'], 1.0) == False:
                        continue
                    node.free_mem = node.free_mem - JOBS.worker_mem
                    traffic = JOBS.create_single_node_placement(job, self.id, node.id, need_gpu, need_cpu, JOBS.worker_mem, gpu_list)
                else:
                    if node.alloc_job_res(need_gpu, need_cpu) == False:
                        continue
                    node.free_mem = node.free_mem - JOBS.worker_mem

                # node.add_network_load(traffic, traffic)

                return True
            else:
                continue

        return False


    def ms_yarn_alloc_gpus(self, job):
        '''
        ms_yarn allocates gpus from a single switch, 
        if no enough gpus, give up, return False (all-or-nothing)

        if need_gpu > gpu_p_node
            then get [need_gpu / gpu_p_node] nodes, and one node with [need_gpu % gpu_p_node]
        if need_gpu <= gpu_p_node
            then get one node with enough gpus
        '''
        need_gpu = job['num_gpu']
        ret = False
        if need_gpu > self.num_gpu_p_node:
            ret = self.try_cross_node_alloc(job)
        else:
            ret = self.try_single_node_alloc(job)

        return ret

    def ms_yarn_alloc_res(self, job, not_place=False):
        '''
        ms_yarn allocates res from a single switch, 
        if no enough gpus, give up, return False (all-or-nothing)

        if need_gpu > gpu_p_node
            then get [need_gpu / gpu_p_node] nodes, and one node with [need_gpu % gpu_p_node]
        if need_gpu <= gpu_p_node
            then get one node with enough gpus
        '''
        need_gpu = job['num_gpu']
        ret = False
        if need_gpu > self.num_gpu_p_node:
            ret = self.try_cross_node_alloc(job, not_place)
        else:
            ret = self.try_single_node_alloc(job, not_place)

        return ret

    def add_job_gpu_util(self, job):
        for placement in job['placements']:
            for node_pl in placement['nodes']:
                print("node_pl: ", node_pl)
                for gpu_id in node_pl['gpu_list']:
                    self.node_list[node_pl['id']].gpu_util_list[gpu_id] += 0.01

    def try_cross_node_alloc_antman(self, job):
        '''
        used in MS_YARN placement
        try get gpus from multiple nodes
            [need_gpu / gpu_p_node] nodes, and one node with [need_gpu % gpu_p_node]
        if can't find , give up, and return False
        '''
        assert job['remaining_gpu']>0
        need_gpu = job['remaining_gpu']
        num_full_nodes = math.floor(need_gpu / self.num_gpu_p_node)
        last_node_gpu =  need_gpu % self.num_gpu_p_node
        last_node_cpu = int(last_node_gpu * 6)
        last_node = None
        idle_node_cpu = int(self.num_gpu_p_node * 6) #w:2, ps:4
        enough_flag = True

        model_size = job['model']['total_size']

        ps_mem = JOBS.ps_mem + need_gpu * JOBS.p_w_mem
        ps_w_mem = ps_mem + JOBS.worker_mem 

        full_node_list = list()
        if num_full_nodes>0:
            for node in self.node_list:
                avail_gpu_list = node.get_free_gpus(job['priority'])
                if len(avail_gpu_list)>=self.num_gpu_p_node and node.check_free_cpus() >= idle_node_cpu and node.free_mem >= (ps_w_mem * node.num_gpu):
                    #get idle node
                    full_node_list.append(node)
                    if len(full_node_list) == num_full_nodes:
                        #enough full nodes
                        break
        if len(full_node_list) < num_full_nodes:
            enough_flag = False

        if last_node_gpu != 0:
            if last_node_gpu < job['num_gpu']%self.num_gpu_p_node:
                last_node = self.node_list[job['last_node_id']]
            else:
                max_node_cnt = 0
                max_node = None
                for node in self.node_list:
                    if node not in full_node_list:
                        avail_gpu_list = node.get_free_gpus(job['priority'])
                        avail_gpu_cnt = len(avail_gpu_list)
                        if len(avail_gpu_list) >= last_node_gpu and node.check_free_cpus() >= last_node_cpu and node.free_mem >= (ps_w_mem * last_node_gpu):
                            #get last node
                            last_node = node
                            break
                        else:
                            if avail_gpu_cnt >max_node_cnt:
                                max_node_cnt = avail_gpu_cnt
                                max_node = node
            if last_node == None:
                enough_flag = False
                if max_node != None:
                    last_node = max_node

        ''' can allocate, do resource counting and record job placement '''
        node_list = list()
        idx = 0
        if last_node_gpu != 0:
            if last_node_gpu == job['num_gpu']%self.num_gpu_p_node:
                if last_node == None: # why this situation?
                    node_dict = dict()
                    node_dict['id'] = -1
                    node_dict['num_gpu'] = 0
                    node_dict['num_cpu'] = 0
                    node_dict['mem'] = 0
                    node_dict['gpu_list'] = list()
                    node_dict['network'] = 0
                    node_dict['tasks'] = list()
                    node_list.append(node_dict)
                else:
                    job['last_node_id'] = last_node.id
                    avail_gpu_list = last_node.get_free_gpus(job['priority'])
                    if len(avail_gpu_list)>last_node_gpu:
                        avail_gpu_list = avail_gpu_list[:last_node_gpu]
                    avail_gpu_cnt = len(avail_gpu_list)
                    last_node_cpu = avail_gpu_cnt * 6
                    assert last_node.alloc_job_res(avail_gpu_cnt, last_node_cpu, job['priority'], avail_gpu_list, job['job_idx'], gpu_util=job['gpu_util']-0.01)==True
                    last_node.free_mem -= ps_w_mem * avail_gpu_cnt
                    has_flag = False
                    if len(job['placements'])>0 and len(job['placements'][0]['nodes'])>0:
                        has_flag = True
                        node_dict = job['placements'][0]['nodes'][0]
                    else:
                        node_dict = dict()
                        node_dict['id'] = -1
                        node_dict['num_gpu'] = 0
                        node_dict['num_cpu'] = 0
                        node_dict['mem'] = 0
                        node_dict['gpu_list'] = list()
                        node_dict['network'] = 0
                        node_dict['tasks'] = list()

                    assert node_dict['id'] == -1 or node_dict['id'] == last_node.id
                    node_dict['id'] = last_node.id
                    node_dict['num_gpu'] += avail_gpu_cnt
                    node_dict['num_cpu'] += last_node_cpu
                    node_dict['mem'] += ps_w_mem * avail_gpu_cnt
                    node_dict['gpu_list'] = node_dict['gpu_list'] + avail_gpu_list

                    traffic = round(model_size * avail_gpu_cnt, 1)
                    # for i in range(0, last_node_gpu):
                    #     traffic += job['ps_network'][idx]
                    #     traffic = round(traffic, 1)
                    #     idx += 1
                    for i in range(0, last_node_gpu):
                        traffic += job['ps_network'][idx] * (need_gpu - last_node_gpu) #send to (need-last_gpu), no need for local PS-to-worker
                        traffic -= job['ps_network'][idx] * last_node_gpu #no need for local worker-to-PS
                        traffic = round(traffic, 1)
                        idx += 1
                    node_dict['network'] += traffic
                    last_node.add_network_load(traffic, traffic)
                    job['remaining_gpu'] -= avail_gpu_cnt

                    node_dict['tasks'] = list()
                    if not has_flag:
                        node_list.append(node_dict)           
            else:
                if last_node != None:
                    avail_gpu_list = last_node.get_free_gpus(job['priority'])
                    if len(avail_gpu_list)>last_node_gpu:
                        avail_gpu_list = avail_gpu_list[:last_node_gpu]
                    avail_gpu_cnt = len(avail_gpu_list)
                    last_node_cpu = avail_gpu_cnt * 6
                    assert last_node.alloc_job_res(avail_gpu_cnt, last_node_cpu, job['priority'], avail_gpu_list, job['job_idx'], gpu_util=job['gpu_util']-0.01)==True
                    last_node.free_mem -= ps_w_mem * avail_gpu_cnt
                    node_dict = job['placements'][0]['nodes'][0]
                    assert node_dict['id'] == last_node.id
                    node_dict['num_gpu'] += avail_gpu_cnt
                    node_dict['num_cpu'] += last_node_cpu
                    node_dict['mem'] += ps_w_mem * avail_gpu_cnt
                    node_dict['gpu_list'] = node_dict['gpu_list']+avail_gpu_list

                    job['remaining_gpu'] -= avail_gpu_cnt

        for node in full_node_list:
            assert node.alloc_job_res(node.num_gpu, idle_node_cpu, job['priority'], [i for i in range(node.num_gpu)], job['job_idx'], gpu_util=job['gpu_util']-0.01)==True 
            node.free_mem -= ps_w_mem * node.num_gpu
            node_dict = dict()
            node_dict['id'] = node.id
            node_dict['num_gpu'] = node.num_gpu
            node_dict['num_cpu'] = idle_node_cpu
            node_dict['mem'] = ps_w_mem * node.num_gpu
            node_dict['gpu_list'] = [i for i in range(self.num_gpu_p_node)]

            # traffic = round(model_size * node.num_gpu, 1)
            # for i in range(0, node.num_gpu):
            #     traffic += traffic + job['ps_network'][idx]
            #     traffic = round(traffic, 1)
            #     idx += 1

            #worker traffic
            traffic = round(model_size * node.num_gpu, 1)
            #ps traffic
            for i in range(0, node.num_gpu):
                #add ps traffic
                traffic += job['ps_network'][idx] * (need_gpu - node.num_gpu) #send to (need - local_gpu) workers, no need for local PS-to-worker
                #remove co-locate worker traffic
                traffic -= job['ps_network'][idx] * node.num_gpu #no need for local worker-to-PS
                traffic = round(traffic, 1)
                idx += 1
            node_dict['network'] = traffic
            node.add_network_load(traffic, traffic)

            node_dict['tasks'] = list()
            node_list.append(node_dict)
            # for i in range(node.num_gpu):
            #     node.gpu_job_list[i][0].append(job['job_idx'])
            #     node.gpu_util_list[i] += job['gpu_util']
        job['remaining_gpu'] -= len(full_node_list) * self.num_gpu_p_node

        JOBS.create_multi_nodes_placement_same_switch(job, self.id, node_list)

        if enough_flag:
            self.add_job_gpu_util(job)

        return enough_flag

    def try_single_node_alloc_antman(self, job):
        '''
        used in MS_YARN placement
        try get gpus from a single node
        if can't find a node, give up, and return False
        '''
        assert job['remaining_gpu'] >0
        need_gpu = job['remaining_gpu']
        if len(job['ps_network']) == 0 and job['num_gpu'] == 1:
            need_cpu = int(need_gpu * 2) # worker:2
        else:
            need_cpu = int(need_gpu * 6) # worker:2, ps:4

        # print("try_single_node_alloc: ", need_gpu, need_cpu, JOBS.worker_mem)
        max_node_id = -1
        max_node_gpu = 0
        if need_gpu == job['num_gpu']: # no gpu is allocated for job
            for node in self.node_list:
                # print(node.id, node.check_free_gpus(), node.check_free_cpus(), node.free_mem)
                avail_gpu_list = node.get_free_gpus(job['priority'])
                avail_gpu_cnt = len(avail_gpu_list)
                if (avail_gpu_cnt>=need_gpu) and (node.check_free_cpus() >= need_cpu) and (node.free_mem >= JOBS.worker_mem):
                    # if node.alloc_gpus(need_gpu) == False:
                    assert node.alloc_job_res(need_gpu, need_cpu, job['priority'], avail_gpu_list[:need_gpu], job['job_idx'], gpu_util=job['gpu_util']-0.01) == True
                    node.free_mem = node.free_mem - JOBS.worker_mem
                    traffic = JOBS.create_single_node_placement(job, self.id, node.id, need_gpu, need_cpu, JOBS.worker_mem, avail_gpu_list[:need_gpu])
                    # node.add_network_load(traffic, traffic)
                    job['remaining_gpu'] -= need_gpu
                    job['last_node_id'] = node.id
                    self.add_job_gpu_util(job)
                    return True
                else:
                    if avail_gpu_cnt>max_node_gpu:
                        max_node_gpu = avail_gpu_cnt
                        max_node_id = node.id
                        max_node_gpu_list = copy.deepcopy(avail_gpu_list)
            # not enough gpu, reserve available
            need_gpu = max_node_gpu
            if need_gpu>0:
                if len(job['ps_network']) == 0 and job['num_gpu'] == 1:
                    need_cpu = int(need_gpu * 2) # worker:2
                else:
                    need_cpu = int(need_gpu * 6) # worker:2, ps:4      
                assert self.node_list[max_node_id].alloc_job_res(need_gpu, need_cpu, job['priority'], max_node_gpu_list, job['job_idx'], gpu_util=job['gpu_util']-0.01) == True
                self.node_list[max_node_id].free_mem = self.node_list[max_node_id].free_mem - JOBS.worker_mem
                traffic = JOBS.create_single_node_placement(job, self.id, max_node_id, need_gpu, need_cpu, JOBS.worker_mem, max_node_gpu_list)
                job['remaining_gpu'] -= need_gpu
                job['last_node_id'] = max_node_id
        else: # have allocated some gpus for job
            assert len(job['placements'])==1
            node_id = job['last_node_id']
            node = self.node_list[node_id]
            avail_gpu_list = node.get_free_gpus(job['priority'])
            avail_gpu_cnt = len(avail_gpu_list)
            if (avail_gpu_cnt>=need_gpu) and (node.check_free_cpus() >= need_cpu) and (node.free_mem >= JOBS.worker_mem): # enough gpu
                # if node.alloc_gpus(need_gpu) == False:
                assert node.alloc_job_res(need_gpu, need_cpu, job['priority'], avail_gpu_list[:need_gpu], job['job_idx'], gpu_util=job['gpu_util']-0.01) == True
                node.free_mem = node.free_mem - JOBS.worker_mem
                # print(job['job_idx'], job['placements'])
                traffic = JOBS.create_single_node_placement(job, self.id, node_id, need_gpu, need_cpu, JOBS.worker_mem, avail_gpu_list[:need_gpu], True)
                job['remaining_gpu'] -= need_gpu
                # node.add_network_load(traffic, traffic)
                self.add_job_gpu_util(job)
                return True
            else: # not enough gpu
                need_gpu = avail_gpu_cnt
                if need_gpu>0:
                    need_cpu = int(need_gpu * 6) # worker:2, ps:4
                    assert node.alloc_job_res(need_gpu, need_cpu, job['priority'], avail_gpu_list, job['job_idx'], gpu_util=job['gpu_util']-0.01) == True
                    # node.free_mem = node.free_mem
                    traffic = JOBS.create_single_node_placement(job, self.id, node_id, need_gpu, need_cpu, JOBS.worker_mem, avail_gpu_list, True)    
                    job['remaining_gpu'] -= need_gpu
        return False

    def find_gpu_util(self, gpu_util_upper=0.8):
        '''
        find gpus whose gpu util < gpu_util_upper
        '''
        gpu_list = []
        for node in self.node_list:
            gpu_list_node = node.find_gpu_util(gpu_util_upper)
            gpu_list.extend(gpu_list_node)
        return gpu_list
    
    def sortGPUutil(self, elem):
        return self.node_list[elem['node']].gpu_util_list[elem['gpu']]

    def min_load_nodes(self, gpus1, need_gpu):
        '''
        return need_gpu gpus whose gpu util are minimum
        '''
        gpus1.sort(key=self.sortGPUutil)
        return gpus1[:need_gpu]


    def antman_alloc_res(self, job, gpu_util_upper=0.8):
        '''
        antman allocates res from a single switch
        for resource-guarantee job:

        for opportunistic job:
        if no enough gpus, give up, return False (all-or-nothing)
        '''
        need_gpu = job['num_gpu']
        if job['priority']==0: # resource-guarantee job
            if need_gpu > self.num_gpu_p_node:
                ret = self.try_cross_node_alloc_antman(job)
            else:
                ret = self.try_single_node_alloc_antman(job)
        else:
            gpus1 = self.find_gpu_util(gpu_util_upper)
            if len(gpus1)<need_gpu:
                return False
            gpus2 = self.min_load_nodes(gpus1, need_gpu)
            # print(job['job_idx'], "low priority")
            # print(gpus1)
            # print(gpus2)
            tmp_node_dict = dict()
            for gpu_id, gpu2 in enumerate(gpus2):
                self.node_list[gpu2['node']].gpu_util_list[gpu2['gpu']] += job['gpu_util']
                self.node_list[gpu2['node']].gpu_job_list[gpu2['gpu']][1].append(job['job_idx'])
                if gpu2['node'] not in tmp_node_dict:
                    tmp_node_dict[gpu2['node']]=list()
                tmp_node_dict[gpu2['node']].append(gpu2['gpu'])
            node_key_list = tmp_node_dict.keys()
            for node_key in node_key_list:
                need_gpu = len(tmp_node_dict[node_key])
                if len(job['ps_network']) == 0 and job['num_gpu'] == 1:
                    need_cpu = int(need_gpu * 2) # worker:2
                else:
                    need_cpu = int(need_gpu * 6) # worker:2, ps:4
                JOBS.create_single_node_placement(job, self.id, node_key, need_gpu, need_cpu, JOBS.worker_mem, tmp_node_dict[node_key])
                self.node_list[node_key].free_mem -= JOBS.worker_mem
            job['remaining_gpu'] = 0
            ret = True
        return ret

    # not used
    def release_gpus(self, nodes):
        '''
        release gpus from nodes
        nodes:
        [{'id':xx, 'num_gpu':xxx}, {'id':xx, 'num_gpu':xxx}]
        '''
        for node_dict in nodes:
            if ('id' not in node_dict) or ('num_gpu' not in node_dict):
                return False
            node = self.node_list[node_dict['id']]
            ret = node.release_gpus(node_dict['num_gpu'])
            if ret == False:
                return False

        return True

    def release_job_res(self, nodes, priority=-1, job_idx=-1, gpu_util=0.5):
        '''
        release job resources from nodes
        nodes:
        [{'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'network': xxxx, 'tasks': [w2, ps2]}, 
        {'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'network': xxxx, 'tasks': [ps0]}]
        '''
        for node_dict in nodes:
            if ('id' not in node_dict) or ('num_gpu' not in node_dict) or ('num_cpu' not in node_dict) or ('tasks' not in node_dict):
                print("switch release error, no info", job_idx)
                return False
            node = self.node_list[node_dict['id']]
            # ret = node.release_gpus(node_dict['num_gpu'])
            ret = node.release_job_res(node_dict, priority, node_dict['gpu_list'], job_idx, gpu_util=gpu_util)
            if ret == False:
                print("switch release error, node release error", job_idx)
                return False

        return True