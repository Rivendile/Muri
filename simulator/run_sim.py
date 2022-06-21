from __future__ import print_function
import csv
import re
import sys
import types
import time
import math
#parse args
import argparse
import copy
import os

import numpy as np
import util
import flags
import jobs
import cluster
import log
import cvxpy as cp
from cvxpy import SolverError
from matching import Blossom_Same, _Packing

sys.setrecursionlimit(1000000000)
# import hosts
# import placement_scheme as scheme
# import cmd

# profiled overhead of start for each workloads
overhead_dict = {1:dict(), 2:dict(), 4:dict(), 8:dict(), 16:dict(), 32:dict()}
overhead_dict[1] = {'vgg16':7, 'vgg19':7, 'resnet18':4, 'shufflenet_v2_x1_0':4, 'bert':10, 'gpt2':10, 'a2c':38, 'dqn':5}
overhead_dict[2] = {'vgg16':7, 'vgg19':7, 'resnet18':5, 'shufflenet_v2_x1_0':4, 'bert':10, 'gpt2':10, 'a2c':39, 'dqn':5}
overhead_dict[4] = {'vgg16':8, 'vgg19':8, 'resnet18':5, 'shufflenet_v2_x1_0':5, 'bert':10, 'gpt2':10, 'a2c':39, 'dqn':6}
overhead_dict[8] = {'vgg16':9, 'vgg19':9, 'resnet18':7, 'shufflenet_v2_x1_0':5, 'bert':10, 'gpt2':10, 'a2c':46, 'dqn':6}
overhead_dict[16] = {'vgg16':9, 'vgg19':10, 'resnet18':7, 'shufflenet_v2_x1_0':7, 'bert':10, 'gpt2':10, 'a2c':46, 'dqn':8}
overhead_dict[32] = {'vgg16':10, 'vgg19':10, 'resnet18':7, 'shufflenet_v2_x1_0':8, 'bert':10, 'gpt2':10, 'a2c':46, 'dqn':8}
#parse input arguments
flags.DEFINE_string('trace_file', 'tf_job.csv',
                '''Provide TF job trace file (*.csv, *.txt).
                    *.csv file, use \',\' as delimiter; *.txt file, user \' \' as deliminter. 
                    Default file is tf_job.csv ''')
flags.DEFINE_string('log_path', 'result-' + time.strftime("%Y%m%d-%H-%M-%S", time.localtime()),
                '''Simulation output folder, including cluster/node/gpu usage trace, pending job_queue info.
                Default folder is result-[time]''')
flags.DEFINE_string('scheme', 'yarn',
                '''
                Job placement scheme:
                0.count, just resource counting, without assignment (which gpu, which cpu)
                1.yarn, ms yarn
                2.random
                3.crandom (consolidate + random)
                4.greedy
                5.balance
                6.cbalance (consolidate + balance)
                Default is yarn''')
flags.DEFINE_string('schedule', 'fifo',
                '''
                Job schedule scheme:
                1.fifo
                2.shortest, shortest-remaining-time job first
                3.shortest-gpu, shortest-remaining-gputime job first 
                4.dlas, discretized las 
                5.dlas-gpu, dlas using gpu time
                6.antman, AntMan
                7.themis, Themis
                8.multi-resource-blossom-same-gpu(-unaware), match jobs with same #GPU using blossom algorithm using gputime (unaware job duration)
                Default is fifo''')
flags.DEFINE_integer('num_switch', 1, 
                '''Part of cluster spec: the number of switches in this cluster, default is 1''')
flags.DEFINE_integer('num_node_p_switch', 32, 
                '''Part of cluster spec: the number of nodes under a single switch, default is 32''')
flags.DEFINE_integer('num_gpu_p_node', 8, 
                '''Part of cluster spec: the number of gpus on each node, default is 8''')
flags.DEFINE_integer('num_cpu_p_node', 64,
                '''Part of cluster spec: the number of cpus on each node, default is 64''')
flags.DEFINE_integer('mem_p_node', 256,
                '''Part of cluster spec: memory capacity on each node, default is 128''')
flags.DEFINE_string('cluster_spec', None,
                '''Part of cluster spec: cluster infra spec file, 
                this file will overwrite the specs from num_switch, num_node_p_switch, and num_gpu_p_node
                Spec format:
                    num_switch,num_node_p_switch,num_gpu_p_node
                    int,int,int''')
# for multi_resource sharing
flags.DEFINE_integer('multi_resource', 4, 
                '''Part of job spec: the num of resources used for each job, default is 4''')
flags.DEFINE_integer('packing_num', 4, 
                '''maximum number of packing jobs''')
flags.DEFINE_float('weight_lbd', 0.0, '''The factor of the lower bound of expected weight (i jobs packing of n resources: i/n)''')
flags.DEFINE_boolean('autopack', False, '''Unpack job if the combined normalized tput is slower than 1''')
flags.DEFINE_boolean('print', False, 
                '''Enable print out information, default is False''')
flags.DEFINE_boolean('flush_stdout', True, 
                '''Flush stdout, default is True''')
flags.DEFINE_version('0.1')


FLAGS = flags.FLAGS

#prepare JOBS list
JOBS = jobs.JOBS

#get host info
CLUSTER = cluster.CLUSTER

#get LOG object
LOG = log.LOG


def parse_job_file(trace_file):
    #check trace_file is *.csv
    fd = open(trace_file, 'r')
    deli = ','
    if ((trace_file.find('.csv') == (len(trace_file) - 4))):
        deli = ','
    elif ((trace_file.find('.txt') == (len(trace_file) - 4))):
        deli = ' '

    reader = csv.DictReader(fd, delimiter = deli) 
    ''' Add job from job trace file'''
    keys = reader.fieldnames
    util.print_fn('--------------------------------- Read TF jobs from: %s ---------------------------------' % trace_file) 
    util.print_fn('    we get the following fields:\n        %s' % keys)
    job_idx = 0
    for row in reader:
        #add job into JOBS
        JOBS.add_job(row)
        # JOBS.read_job_info(job_idx, 'num_gpu')
        job_idx += 1

    assert job_idx == len(JOBS.job_list) 
    assert JOBS.num_job == len(JOBS.job_list) 
    # JOBS.print_all_job_size_info()
    JOBS.sort_all_jobs()
    # print(lp.prepare_job_info(JOBS.job_list[0]))
    util.print_fn('---------------------------------- Get %d TF jobs in total ----------------------------------' % job_idx)
    # JOBS.read_all_jobs()
    fd.close()

def parse_cluster_spec():
    if FLAGS.cluster_spec:
        print(FLAGS.cluster_spec)
        spec_file = FLAGS.cluster_spec
        fd = open(spec_file, 'r')
        deli = ','
        if ((spec_file.find('.csv') == (len(spec_file) - 4))):
            deli = ','
        elif ((spec_file.find('.txt') == (len(spec_file) - 4))):
            deli = ' '
        reader = csv.DictReader(fd, delimiter = deli) 
        keys = reader.fieldnames
        util.print_fn(keys)
        if 'num_switch' not in keys:
            return
        if 'num_node_p_switch' not in keys:
            return
        if 'num_gpu_p_node' not in keys:
            return
        if 'num_cpu_p_node' not in keys:
            return
        if 'mem_p_node' not in keys:
            return
        
        ''' there should be only one line remaining'''
        assert reader.line_num == 1

        ''' get cluster spec '''
        for row in reader:
            # util.print_fn('num_switch %s' % row['num_switch'])
            FLAGS.num_switch = int(row['num_switch'])
            FLAGS.num_node_p_switch = int(row['num_node_p_switch'])
            FLAGS.num_gpu_p_node = int(row['num_gpu_p_node'])
            FLAGS.num_cpu_p_node = int(row['num_cpu_p_node'])
            FLAGS.mem_p_node = int(row['mem_p_node'])
        fd.close()

    util.print_fn("num_switch: %d" % FLAGS.num_switch)
    util.print_fn("num_node_p_switch: %d" % FLAGS.num_node_p_switch)
    util.print_fn("num_gpu_p_node: %d" % FLAGS.num_gpu_p_node)
    util.print_fn("num_cpu_p_node: %d" % FLAGS.num_cpu_p_node)
    util.print_fn("mem_p_node: %d" % FLAGS.mem_p_node)

    '''init infra'''
    CLUSTER.init_infra()
    # util.print_fn(lp.prepare_cluster_info())
    util.print_fn('--------------------------------- End of cluster spec ---------------------------------')
    return 


'''
Allocate job resource
'''
def try_get_job_res(job, not_place=False):
    '''
    select placement scheme
    '''
    if 'antman' in FLAGS.schedule:
        ret = CLUSTER.antman_placement(job)
    elif FLAGS.scheme == 'yarn':
        ret = CLUSTER.ms_yarn_placement(job, not_place)
    elif FLAGS.scheme == 'random':
        ret = CLUSTER.random_placement(job)
    elif FLAGS.scheme == 'crandom':
        ret = CLUSTER.consolidate_random_placement(job)
    elif FLAGS.scheme == 'greedy':
        ret = CLUSTER.greedy_placement(job)
    elif FLAGS.scheme == 'gandiva':
        ret = CLUSTER.gandiva_placement(job)
    elif FLAGS.scheme == 'count':
        ret = CLUSTER.none_placement(job)
    else:
        ret = CLUSTER.ms_yarn_placement(job)
    if ret == True:
        # job['status'] = 'RUNNING'
        pass
    return ret

def cal_shortest_expected_remaining(job_data, a):
    data = job_data['data']
    idx = next(x[0] for x in enumerate(data) if x[1] > a)

    if idx == (job_data['num'] - 1):
        return data[idx]

    num = job_data['num'] - 1 - idx 
    return round(sum(data[idx: (job_data['num'] - 1)]) * 1.0 / num, 2)

def shortest_first_sim_jobs(gputime=False):
    '''
    new jobs are added to the end of the ending queue
    but in the queue, shortest (gpu) job first be served, until no resource
    '''
    end_events = list()
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']

        if end_time < start_time:
            event_time = end_time
            event = end_events[0]
        elif end_time > start_time:        
            event_time = start_time
            # print("start-time %d, end_time %d" % (start_time, end_time))
            event = JOBS.job_events.pop(0)
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            event = JOBS.job_events.pop(0)
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                tmp = event_time - e_job['last_check_time']
                e_job['total_executed_time'] = e_job['total_executed_time'] + tmp
                #job completes
                CLUSTER.release_job_res(e_job)
                # CLUSTER.release_gpus(e_job)
                LOG.job_complete(e_job, event_time)
                JOBS.runnable_jobs.remove(e_job)


        #for new-start jobs, add to runnable
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                #add into runnable list with pending status
                JOBS.move_to_runnable(s_job)
                if FLAGS.schedule == 'shortest-expected':
                    s_job['remaining_expected'] = cal_shortest_expected_remaining(JOBS.job_dist_data, 0)

                s_job['remaining_time'] = s_job['remaining_iteration']*s_job['iteration_time']
                s_job['remaining_gputime'] = s_job['remaining_time'] * s_job['num_gpu']
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])

        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                if rjob['model_name'] in overhead_dict[rjob['num_gpu']]:
                    tmp_oh = overhead_dict[rjob['num_gpu']][rjob['model_name']]
                else:
                    tmp_oh = 10
                # tmp_oh = 0
                tmp = max(event_time - rjob['last_check_time']-tmp_oh, 0)
                rjob['total_executed_time'] = rjob['total_executed_time'] + event_time - rjob['last_check_time']
                rjob['remaining_iteration'] -= tmp/rjob['iteration_time']
                rjob['last_check_time'] = event_time
                rjob['remaining_time'] = rjob['remaining_iteration']*rjob['iteration_time']
                if FLAGS.schedule == 'shortest-expected':
                    rjob['remaining_expected'] = cal_shortest_expected_remaining(JOBS.job_dist_data, rjob['total_executed_time'])
                if gputime:
                    rjob['remaining_gputime'] = rjob['remaining_time'] * rjob['num_gpu']
            elif 'PENDING' == rjob['status']:
                tmp = event_time - rjob['last_check_time']
                rjob['pending_time'] = rjob['pending_time'] + tmp
                rjob['last_check_time'] = event_time
            elif 'END' == rjob['status']: #almost impossible
                JOBS.runnable_jobs.remove(rjob)
                pass
        #sort jobs with shortest first
        if FLAGS.schedule == 'shortest-expected':
            JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_expected'))
        else:
            if gputime:
                JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_gputime'))
            else:
                JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_time'))

        run_jobs = list()
        preempt_jobs = list()
        #scan / execute jobs one by one
        CLUSTER.empty_infra()
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                if 'placements' in rjob: 
                    del rjob['placements'][:]
            ret = try_get_job_res(rjob) 
            if True == ret:
                # rjob['status'] = 'RUNNING'
                # if 0 == rjob['start_time'] and 0 != rjob['submit_time']:
                #     rjob['start_time'] = event_time
                if sys.maxsize == rjob['start_time']:
                    rjob['start_time'] = event_time
                if rjob['status'] == 'PENDING':
                    run_jobs.append(rjob)

            else:
                # rjob['status'] = 'PENDING'
                if rjob['status'] == 'RUNNING':
                    preempt_jobs.append(rjob)
                continue

        for job in preempt_jobs:
            job['status'] = 'PENDING'
            job['preempt'] = int(job['preempt'] + 1)
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)

        # get the next end_event
        del end_events[:]
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                end_time = event_time + rjob['remaining_iteration']*rjob['iteration_time']
                tmp_dict = util.search_dict_list(end_events, 'time', end_time)
                if tmp_dict == None:
                    #not found, add the time into to job_events
                    tmp_dict = dict()
                    tmp_dict['time'] = end_time
                    tmp_dict['end_jobs'] = list()
                    tmp_dict['end_jobs'].append(rjob)
                    end_events.append(tmp_dict)
                else:
                    tmp_dict['end_jobs'].append(rjob)
        end_events.sort(key = lambda e:e.__getitem__('time'))


        LOG.checkpoint(event_time)

def dlas_sim_jobs(gputime=False, solve_starvation=0):
    '''
    Job's executed time -- priority queue
    Q0:[0, 30min)
    Q1:[30min,1h)
    Q2:[1h, 2h)
    Q3:[2h, 00)

    in each queue, jobs are scheduled in fit-first with FIFO
    how to avoid starvation?

    TODO:  2. add move_back for avoiding starvation
    '''
    end_events = list()
    next_job_jump = sys.maxsize
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_event = None
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_event = None
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']
        
        event_time = sys.maxsize
        event = dict()
        event['time'] = sys.maxsize
        if end_time < start_time:
            event_time = end_time
            event = end_event
        elif end_time > start_time:        
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        #decide if job_jump first or (start/end) first
        if event_time > next_job_jump:
            event_time = next_job_jump
            event = dict()
        
        print(start_time, end_time, next_job_jump)

        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                tmp = event_time - e_job['last_check_time']
                e_job['total_executed_time'] = e_job['total_executed_time'] + tmp
                CLUSTER.release_job_res(e_job)
                LOG.job_complete(e_job, event_time)
                # util.print_fn('---- job[%d] is completed' % e_job['job_idx'])
                JOBS.runnable_jobs.remove(e_job)
                JOBS.queues[e_job['q_id']].remove(e_job)

        #for new jobs, append to runnable jobs with pending status
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                JOBS.move_to_runnable(s_job)
                s_job['q_id'] = 0 #any new start job should be in Q0
                JOBS.queues[0].append(s_job)
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])
            #pop start event
            JOBS.job_events.pop(0)

        #update executed_time
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                if rjob['model_name'] in overhead_dict[rjob['num_gpu']]:
                    tmp_oh = overhead_dict[rjob['num_gpu']][rjob['model_name']]
                else:
                    tmp_oh = 10
                # tmp_oh=0
                tmp = max(event_time - rjob['last_check_time']-tmp_oh, 0)
                rjob['remaining_iteration'] -= tmp/rjob['iteration_time']
                rjob['total_executed_time'] = rjob['total_executed_time'] + event_time - rjob['last_check_time']
                rjob['executed_time'] = rjob['executed_time'] + event_time - rjob['last_check_time'] # decide job priority queue
                rjob['last_check_time'] = event_time

                #check demotion
                j_gt = 0
                if gputime:
                    j_gt = rjob['executed_time'] * rjob['num_gpu']
                else:
                    j_gt = rjob['executed_time']
                cur_qid = rjob['q_id']
                if cur_qid < int(JOBS.num_queue - 1): #not for the last queue 
                    if j_gt >= JOBS.queue_limit[cur_qid]:
                        rjob['q_id'] = int(cur_qid + 1)
                        JOBS.queues[rjob['q_id']].append(rjob)
                        JOBS.queues[cur_qid].remove(rjob)
                        print("job %d demote to Q%d" % (rjob['job_idx'], rjob['q_id']))

                if FLAGS.schedule == 'dlas-gpu-gittins': 
                    # rjob['rank'] = cal_r_gittins_index(JOBS.job_dist_data, j_gt)
                    rjob['rank'] = get_gittins_index(j_gt)

            elif 'PENDING' == rjob['status']:
                tmp = event_time - rjob['last_check_time']
                rjob['last_check_time'] = event_time
                rjob['pending_time'] = rjob['pending_time'] + tmp #this is the total pending_time
                if rjob['executed_time'] > 0: # if not started yet, job is always in Q0 and no need to push_back
                    rjob['last_pending_time'] = rjob['last_pending_time'] + tmp #this is the total pending_time
                #Q0 job no need to push_back, and must be a runned 
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] > 0 and rjob['executed_time'] > 0:
                    if rjob['last_pending_time'] >= int(rjob['executed_time'] * solve_starvation):
                        rjob['executed_time'] = 0
                        rjob['last_pending_time'] = 0
                        JOBS.queues[0].append(rjob)
                        JOBS.queues[rjob['q_id']].remove(rjob)
                        rjob['q_id'] = 0
                        rjob['promote'] = int(rjob['promote'] + 1)

                if FLAGS.schedule == 'dlas-gpu-gittins': 
                    if gputime:
                        j_gt = rjob['executed_time'] * rjob['num_gpu']
                    else:
                        j_gt = rjob['executed_time']
                    # rjob['rank'] = cal_r_gittins_index(JOBS.job_dist_data, j_gt)
                    rjob['rank'] = get_gittins_index(j_gt)

            elif 'END' == rjob['status']: # won't happen
                JOBS.runnable_jobs.remove(rjob)
                # util.print_fn('---- job[%d] completed' % rjob['job_idx'])
                pass

        #push job to their new queue
        # JOBS.update_priority_queues(gputime)

        ''' schedule jobs in each queue '''
        #empty_cluster resource
        CLUSTER.empty_infra()
        # for "count" placement
        run_jobs = list()
        preempt_jobs = list()

        # if FLAGS.schedule == 'dlas-gpu-gittins': 
        #     q = JOBS.queues[0]
        #     q.sort(key = lambda e:(e.__getitem__('rank'), e.__getitem__('r_submit_time')), reverse=True)

        for queue in JOBS.queues:
            if FLAGS.schedule == 'dlas-gpu-gittins': 
                queue.sort(key = lambda e:(e.__getitem__('rank'), e.__getitem__('r_submit_time')), reverse=True)
            for job in queue:
                if CLUSTER.free_gpu >= job['num_gpu']:
                    #should run
                    if job['status'] == 'PENDING':                   
                        #not running
                        run_jobs.append(job)
                    CLUSTER.free_gpu = int(CLUSTER.free_gpu - job['num_gpu'])
                else:
                    #should NOT run
                    if job['status'] == 'RUNNING':                   
                        #running
                        preempt_jobs.append(job)
                    continue
                # if 'RUNNING' == job['status']:
                #     if 'placements' in job:
                #         del job['placements'][:]
                # ret = try_get_job_res(job)
                # if True == ret:
                #     if job['status']=='PENDING':
                #         run_jobs.append(job)
                # else:
                #     if job['status'] == 'RUNNING':
                #         preempt_jobs.append(job)
        
        for job in preempt_jobs:
            job['status'] = 'PENDING'
            # if job['q_id'] == 0:
            #     job['preempt'] = int(job['preempt'] + 1)
            job['preempt'] = int(job['preempt'] + 1)
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)
            if job['start_time'] == sys.maxsize:
                job['start_time'] = event_time


        #sort based on the job start time
        for queue in JOBS.queues:
            #job there are many students            
            pending_job = list()
            for job in queue: 
                # if sys.maxsize == job['start_time'] and job['status'] == 'PENDING':
                if job['status'] == 'PENDING':
                    pending_job.append(job)
                    # print(job['job_idx'])
            for job in pending_job: 
                queue.remove(job)
            queue.extend(pending_job)


        #for fit-job-first, move small running jobs in front of large pending jobs in each queue
        # for queue in JOBS.queues:
        #     num_j = len(queue)
        #     #find the first pending job
        #     for i in range(num_j):
        #         if queue[i]['status'] == 'PENDING':
        #             break
        #     first_pending_idx = i

        #     #picking running_after_pending_jobs
        #     run_after_pending = list()
        #     for j in range(first_pending_idx, num_j):
        #         if queue[j]['status'] == 'RUNNING': 
        #             run_after_pending.append(queue[j])

        #     #reinsert all those jobs
        #     for job in run_after_pending: 
        #         queue.remove(job)
        #     for job in run_after_pending:
        #         queue.insert(i, job)
        #         i = int(i + 1)


        # for queue in JOBS.queues:
        #     for job in queue:
        #         if 'RUNNING' == job['status']:
        #             if 'placements' in job:
        #                 del job['placements'][:]
        #             job['status'] = 'PENDING'
        #         ret = try_get_job_res(job)
        #         if True == ret:
        #             job['status'] = 'RUNNING'
        #             if 0 == job['start_time'] and 0 != job['submit_time']:
        #                 job['start_time'] = event_time
        #         else:
        #             job['status'] = 'PENDING'
        #             continue



        #update end events and sort, and get the most recent one
        del end_events[:]
        # for rjob in JOBS.runnable_jobs:
        #     if 'RUNNING' == rjob['status']:
        #         remaining_time = rjob['duration'] - rjob['total_executed_time']
        #         end_time = int(event_time + remaining_time)
        #         tmp_dict = util.search_dict_list(end_events, 'time', end_time)
        #         if tmp_dict == None:
        #             #not found, add the time into to job_events
        #             tmp_dict = dict()
        #             tmp_dict['time'] = end_time
        #             tmp_dict['end_jobs'] = list()
        #             tmp_dict['end_jobs'].append(rjob)
        #             end_events.append(tmp_dict)
        #         else:
        #             tmp_dict['end_jobs'].append(rjob)
        # end_events.sort(key = lambda e:e.__getitem__('time'))
        min_end_time = sys.maxsize
        tmp_end_event = dict()
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                remaining_time = rjob['remaining_iteration'] * rjob['iteration_time']
                end_time = event_time + remaining_time
                if end_time < min_end_time:
                    tmp_end_event['time'] = end_time
                    tmp_end_event['end_jobs'] = list()
                    tmp_end_event['end_jobs'].append(rjob)
                    min_end_time = end_time
                elif min_end_time == end_time:
                    tmp_end_event['end_jobs'].append(rjob)
        if min_end_time < sys.maxsize:
            end_events.append(tmp_end_event)

        # what's the closest queue_jump (demotion, and promotion) among all the jobs
        next_job_jump = sys.maxsize
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                qid = rjob['q_id']
                if qid < int(JOBS.num_queue - 1):
                    if gputime:
                        print('jump_time: ', JOBS.queue_limit[qid]/rjob['num_gpu'], rjob['executed_time'])
                        jump_time = math.ceil(JOBS.queue_limit[qid]/rjob['num_gpu']) - rjob['executed_time'] + event_time
                    else:
                        jump_time = JOBS.queue_limit[qid] - rjob['executed_time'] + event_time
                    if jump_time < next_job_jump:
                        next_job_jump = jump_time

            elif 'PENDING' == rjob['status']: # when pending job will be push back to Q0
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] and rjob['executed_time'] > 0:
                    diff_time = int(rjob['executed_time'] * solve_starvation - rjob['last_pending_time'])
                    if diff_time > 0:
                        jump_time = int(diff_time + event_time)
                        if jump_time < next_job_jump:
                            next_job_jump = jump_time
                    
        

        LOG.checkpoint(event_time)


def get_gittins_index(a):
    job_info = JOBS.job_dist_data
    if a > job_info['data'][-2]:
        return 0
    idx = next(x[0] for x in enumerate(job_info['data']) if x[1] > a)
    return job_info['gittins'][idx]


def gittins_sim_jobs(job_dist_data, gputime=False, static_delta=True):
    '''
    gittins index
    '''
    solve_starvation = 0
    end_events = list()
    next_job_jump = sys.maxsize
    next_gittins_unit = copy.copy(JOBS.gittins_delta)

    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_event = None
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_event = None
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']
        
        event_time = sys.maxsize
        event = dict()
        event['time'] = sys.maxsize
        if end_time < start_time:
            event_time = end_time
            event = end_event
        elif end_time > start_time:        
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        #decide if job_jump first or (start/end) first
        # if event_time > next_job_jump:
        #     event_time = next_job_jump
        #     event = dict()

        #check the next gittins_unit
        if event_time > next_gittins_unit:
            event_time = next_gittins_unit
            event = dict()

        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                CLUSTER.release_job_res(e_job)
                LOG.job_complete(e_job, event_time)
                # util.print_fn('---- job[%d] is completed' % e_job['job_idx'])
                JOBS.runnable_jobs.remove(e_job)
                # JOBS.queues[e_job['q_id']].remove(e_job)

        #for new jobs, append to runnable jobs with pending status
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                JOBS.move_to_runnable(s_job)
                s_job['q_id'] = 0 #any new start job should be in Q0
                # JOBS.queues[0].append(s_job)
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])
            #pop start event
            JOBS.job_events.pop(0)

        #update executed_time
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time']) 
                rjob['total_executed_time'] = int(rjob['total_executed_time'] + tmp)
                rjob['executed_time'] = int(rjob['executed_time'] + tmp) # decide job priority queue
                rjob['last_check_time'] = event_time

                #check demotion
                j_gt = 0
                if gputime:
                    j_gt = int(rjob['executed_time'] * rjob['num_gpu'])
                else:
                    j_gt = int(rjob['executed_time'])
                # cur_qid = rjob['q_id']
                # if cur_qid < int(JOBS.num_queue - 1): #not for the last queue 
                #     if j_gt >= JOBS.queue_limit[cur_qid]:
                #         rjob['q_id'] = int(cur_qid + 1)
                #         JOBS.queues[rjob['q_id']].append(rjob)
                #         JOBS.queues[cur_qid].remove(rjob)
                #         print("job %d demote to Q%d" % (rjob['job_idx'], rjob['q_id']))

                # rjob['rank'] = cal_r_gittins_index(job_dist_data, j_gt)
                rjob['rank'] = get_gittins_index(j_gt)

            elif 'PENDING' == rjob['status']:
                tmp = int(event_time - rjob['last_check_time']) 
                rjob['last_check_time'] = event_time
                rjob['pending_time'] = int(rjob['pending_time'] + tmp) #this is the total pending_time
                if rjob['executed_time'] > 0: # if not started yet, job is always in Q0 and no need to push_back
                    rjob['last_pending_time'] = int(rjob['last_pending_time'] + tmp) #this is the total pending_time
                #Q0 job no need to push_back, and must be a runned 
                if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] > 0 and rjob['executed_time'] > 0:
                    if rjob['last_pending_time'] >= int(rjob['executed_time'] * solve_starvation):
                        rjob['executed_time'] = 0
                        rjob['last_pending_time'] = 0
                        JOBS.queues[0].append(rjob)
                        JOBS.queues[rjob['q_id']].remove(rjob)
                        rjob['q_id'] = 0
                        rjob['promote'] = int(rjob['promote'] + 1)

                j_gt = rjob['executed_time']
                # rjob['rank'] = cal_r_gittins_index(job_dist_data, j_gt)
                rjob['rank'] = get_gittins_index(j_gt)

            elif 'END' == rjob['status']: # won't happen
                JOBS.runnable_jobs.remove(rjob)
                # util.print_fn('---- job[%d] completed' % rjob['job_idx'])
                pass

        #push job to their new queue
        # JOBS.update_priority_queues(gputime)

        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('rank'))

        ''' schedule jobs in each queue '''
        #empty_cluster resource
        CLUSTER.empty_infra()
        # for "count" placement
        run_jobs = list()
        preempt_jobs = list()
        # for queue in JOBS.queues:
        #     queue.sort(key = lambda e:e.__getitem__('rank'))
        #     for job in queue:
        #         if CLUSTER.free_gpu >= job['num_gpu']:
        #             #should run
        #             if job['status'] == 'PENDING':                   
        #                 #not running
        #                 run_jobs.append(job)
        #             CLUSTER.free_gpu = int(CLUSTER.free_gpu - job['num_gpu'])
        #         else:
        #             #should NOT run
        #             if job['status'] == 'RUNNING':                   
        #                 #running
        #                 preempt_jobs.append(job)
        #             continue

        for job in JOBS.runnable_jobs:
            if CLUSTER.free_gpu >= job['num_gpu']:
                #should run
                if job['status'] == 'PENDING':                   
                    #not running
                    run_jobs.append(job)
                CLUSTER.free_gpu = int(CLUSTER.free_gpu - job['num_gpu'])
            else:
                #should NOT run
                if job['status'] == 'RUNNING':                   
                    #running
                    preempt_jobs.append(job)
                continue

        for job in preempt_jobs:
            job['status'] = 'PENDING'
            # if job['q_id'] == 0:
            #     job['preempt'] = int(job['preempt'] + 1)
            job['preempt'] = int(job['preempt'] + 1)
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)
            if job['start_time'] == sys.maxsize:
                job['start_time'] = event_time


        # #sort based on the job start time
        # for queue in JOBS.queues:
        #     #job there are many students            
        #     pending_job = list()
        #     for job in queue: 
        #         # if sys.maxsize == job['start_time'] and job['status'] == 'PENDING':
        #         if job['status'] == 'PENDING':
        #             pending_job.append(job)
        #             # print(job['job_idx'])
        #     for job in pending_job: 
        #         queue.remove(job)
        #     queue.extend(pending_job)


        #update end events and sort, and get the most recent one
        del end_events[:]
        # for rjob in JOBS.runnable_jobs:
        #     if 'RUNNING' == rjob['status']:
        #         remaining_time = rjob['duration'] - rjob['total_executed_time']
        #         end_time = int(event_time + remaining_time)
        #         tmp_dict = util.search_dict_list(end_events, 'time', end_time)
        #         if tmp_dict == None:
        #             #not found, add the time into to job_events
        #             tmp_dict = dict()
        #             tmp_dict['time'] = end_time
        #             tmp_dict['end_jobs'] = list()
        #             tmp_dict['end_jobs'].append(rjob)
        #             end_events.append(tmp_dict)
        #         else:
        #             tmp_dict['end_jobs'].append(rjob)
        # end_events.sort(key = lambda e:e.__getitem__('time'))
        min_end_time = sys.maxsize
        tmp_end_event = dict()
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                remaining_time = rjob['duration'] - rjob['total_executed_time']
                end_time = int(event_time + remaining_time)
                if end_time < min_end_time:
                    tmp_end_event['time'] = end_time
                    tmp_end_event['end_jobs'] = list()
                    tmp_end_event['end_jobs'].append(rjob)
                    min_end_time = end_time
                elif min_end_time == end_time:
                    tmp_end_event['end_jobs'].append(rjob)
        if min_end_time < sys.maxsize:
            end_events.append(tmp_end_event)

        # what's the closest queue_jump (demotion, and promotion) among all the jobs
        # next_job_jump = sys.maxsize
        # for rjob in JOBS.runnable_jobs:
        #     if 'RUNNING' == rjob['status']:
        #         qid = rjob['q_id']
        #         if qid < int(JOBS.num_queue - 1):
        #             if gputime:
        #                 jump_time = int(math.ceil((JOBS.queue_limit[qid] - rjob['executed_time'])/rjob['num_gpu']) + event_time)
        #             else:
        #                 jump_time = int(JOBS.queue_limit[qid] - rjob['executed_time'] + event_time)
        #             if jump_time < next_job_jump:
        #                 next_job_jump = jump_time

        #     elif 'PENDING' == rjob['status']: # when pending job will be push back to Q0
        #         if solve_starvation > 0 and rjob['q_id'] > 0 and rjob['total_executed_time'] and rjob['executed_time'] > 0:
        #             diff_time = int(rjob['executed_time'] * solve_starvation - rjob['last_pending_time'])
        #             if diff_time > 0:
        #                 jump_time = int(diff_time + event_time)
        #                 if jump_time < next_job_jump:
        #                     next_job_jump = jump_time
                    
        
        next_gittins_unit += event_time
        LOG.checkpoint(event_time)

def multi_resource_blossom_same_sim_jobs(gputime=False, know_duration=True, ordering=1, blossom=True):
    '''
    new jobs are added to the end of the ending queue
    but in the queue, shortest (gpu) job first be served
    and pack other jobs with the same #GPU according to 
    graph matching
    '''
    end_events = list()
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_time = sys.float_info.max
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_time = sys.float_info.max
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']

        if math.isclose(end_time, start_time, abs_tol=1e-4) and end_time < sys.maxsize:
            event_time = start_time
            event = JOBS.job_events.pop(0)
            event['end_jobs'] = end_events[0]['end_jobs']
        elif end_time < start_time:
            event_time = end_time
            event = end_events[0]
        elif end_time > start_time:        
            event_time = start_time
            # print("start-time %d, end_time %d" % (start_time, end_time))
            event = JOBS.job_events.pop(0)

        assert math.isclose(event_time, event['time'], abs_tol=1e-4)
        print("Event Time: ", event_time)

        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                tmp = float(event_time - e_job['last_check_time']) 
                e_job['total_executed_time'] = float(e_job['total_executed_time'] + tmp)
                #job completes
                CLUSTER.release_job_res(e_job)
                # CLUSTER.release_gpus(e_job)
                LOG.job_complete(e_job, event_time)
                JOBS.runnable_jobs.remove(e_job)
                # print("11111111111", e_job['job_idx'], e_job['num_gpu'], e_job['duration'], e_job['end_time']-e_job['start_time'])


        #for new-start jobs, add to runnable
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                #add into runnable list with pending status
                JOBS.move_to_runnable(s_job)

                s_job['remaining_time'] = s_job['duration']
                s_job['remaining_gputime'] = float(s_job['remaining_time'] * s_job['num_gpu'])
                s_job['total_executed_time'] = 0.0
                s_job['total_executed_gputime'] = 0.0
                s_job['calc_executed_time'] = 0.0
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])

        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                tmp_oh = rjob['overhead']
                tmp = max(event_time - rjob['last_check_time']-tmp_oh, 0) 
                rjob['remaining_iteration'] -= tmp/rjob['iteration_time_cur']
                rjob['calc_executed_time'] = float(rjob['calc_executed_time'] + tmp/rjob['iteration_time_cur']*rjob['iteration_time'])
                rjob['total_executed_time'] = float(rjob['total_executed_time'] + event_time - rjob['last_check_time'])
                rjob['last_check_time'] = event_time
                rjob['remaining_time'] = rjob['remaining_iteration'] * rjob['iteration_time']
                if gputime:
                    rjob['remaining_gputime'] = float(rjob['remaining_time'] * rjob['num_gpu'])
                    if not know_duration:
                        rjob['total_executed_gputime'] = float(rjob['total_executed_time'] * rjob['num_gpu'])
                # print(event_time, 'check: running ', rjob['job_idx'], rjob['num_gpu'], rjob['total_executed_time'], rjob['calc_executed_time'], rjob['remaining_time'], rjob['duration'], rjob['pending_time'], rjob['iteration_time_cur'], rjob['iteration_time'])
            elif 'PENDING' == rjob['status']:
                tmp = float(event_time - rjob['last_check_time'])
                rjob['pending_time'] = float(rjob['pending_time'] + tmp)
                rjob['last_check_time'] = event_time
                # print(event_time, 'check: pending ', rjob['job_idx'], rjob['num_gpu'], rjob['total_executed_time'], rjob['calc_executed_time'], rjob['remaining_time'], rjob['duration'], rjob['pending_time'], rjob['iteration_time_cur'], rjob['iteration_time'])
            elif 'END' == rjob['status']: #almost impossible
                JOBS.runnable_jobs.remove(rjob)
                # print(event_time, 'check: end ', rjob['job_idx'], rjob['total_executed_time'], rjob['duration'])
                pass
            if rjob['status'] != 'END':
                if know_duration: 
                    if gputime:
                        rjob['sort_val']=rjob['remaining_gputime']
                    else:
                        rjob['sort_val']=rjob['remaining_time']
                else:
                    if gputime:
                        rjob['sort_val']=rjob['total_executed_gputime']
                    else:
                        rjob['sort_val']=rjob['total_executed_time']
        #sort jobs with shortest first
        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('sort_val'))
        
        run_jobs = list()
        preempt_jobs = list()
        GPU_num_job = dict()
        GPU_chosen_job = dict()
        GPU_nums = dict()
        required_gpu = 0
        for rjob in JOBS.runnable_jobs:
            # assert rjob['packing_used'] < 2
            rjob['packing_used'] = 0
            num_gpu = rjob['num_gpu']
            if num_gpu not in GPU_num_job:
                GPU_num_job[num_gpu] = list()
            GPU_num_job[num_gpu].append(rjob)
            if num_gpu not in GPU_chosen_job:
                GPU_chosen_job[num_gpu] = 0
                GPU_nums[num_gpu] = 0
        #scan / execute jobs one by one
        CLUSTER.empty_infra()
        for rjob in JOBS.runnable_jobs: 
            if rjob['packing_used'] == 1:
                continue
            ret = try_get_job_res(rjob, True)
            num_gpu = rjob['num_gpu']
            if ret == True:
                up_bd = min(GPU_chosen_job[num_gpu]+FLAGS.packing_num, len(GPU_num_job[num_gpu]))
                GPU_nums[num_gpu] += 1
                for tmp_id in range(GPU_chosen_job[num_gpu], up_bd):
                    GPU_num_job[num_gpu][tmp_id]['packing_used']=1
                GPU_chosen_job[num_gpu] = up_bd
        for key in GPU_num_job.keys():
            GPU_num_job[key] = GPU_num_job[key][:GPU_chosen_job[key]]
            required_gpu += GPU_chosen_job[key]*key

        # print('before packing')
        # for key,value in GPU_num_job.items():
        #     print(key, 'GPU(s): ', len(value))
        #     print([rjob['job_idx'] for rjob in value])

        # time_before = time.time()
        # matching algorithm
        if blossom==True:
            packings = Blossom_Same.run(GPU_num_job, CLUSTER.num_gpu, ordering)
        else:
            packings = dict()
            for key in GPU_num_job.keys():
                packings[key] = list()
                for i in range(GPU_nums[key]):
                    packing = _Packing(GPU_num_job[key][i*FLAGS.packing_num])
                    for j in range(1, FLAGS.packing_num):
                        if j+i*FLAGS.packing_num>=GPU_chosen_job[key]:
                            break
                        rpacking = _Packing(GPU_num_job[key][j+i*FLAGS.packing_num])
                        if required_gpu>CLUSTER.num_gpu:
                            packing.add_job(rpacking)
                            required_gpu -= key
                        else:
                            packings[key].append(rpacking)
                    packings[key].append(packing)
        # print('after packing', time.time()-time_before)
        # for key, value in packings.items():
        #     for packing in value:
        #         print([packing_job.job_idx for packing_job in packing.packing_jobs], end=':::')
        #         print('gpu', [packing_job.num_gpu for packing_job in packing.packing_jobs])
        if FLAGS.autopack:
            new_packing = list()
            for key, value in packings.items():
                for packing in value:
                    itertime_all = packing.calc_iteration_time()
                    itertime_sum = sum([job.iteration_time for job in packing.packing_jobs])
                    if itertime_all/itertime_sum >1:
                        print('unpack: ', [job.job_idx for job in packing.packing_jobs])
                        for job in packing.packing_jobs:
                            rjob = JOBS.find_runnable_job(job.job_idx)
                            new_packing.append((key, _Packing(rjob), rjob['sort_val']))
                    else:
                        sort_val = min([JOBS.find_runnable_job(job.job_idx)['sort_val'] for job in packing.packing_jobs])
                        new_packing.append((key, packing, sort_val))
            new_packing.sort(key=lambda e:e[2])
            CLUSTER.empty_infra()
            packings = dict()
            for packing in new_packing:
                rjob = JOBS.find_runnable_job(packing[1].packing_jobs[0].job_idx)
                if 'RUNNING'==rjob['status']:
                    if 'placements' in rjob:
                        del rjob['placements'][:]
                ret = try_get_job_res(rjob, True)
                if ret==True:
                    if packing[0] not in packings:
                        packings[packing[0]] = list()
                    packings[packing[0]].append(packing[1])

        # print("_______________________new placement____________________")
        # deal with the packing plan
        CLUSTER.empty_infra()

        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('num_gpu'), reverse=True)
        tmp_job_placement = dict()
        for rjob in JOBS.runnable_jobs:
            # print("after packing: ", rjob['job_idx'], rjob['placements'])
            if 'RUNNING' == rjob['status']:
                if 'placements' in rjob: 
                    del rjob['placements'][:]
            ret = False
            for key, value in packings.items():
                for packing in value:
                    for packing_job in packing.packing_jobs:
                        if packing_job.job_idx == rjob['job_idx']:
                            packing_cur = packing
                            ret = True
                            break
                    if ret:
                        break
                if ret:
                    break
            if ret:
                rjob['iteration_time_cur'] = packing_cur.calc_iteration_time(ordering=ordering)
                rjob['packing'] = packing_cur
                rjob['overhead'] = 0
                for pjob_ in packing_cur.packing_jobs:
                    pjob = JOBS.find_runnable_job(pjob_.job_idx)
                    if pjob['model_name'] in overhead_dict[rjob['num_gpu']]:
                        rjob['overhead'] += overhead_dict[rjob['num_gpu']][pjob['model_name']]
                    else:
                        rjob['overhead'] += 10
                    # rjob['overhead'] = 0
                # print(rjob['job_idx'], [pjob.job_idx for pjob in packing_cur.packing_jobs], rjob['iteration_time'], rjob['iteration_time_cur'])
                # if rjob['iteration_time']/rjob['iteration_time_cur']>len(packing_cur.packing_jobs):
                    # print("111111111", rjob['job_idx'], rjob['iteration_time_cur']/rjob['iteration_time'])
                # print("11111111111111 job: ", rjob['job_idx'], rjob['num_gpu'], rjob['iteration_time_cur'], rjob['iteration_time'], (rjob['iteration_time_cur']-rjob['iteration_time'])/rjob['iteration_time'])
                # print([pjob.job_idx for pjob in packing_cur.best_permutation])
                if rjob['job_idx'] in tmp_job_placement:
                    rjob['placements'] = tmp_job_placement[rjob['job_idx']]
                    # print("other")
                else:
                    ret_1 = try_get_job_res(rjob)
                    if not ret_1:
                        print(f"job {rjob['job_idx']} is unable to place")
                        if rjob['status'] == 'RUNNING':
                            preempt_jobs.append(rjob)
                        continue
                    # assert ret_1==True
                    for packing_job in packing_cur.packing_jobs:
                        tmp_job_placement[packing_job.job_idx] = copy.deepcopy(rjob['placements'])
                    # print('first')
                # print(rjob['placements'])
                if sys.maxsize == rjob['start_time']:
                    rjob['start_time'] = event_time
                if rjob['status'] == 'PENDING':
                    run_jobs.append(rjob)
            else:
                if rjob['status'] == 'RUNNING':
                    preempt_jobs.append(rjob)
                continue

        for job in preempt_jobs:
            job['status'] = 'PENDING'
            job['preempt'] = int(job['preempt'] + 1)
            # job['packing_used'] = 0
        # print("-----placement-------")
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)
            # job['packing_used'] = 1
            # print(job['placements'])
            #        

        # get the next end_event
        del end_events[:]
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                end_time = float(event_time + rjob['remaining_iteration']*rjob['iteration_time_cur'])
                # print(event_time, rjob['job_idx'], rjob['remaining_time'], rjob['iteration_time'], rjob['iteration_time_cur'], end_time)
                tmp_dict = util.search_dict_list(end_events, 'time', end_time)
                if tmp_dict == None:
                    #not found, add the time into to job_events
                    tmp_dict = dict()
                    tmp_dict['time'] = end_time
                    tmp_dict['end_jobs'] = list()
                    tmp_dict['end_jobs'].append(rjob)
                    end_events.append(tmp_dict)
                else:
                    tmp_dict['end_jobs'].append(rjob)
        end_events.sort(key = lambda e:e.__getitem__('time'))

        LOG.checkpoint(event_time)     

def multi_resource_blossom_same_dlas_sim_jobs(gputime=False, know_duration=True, ordering=1, blossom=True):
    '''
    new jobs are added to the end of the ending queue
    but in the queue, shortest (gpu) job first be served
    and pack other jobs with the same #GPU according to 
    graph matching
    '''
    end_events = list()
    next_job_jump = sys.maxsize
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_time = sys.float_info.max
        start_event = None
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_time = sys.float_info.max
        end_event = None
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']

        event_time = sys.maxsize
        event = dict()
        event['time'] = sys.maxsize
        if math.isclose(end_time, start_time, abs_tol=1e-4) and end_time < sys.maxsize:
            event_time = start_time
            # event = JOBS.job_events.pop(0)
            event = start_event
            event['end_jobs'] = end_events[0]['end_jobs']
        elif end_time < start_time:
            event_time = end_time
            # event = end_events[0]
            event = end_event
        elif end_time > start_time:        
            event_time = start_time
            # print("start-time %d, end_time %d" % (start_time, end_time))
            # event = JOBS.job_events.pop(0)
            event = start_event

        assert math.isclose(event_time, event['time'], abs_tol=1e-4)

        if event_time>next_job_jump:
            event_time = next_job_jump
            event = dict()
        print("Event Time: ", event_time)

        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                tmp = float(event_time - e_job['last_check_time']) 
                e_job['total_executed_time'] = float(e_job['total_executed_time'] + tmp)
                #job completes
                CLUSTER.release_job_res(e_job)
                # CLUSTER.release_gpus(e_job)
                LOG.job_complete(e_job, event_time)
                JOBS.runnable_jobs.remove(e_job)
                JOBS.queues[e_job['q_id']].remove(e_job)
                # print("11111111111", e_job['job_idx'], e_job['num_gpu'], e_job['duration'], e_job['end_time']-e_job['start_time'])


        #for new-start jobs, add to runnable
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                #add into runnable list with pending status
                JOBS.move_to_runnable(s_job)

                s_job['remaining_time'] = s_job['duration']
                s_job['remaining_gputime'] = float(s_job['remaining_time'] * s_job['num_gpu'])
                s_job['total_executed_time'] = 0.0
                s_job['total_executed_gputime'] = 0.0
                s_job['calc_executed_time'] = 0.0
                s_job['q_id'] = 0
                JOBS.queues[0].append(s_job)
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])
            JOBS.job_events.pop(0)

        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                tmp_oh = rjob['overhead']
                # tmp_oh = 0
                tmp = max(event_time - rjob['last_check_time']-tmp_oh, 0) 
                rjob['remaining_iteration'] -= tmp/rjob['iteration_time_cur']
                rjob['calc_executed_time'] = float(rjob['calc_executed_time'] + tmp/rjob['iteration_time_cur']*rjob['iteration_time'])
                rjob['total_executed_time'] = float(rjob['total_executed_time'] + event_time - rjob['last_check_time'])
                rjob['last_check_time'] = event_time
                rjob['remaining_time'] = rjob['remaining_iteration'] * rjob['iteration_time']
                if gputime:
                    rjob['remaining_gputime'] = float(rjob['remaining_time'] * rjob['num_gpu'])
                    if not know_duration:
                        rjob['total_executed_gputime'] = float(rjob['total_executed_time'] * rjob['num_gpu'])
                cur_qid = rjob['q_id']
                j_gt = 0
                if gputime:
                    j_gt = rjob['total_executed_time'] * rjob['num_gpu']
                else:
                    j_gt = rjob['total_executed_time']
                if cur_qid<int(JOBS.num_queue-1):
                    if j_gt >= JOBS.queue_limit[cur_qid]:
                        rjob['q_id'] = int(cur_qid+1)
                        JOBS.queues[rjob['q_id']].append(rjob)
                        JOBS.queues[cur_qid].remove(rjob)
                        print("job %d demote to Q%d" % (rjob['job_idx'], rjob['q_id']))
                # print(event_time, 'check: running ', rjob['job_idx'], rjob['num_gpu'], rjob['total_executed_time'], rjob['calc_executed_time'], rjob['remaining_time'], rjob['duration'], rjob['pending_time'], rjob['iteration_time_cur'], rjob['iteration_time'])
            elif 'PENDING' == rjob['status']:
                tmp = float(event_time - rjob['last_check_time'])
                rjob['pending_time'] = float(rjob['pending_time'] + tmp)
                rjob['last_check_time'] = event_time
                # print(event_time, 'check: pending ', rjob['job_idx'], rjob['num_gpu'], rjob['total_executed_time'], rjob['calc_executed_time'], rjob['remaining_time'], rjob['duration'], rjob['pending_time'], rjob['iteration_time_cur'], rjob['iteration_time'])
            elif 'END' == rjob['status']: #almost impossible
                JOBS.runnable_jobs.remove(rjob)
                # print(event_time, 'check: end ', rjob['job_idx'], rjob['total_executed_time'], rjob['duration'])
                pass
            if rjob['status'] != 'END':
                if know_duration: 
                    if gputime:
                        rjob['sort_val']=rjob['remaining_gputime']
                    else:
                        rjob['sort_val']=rjob['remaining_time']
                else:
                    if gputime:
                        rjob['sort_val']=rjob['total_executed_gputime']
                    else:
                        rjob['sort_val']=rjob['total_executed_time']
        #sort jobs with shortest first
        # JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('sort_val'))
        
        run_jobs = list()
        preempt_jobs = list()
        GPU_num_job = dict()
        GPU_chosen_job = dict()
        GPU_nums = dict()
        required_gpu = 0
        # for rjob in JOBS.runnable_jobs:
        for queue in JOBS.queues:
            for rjob in queue:
                # assert rjob['packing_used'] < 2
                rjob['packing_used'] = 0
                num_gpu = rjob['num_gpu']
                if num_gpu not in GPU_num_job:
                    GPU_num_job[num_gpu] = list()
                GPU_num_job[num_gpu].append(rjob)
                if num_gpu not in GPU_chosen_job:
                    GPU_chosen_job[num_gpu] = 0
                    GPU_nums[num_gpu] = 0
        #scan / execute jobs one by one
        CLUSTER.empty_infra()
        # for rjob in JOBS.runnable_jobs:
        for queue in JOBS.queues:
            for rjob in queue: 
                if rjob['packing_used'] == 1:
                    continue
                ret = try_get_job_res(rjob, True)
                num_gpu = rjob['num_gpu']
                if ret == True:
                    up_bd = min(GPU_chosen_job[num_gpu]+FLAGS.packing_num, len(GPU_num_job[num_gpu]))
                    GPU_nums[num_gpu] += 1
                    for tmp_id in range(GPU_chosen_job[num_gpu], up_bd):
                        GPU_num_job[num_gpu][tmp_id]['packing_used']=1
                    GPU_chosen_job[num_gpu] = up_bd
        for key in GPU_num_job.keys():
            GPU_num_job[key] = GPU_num_job[key][:GPU_chosen_job[key]]
            required_gpu += GPU_chosen_job[key]*key

        # print('before packing')
        # for key,value in GPU_num_job.items():
        #     print(key, 'GPU(s): ', len(value))
        #     print([rjob['job_idx'] for rjob in value])

        # time_before = time.time()
        # matching algorithm
        if blossom==True:
            packings = Blossom_Same.run(GPU_num_job, CLUSTER.num_gpu, ordering)
        else:
            packings = dict()
            for key in GPU_num_job.keys():
                packings[key] = list()
                for i in range(GPU_nums[key]):
                    packing = _Packing(GPU_num_job[key][i*FLAGS.packing_num])
                    for j in range(1, FLAGS.packing_num):
                        if j+i*FLAGS.packing_num>=GPU_chosen_job[key]:
                            break
                        rpacking = _Packing(GPU_num_job[key][j+i*FLAGS.packing_num])
                        if required_gpu>CLUSTER.num_gpu:
                            packing.add_job(rpacking)
                            required_gpu -= key
                        else:
                            packings[key].append(rpacking)
                    packings[key].append(packing)
        # print('after packing', time.time()-time_before)
        # for key, value in packings.items():
        #     for packing in value:
        #         print([packing_job.job_idx for packing_job in packing.packing_jobs], end=':::')
        #         print('gpu', [packing_job.num_gpu for packing_job in packing.packing_jobs])
        if FLAGS.autopack:
            new_packing = list()
            for key, value in packings.items():
                for packing in value:
                    itertime_all = packing.calc_iteration_time()
                    itertime_sum = sum([job.iteration_time for job in packing.packing_jobs])
                    if itertime_all/itertime_sum >1:
                        print('unpack: ', [job.job_idx for job in packing.packing_jobs])
                        for job in packing.packing_jobs:
                            rjob = JOBS.find_runnable_job(job.job_idx)
                            assert rjob in JOBS.queues[rjob['q_id']]
                            new_packing.append((key, _Packing(rjob), rjob['q_id'], JOBS.queues[rjob['q_id']].index(rjob)))
                    else:
                        # sort_val = min([JOBS.find_runnable_job(job.job_idx)['sort_val'] for job in packing.packing_jobs])
                        min_q_id = 1e8
                        min_index = 1e8
                        for job in packing.packing_jobs:
                            rjob = JOBS.find_runnable_job(job.job_idx)
                            assert rjob in JOBS.queues[rjob['q_id']]
                            cur_index = JOBS.queues[rjob['q_id']].index(rjob)
                            if rjob['q_id']<min_q_id or (rjob['q_id']==min_q_id and cur_index<min_index):
                                min_q_id = rjob['q_id']
                                min_index = cur_index
                        new_packing.append((key, packing, min_q_id, min_index))
            new_packing.sort(key=lambda e:(e[2],e[3]))
            CLUSTER.empty_infra()
            packings = dict()
            for packing in new_packing:
                rjob = JOBS.find_runnable_job(packing[1].packing_jobs[0].job_idx)
                if 'RUNNING'==rjob['status']:
                    if 'placements' in rjob:
                        del rjob['placements'][:]
                ret = try_get_job_res(rjob, True)
                if ret==True:
                    if packing[0] not in packings:
                        packings[packing[0]] = list()
                    packings[packing[0]].append(packing[1])

        # print("_______________________new placement____________________")
        # deal with the packing plan
        CLUSTER.empty_infra()

        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('num_gpu'), reverse=True)
        tmp_job_placement = dict()
        for rjob in JOBS.runnable_jobs:
            # print("after packing: ", rjob['job_idx'], rjob['placements'])
            if 'RUNNING' == rjob['status']:
                if 'placements' in rjob: 
                    del rjob['placements'][:]
            ret = False
            for key, value in packings.items():
                for packing in value:
                    for packing_job in packing.packing_jobs:
                        if packing_job.job_idx == rjob['job_idx']:
                            packing_cur = packing
                            ret = True
                            break
                    if ret:
                        break
                if ret:
                    break
            if ret:
                rjob['iteration_time_cur'] = packing_cur.calc_iteration_time(ordering=ordering)
                rjob['packing'] = packing_cur
                rjob['overhead'] = 0
                for pjob_ in packing_cur.packing_jobs:
                    pjob = JOBS.find_runnable_job(pjob_.job_idx)
                    if pjob['model_name'] in overhead_dict[rjob['num_gpu']]:
                        rjob['overhead'] += overhead_dict[rjob['num_gpu']][pjob['model_name']]
                    else:
                        rjob['overhead'] += 10
                # print(rjob['job_idx'], [pjob.job_idx for pjob in packing_cur.packing_jobs], rjob['iteration_time'], rjob['iteration_time_cur'])
                # if rjob['iteration_time']/rjob['iteration_time_cur']>len(packing_cur.packing_jobs):
                    # print("111111111", rjob['job_idx'], rjob['iteration_time_cur']/rjob['iteration_time'])
                # print("11111111111111 job: ", rjob['job_idx'], rjob['num_gpu'], rjob['iteration_time_cur'], rjob['iteration_time'], (rjob['iteration_time_cur']-rjob['iteration_time'])/rjob['iteration_time'])
                # print([pjob.job_idx for pjob in packing_cur.best_permutation])
                if rjob['job_idx'] in tmp_job_placement:
                    rjob['placements'] = tmp_job_placement[rjob['job_idx']]
                    # print("other")
                else:
                    ret_1 = try_get_job_res(rjob)
                    if not ret_1:
                        print(f"job {rjob['job_idx']} is unable to place")
                        if rjob['status'] == 'RUNNING':
                            preempt_jobs.append(rjob)
                        continue
                    # assert ret_1==True
                    for packing_job in packing_cur.packing_jobs:
                        tmp_job_placement[packing_job.job_idx] = copy.deepcopy(rjob['placements'])
                    # print('first')
                # print(rjob['placements'])
                if sys.maxsize == rjob['start_time']:
                    rjob['start_time'] = event_time
                if rjob['status'] == 'PENDING':
                    run_jobs.append(rjob)
            else:
                if rjob['status'] == 'RUNNING':
                    preempt_jobs.append(rjob)
                continue

        for job in preempt_jobs:
            job['status'] = 'PENDING'
            job['preempt'] = int(job['preempt'] + 1)
            # job['packing_used'] = 0
        # print("-----placement-------")
        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)
            # job['packing_used'] = 1
            # print(job['placements'])

        for queue in JOBS.queues:
            pending_job = list()
            for job in queue:
                if job['status'] == 'PENDING':
                    pending_job.append(job)
            for job in pending_job:
                queue.remove(job)
            queue.extend(pending_job)
        
        # get the next end_event
        del end_events[:]
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                end_time = float(event_time + rjob['remaining_iteration']*rjob['iteration_time_cur'])
                # print(event_time, rjob['job_idx'], rjob['remaining_time'], rjob['iteration_time'], rjob['iteration_time_cur'], end_time)
                tmp_dict = util.search_dict_list(end_events, 'time', end_time)
                if tmp_dict == None:
                    #not found, add the time into to job_events
                    tmp_dict = dict()
                    tmp_dict['time'] = end_time
                    tmp_dict['end_jobs'] = list()
                    tmp_dict['end_jobs'].append(rjob)
                    end_events.append(tmp_dict)
                else:
                    tmp_dict['end_jobs'].append(rjob)
        end_events.sort(key = lambda e:e.__getitem__('time'))

        next_job_jump = sys.maxsize
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING'==rjob['status']:
                qid = rjob['q_id']
                if qid < int(JOBS.num_queue - 1):
                    if gputime:
                        # print('jump_time: ', JOBS.queue_limit[qid]/rjob['num_gpu'], rjob['calc_executed_time'])
                        jump_time = math.ceil(JOBS.queue_limit[qid]/rjob['num_gpu']) - rjob['total_executed_time'] + event_time
                    else:
                        jump_time = JOBS.queue_limit[qid] - rjob['total_executed_time'] + event_time
                    if jump_time < next_job_jump:
                        next_job_jump = jump_time
        # print('next jump time: ', next_job_jump)

        LOG.checkpoint(event_time)     

def antman_find_collocate_job(job):
    '''
    return the collocate job with job
    job is an opportunistic job
    '''
    collocate_job_id = list()
    # print(job['job_idx'], job['priority'])
    for placement in job['placements']:
        for node_pl in placement['nodes']:
            node = CLUSTER.switch_list[placement['switch']].node_list[node_pl['id']]
            # print(node.gpu_job_list)
            # print(node_pl['gpu_list'])
            for gpu in node_pl['gpu_list']:
                for job_id in node.gpu_job_list[gpu][0]:
                    collocate_job_id.append(job_id)
                for job_id in node.gpu_job_list[gpu][1]:
                    if job_id != job['job_idx']:
                        collocate_job_id.append(job_id)
    collocate_job_id = list(set(collocate_job_id))
    collocate_job = list()
    # print('collocate_job_id', collocate_job_id)
    rjob_id = list()
    for rjob in JOBS.runnable_jobs:
        rjob_id.append(rjob['job_idx'])
    # print(rjob_id)
    for job_id in collocate_job_id:
        # print(job_id)
        collocate_job.append(JOBS.find_runnable_job(job_id))
    return collocate_job

def antman_calc_iter_time(job, collocate_job_list):
    '''
    calculate 1 iter time of job, whose priority is 1
    ignore synchronization among different gpu and return the maximum time of each gpu
    '''
    iter_time = job['iteration_time']
    for c_job in collocate_job_list:
        if c_job['priority']==0:
            if c_job['remaining_gpu']>0:
                continue
            # iters is not accurate
            time_list = copy.deepcopy(job['resource_time'])
            c_time_list = c_job['resource_time']
            iters = math.floor(time_list[0]/(c_time_list[1]+c_time_list[2]))
            data_extra = time_list[0] % (c_time_list[1]+c_time_list[2])
            if data_extra>0:
                iters += 1
                time_list[1] -= min(time_list[1], c_time_list[2], c_time_list[1]+c_time_list[2]-data_extra)
            iters += math.floor(time_list[1]/(c_time_list[0]+c_time_list[2]))
            gpu_extra = time_list[1] % (c_time_list[0]+c_time_list[2])
            if gpu_extra>0:
                iters += 1
                if gpu_extra<c_time_list[0]:
                    time_list[2] -= min(time_list[2], c_time_list[0]+c_time_list[1]-gpu_extra)
            iters += math.floor(time_list[2]/(c_time_list[0]+c_time_list[1]))
            net_extra = time_list[2]%(c_time_list[0]+c_time_list[1])
            if net_extra>c_time_list[0]:
                iters += 1
            iter_time = max(iter_time, iters*c_job['iteration_time'])
        else:
            iter_time = max(iter_time, job['iteration_time']*2.0)
        
    return iter_time


def antman_sim_jobs(gputime=False):
    '''
    simulate antman with the following assumptions:
    1. at most two jobs are packed
    2. the memory is enough
    3. resource guarantee job will not be changed to opportunistic job
    NOT FINISHED!!!
    '''
    end_events = list()
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']

        if end_time < start_time:
            event_time = end_time
            event = end_events[0]
        elif end_time > start_time:        
            event_time = start_time
            # print("start-time %d, end_time %d" % (start_time, end_time))
            event = JOBS.job_events.pop(0)
        elif math.isclose(end_time, start_time, abs_tol=1e-4) and not math.isclose(end_time, sys.maxsize, abs_tol=1e-4):
            event_time = start_time
            event = JOBS.job_events.pop(0)
            event['end_jobs'] = end_events[0]['end_jobs']

        assert math.isclose(event_time, event['time'], abs_tol=1e-4)
        # print(event_time, "event time!!!")

        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                tmp = event_time - e_job['last_check_time']
                e_job['total_executed_time'] = e_job['total_executed_time'] + tmp
                #job completes
                assert CLUSTER.release_job_res(e_job)==True
                # CLUSTER.release_gpus(e_job)
                LOG.job_complete(e_job, event_time)
                JOBS.runnable_jobs.remove(e_job)
                # print("job complete", e_job['job_idx'], e_job['priority'])
                # JOBS.print_placement(e_job)

        #for new-start jobs, add to runnable
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                #add into runnable list with pending status
                JOBS.move_to_runnable(s_job)

                s_job['remaining_time'] = s_job['duration']
                # s_job['remaining_gputime'] = float(s_job['remaining_time'] * s_job['num_gpu'])
                s_job['total_executed_time'] = 0.0
                s_job['calc_executed_time'] = 0.0
                s_job['executed_gputime'] = 0.0

                # s_job['total_executed_gputime'] = 0.0
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])

        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                if rjob['model_name'] in overhead_dict[rjob['num_gpu']]:
                    tmp_oh = overhead_dict[rjob['num_gpu']][rjob['model_name']]
                else:
                    tmp_oh = 10
                tmp = max(event_time - rjob['last_check_time']-tmp_oh, 0) 
                rjob['remaining_iteration'] -= tmp/rjob['iteration_time_cur']
                rjob['calc_executed_time'] = float(rjob['calc_executed_time'] + tmp/rjob['iteration_time_cur']*rjob['iteration_time'])
                rjob['total_executed_time'] = float(rjob['total_executed_time'] + event_time - rjob['last_check_time'])
                rjob['last_check_time'] = event_time
                rjob['remaining_time'] = rjob['remaining_iteration'] * rjob['iteration_time']
                rjob['executed_gputime'] = float(rjob['calc_executed_time'] * rjob['num_gpu'])
            elif 'PENDING' == rjob['status']:
                tmp = float(event_time - rjob['last_check_time'])
                rjob['pending_time'] = float(rjob['pending_time'] + tmp)
                rjob['last_check_time'] = event_time
            elif 'END' == rjob['status']: #almost impossible
                JOBS.runnable_jobs.remove(rjob)
                pass
        #sort jobs with priority as the first key and submit time as the second key
        if gputime:
            JOBS.runnable_jobs.sort(key = lambda e:(e.__getitem__('priority'), e.__getitem__('submit_time'), e.__getitem__('executed_gputime')))
        else:
            JOBS.runnable_jobs.sort(key = lambda e:(e.__getitem__('priority'), e.__getitem__('submit_time'), e.__getitem__('job_idx')))
        run_jobs = list()
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                continue
            assert rjob['status'] == 'PENDING'

            ret = try_get_job_res(rjob)
            if True == ret:
                if sys.maxsize == rjob['start_time']:
                    rjob['start_time'] = event_time
                if rjob['status'] == 'PENDING':
                    run_jobs.append(rjob)

        for job in run_jobs:
            job['status'] = 'RUNNING'
            job['resume'] = int(job['resume'] + 1)
            assert job['resume']==1
            # print("start job: ", job['job_idx'], job['priority'])
            # JOBS.print_placement(job)

        # get the next end_event
        del end_events[:]
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                if rjob['priority']==0 :
                    assert math.isclose(rjob['iteration_time'], rjob['iteration_time_cur'], abs_tol=1e-4)==True
                    end_time = float(event_time + rjob['remaining_iteration']*rjob['iteration_time_cur'])
                    tmp_dict = util.search_dict_list(end_events, 'time', end_time)
                    if tmp_dict == None:
                        #not found, add the time into to job_events
                        tmp_dict = dict()
                        tmp_dict['time'] = end_time
                        tmp_dict['end_jobs'] = list()
                        tmp_dict['end_jobs'].append(rjob)
                        end_events.append(tmp_dict)
                    else:
                        tmp_dict['end_jobs'].append(rjob)
                else:
                    collocate_job = antman_find_collocate_job(rjob)
                    rjob['iteration_time_cur'] = antman_calc_iter_time(rjob, collocate_job)
                    end_time = float(event_time + rjob['remaining_iteration']*rjob['iteration_time_cur'])
                    tmp_dict = util.search_dict_list(end_events, 'time', end_time)
                    if tmp_dict == None:
                        #not found, add the time into to job_events
                        tmp_dict = dict()
                        tmp_dict['time'] = end_time
                        tmp_dict['end_jobs'] = list()
                        tmp_dict['end_jobs'].append(rjob)
                        end_events.append(tmp_dict)
                    else:
                        tmp_dict['end_jobs'].append(rjob)

        end_events.sort(key = lambda e:e.__getitem__('time'))
        # print('end_events')
        # for event in end_events:
        #     print(event['time'], [job['job_idx'] for job in event['end_jobs']])
        for i in range(1, len(end_events)):
            if math.isclose(end_events[0]['time'], end_events[i]['time'], abs_tol=1e-4):
                end_events[0]['end_jobs'].extend(end_events[i]['end_jobs'])
                del end_events[i]

        # print("node info: ")
        # for switch in CLUSTER.switch_list:
        #     print("switch id: ",switch.id)
        #     for node in switch.node_list:
        #         print("node: ", node.id, node.gpu_job_list, node.gpu_util_list)


        LOG.checkpoint(event_time)

def get_scale_factors_array(jobs):
    scale_factors_array = np.zeros((len(jobs), ))
    for i, job in enumerate(jobs):
        scale_factors_array[i] = job['num_gpu']
    return scale_factors_array

def get_isolated_throughputs(jobs):
    allocation = np.array([math.ceil(CLUSTER.num_gpu / len(jobs)) for i in range((len(jobs)))])
    allocation = allocation / get_scale_factors_array(jobs)
    per_row_sum = np.maximum(allocation, np.ones(allocation.shape))
    allocation = allocation / per_row_sum
    isolated_throughputs = np.zeros((len(jobs), ), dtype=np.float64)
    for i, job in enumerate(jobs):
        isolated_throughputs[i] = job['tput'] * allocation[i]
    isolated_throughputs = isolated_throughputs.reshape((len(jobs), 1))
    return allocation

def get_base_constraints(x, scale_factors_array):
    return [
        x >= 0,
        x <= 1,
        cp.sum(cp.multiply(scale_factors_array, x), axis=0)<=CLUSTER.num_gpu
    ]

def themis_sim_jobs():
    '''
    themis finish-time fairness
    '''
    num_steps_remaining_prev_iteration, isolated_throughputs_prev_iteration = {}, {}
    cumulative_isolated_time = {} 
    end_events = list()
    last_event_time = 0
    job_interval = 360
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        if (len(JOBS.job_events) + len(end_events)) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break

        #decide which is the next event: start or end  ?
        start_time = sys.maxsize
        if len(JOBS.job_events) > 0:
            start_event = JOBS.job_events[0]
            start_time = start_event['time']
        end_time = sys.maxsize
        if len(end_events) > 0:
            end_event = end_events[0]
            end_time = end_event['time']

        if end_time < start_time:
            event_time = end_time
            event = end_events[0]
        elif end_time > start_time:        
            event_time = start_time
            # print("start-time %d, end_time %d" % (start_time, end_time))
            event = JOBS.job_events.pop(0)
        elif end_time == start_time and end_time != sys.maxsize:
            event_time = start_time
            event = JOBS.job_events.pop(0)
            event['end_jobs'] = end_events[0]['end_jobs']

        assert event_time == event['time']

        #for ending jobs, release gpu
        if 'end_jobs' in event:
            for e_job in event['end_jobs']:
                tmp = event_time - e_job['last_check_time']
                e_job['total_executed_time'] = e_job['total_executed_time'] + tmp
                #job completes
                CLUSTER.release_job_res(e_job)
                # CLUSTER.release_gpus(e_job)
                LOG.job_complete(e_job, event_time)
                JOBS.runnable_jobs.remove(e_job)


        #for new-start jobs, add to runnable
        if 'start_jobs' in event:
            for s_job in event['start_jobs']:
                #add into runnable list with pending status
                JOBS.move_to_runnable(s_job)
                s_job['deficit'] = 0
                s_job['time_should_received'] = 0
                s_job['remaining_time'] = s_job['remaining_iteration']*s_job['iteration_time']
                s_job['remaining_gputime'] = s_job['remaining_time'] * s_job['num_gpu']
                util.print_fn('---- job[%d] is added' % s_job['job_idx'])

        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                if rjob['model_name'] in overhead_dict[rjob['num_gpu']]:
                    tmp_oh = overhead_dict[rjob['num_gpu']][rjob['model_name']]
                else:
                    tmp_oh = 10
                # tmp_oh = 0
                tmp = max(event_time - rjob['last_check_time']-tmp_oh, 0)
                rjob['total_executed_time'] = rjob['total_executed_time'] + event_time - rjob['last_check_time']
                rjob['remaining_iteration'] -= tmp/rjob['iteration_time']
                rjob['last_check_time'] = event_time
                rjob['remaining_time'] = rjob['remaining_iteration']*rjob['iteration_time']
            elif 'PENDING' == rjob['status']:
                tmp = event_time - rjob['last_check_time']
                rjob['pending_time'] = rjob['pending_time'] + tmp
                rjob['last_check_time'] = event_time
            elif 'END' == rjob['status']: #almost impossible
                JOBS.runnable_jobs.remove(rjob)
                pass

        if len(JOBS.runnable_jobs)>0:
            # print(len(JOBS.runnable_jobs), 'cvxpy')
            scale_factors_array = get_scale_factors_array(JOBS.runnable_jobs)
            isolated_throughputs = get_isolated_throughputs(JOBS.runnable_jobs)
            x = cp.Variable(len(JOBS.runnable_jobs))
            expected_time_fractions = []
            for job_idx, r_job in enumerate(JOBS.runnable_jobs):
                if r_job['job_idx'] not in cumulative_isolated_time:
                    cumulative_isolated_time[r_job['job_idx']] = 0
                if r_job['job_idx'] in num_steps_remaining_prev_iteration:
                    cumulative_isolated_time[r_job['job_idx']] += (
                        num_steps_remaining_prev_iteration[r_job['job_idx']] -
                        r_job['remaining_iteration']) / \
                        isolated_throughputs_prev_iteration[r_job['job_idx']]
                throughput = r_job['tput']
                allocation_throughput = throughput * x[job_idx]
                expected_time_isolated = cumulative_isolated_time[r_job['job_idx']] + \
                (r_job['remaining_iteration'] / isolated_throughputs[job_idx])
                expected_time_allocation = event_time - r_job['submit_time'] + \
                    (r_job['remaining_iteration'] * cp.inv_pos(allocation_throughput))
                num_steps_remaining_prev_iteration[r_job['job_idx']] = r_job['remaining_iteration']
                expected_time_fraction = expected_time_allocation / expected_time_isolated
                # print("expected_time_allocation, expected_time_isolated", job_idx, r_job['job_idx'], expected_time_allocation, expected_time_isolated)
                expected_time_fractions.append(expected_time_fraction)
                isolated_throughputs_prev_iteration[r_job['job_idx']] = isolated_throughputs[job_idx]
            
            if len(expected_time_fractions) == 1:
                objective = cp.Minimize(expected_time_fractions[0])
            else:
                objective = cp.Minimize(cp.maximum(*expected_time_fractions))

            # Make sure that the allocation can fit in the cluster.
            constraints = get_base_constraints(x, scale_factors_array)
            cvxprob = cp.Problem(objective, constraints)
            # try:
            result = cvxprob.solve(solver='ECOS')
            # except SolverError:
                # result = cvxprob.solve(solver='SCS')

            if cvxprob.status != "optimal":
                print('WARNING: Allocation returned by policy not optimal!')

            for i, rjob in enumerate(JOBS.runnable_jobs):
                if rjob['total_executed_time']==0:
                    rjob['sort_val'] = x.value[i]*1e9
                else:
                    rjob['sort_val'] = x.value[i]/rjob['total_executed_time'] #rounds received
                rjob['allocation'] = x.value[i]
                rjob['deficit'] = rjob['time_should_received']-rjob['total_executed_time']
                rjob['time_should_received'] += x.value[i]*(event_time - last_event_time)

            JOBS.runnable_jobs.sort(key=lambda e:(e.__getitem__('sort_val'), e.__getitem__('deficit'), e.__getitem__('allocation')), reverse=True)

            chosen_jobs = list()
            CLUSTER.empty_infra()
            for rjob in JOBS.runnable_jobs:
                if 'RUNNING'==rjob['status']:
                    if 'placements' in rjob:
                        del rjob['placements'][:]
                ret = try_get_job_res(rjob, True)
                if True==ret:
                    chosen_jobs.append(rjob['job_idx'])
            
            JOBS.runnable_jobs.sort(key=lambda e: e.__getitem__('num_gpu'), reverse=True)
            
            run_jobs = list()
            preempt_jobs = list()
            #scan / execute jobs one by one
            CLUSTER.empty_infra()
            for rjob in JOBS.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    if 'placements' in rjob: 
                        del rjob['placements'][:]
                if rjob['job_idx'] in chosen_jobs:
                    ret = try_get_job_res(rjob)
                else:
                    ret = False
                if True == ret:
                    if sys.maxsize == rjob['start_time']:
                        rjob['start_time'] = event_time
                    if rjob['status'] == 'PENDING':
                        run_jobs.append(rjob)

                else:
                    if rjob['status'] == 'RUNNING':
                        preempt_jobs.append(rjob)

            for job in preempt_jobs:
                job['status'] = 'PENDING'
                job['preempt'] = int(job['preempt'] + 1)
            for job in run_jobs:
                job['status'] = 'RUNNING'
                job['resume'] = int(job['resume'] + 1)

        # get the next end_event
        del end_events[:]
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                end_time = event_time + rjob['remaining_iteration']*rjob['iteration_time']
                tmp_dict = util.search_dict_list(end_events, 'time', end_time)
                if tmp_dict == None:
                    #not found, add the time into to job_events
                    tmp_dict = dict()
                    tmp_dict['time'] = end_time
                    tmp_dict['end_jobs'] = list()
                    tmp_dict['end_jobs'].append(rjob)
                    end_events.append(tmp_dict)
                else:
                    tmp_dict['end_jobs'].append(rjob)
        end_events.sort(key = lambda e:e.__getitem__('time'))


        LOG.checkpoint(event_time)
        last_event_time = event_time


def sim_job_events():
    '''
    Simulate job start/end, and gpu allocation
    pick one event from sorted job_event list
    1.ending jobs
    2.check the pending job list, for potential job placements
    3.start jobs
    4.logging  
    '''
    while (len(JOBS.job_events) + len(JOBS.pending_jobs))> 0:
        if len(JOBS.job_events) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break
        event = JOBS.job_events[0]
        event_time = event['time']
        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        for e_job in event['end_jobs']:
            #remove from migratable jobs, if it's there
            # JOBS.remote_migratable(e_job)

            #job completes
            CLUSTER.release_job_res(e_job)
            # CLUSTER.release_gpus(e_job)
            LOG.job_complete(e_job, event_time)

        #for pending jobs, try to start
        for p_job in JOBS.pending_jobs:
            # ret = CLUSTER.alloc_gpus(p_job)
            ret = try_get_job_res(p_job)
            if ret == True:
                #if job is migratable, add into migratable job list
                # JOBS.add_migratable(p_job)
                JOBS.remove_from_pending(p_job, event_time)
                JOBS.add_job_end_event(p_job)
                util.print_fn('----job[%d] starts from pending' % p_job['job_idx'])
                # JOBS.read_job_info(p_job['job_idx'])
            else:
                # pending_jobs are sorted, if one is not able to be placement, then the rest are not necessary to consider
                break

        #for new-start jobs, try to start
        for s_job in event['start_jobs']:
            ret = try_get_job_res(s_job)
            # ret = CLUSTER.alloc_gpus(s_job)
            if ret == False:
                #allocation failed, add into pending jobs
                JOBS.move_to_pending(s_job)
                util.print_fn('----job[%d] move to pending' % s_job['job_idx'])
            else:
                #if job is migratable, add into migratable job list
                # JOBS.add_migratable(s_job)
                JOBS.add_job_end_event(s_job)
                util.print_fn('----job[%d] starts' % s_job['job_idx'])
                # JOBS.read_job_info(s_job['job_idx'])

        #sort pending jobs based on the num_gpu
        JOBS.pending_jobs.sort(key = lambda e:e.__getitem__('num_gpu'))

        #remove time_event
        JOBS.job_events.pop(0)
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        # JOBS.print_job_events()

        LOG.checkpoint(event_time)

    pass

def sim_gpu_demands():
    '''
    Simulate job start/end, and gpu demands
    pick one event from sorted job_event list
    1.ending jobs
    2.check the pending job list, for potential job placements
    3.start jobs
    4.logging  
    '''
    while (len(JOBS.job_events) + len(JOBS.pending_jobs))> 0:
        if len(JOBS.job_events) == 0:
            util.print_fn("This cluster is not large enough to run the job")
            break
        event = JOBS.job_events[0]
        event_time = event['time']
        # util.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
        #for ending jobs, release gpu
        for e_job in event['end_jobs']:
            #remove from migratable jobs, if it's there
            # JOBS.remote_migratable(e_job)

            # CLUSTER.release_job_res(e_job)
            # LOG.job_complete(e_job, event_time)
            JOBS.delete_gpu_job(e_job)

        #for new-start jobs, try to start
        for s_job in event['start_jobs']:
            #if job is migratable, add into migratable job list
            # JOBS.add_migratable(s_job)
            s_job['end_time'] = s_job['submit_time'] + s_job['duration']
            JOBS.add_job_end_event(s_job)
            util.print_fn('----job[%d] starts' % s_job['job_idx'])
            # JOBS.read_job_info(s_job['job_idx'])
            JOBS.add_gpu_job(s_job)



        #sort pending jobs based on the num_gpu
        # JOBS.pending_jobs.sort(key = lambda e:e.__getitem__('num_gpu'))

        #remove time_event
        JOBS.job_events.pop(0)
        JOBS.job_events.sort(key = lambda e:e.__getitem__('time'))
        # JOBS.print_job_events()

        # LOG.checkpoint(event_time)
        LOG.checkpoint_gpu_demands(event_time)

def cal_r_gittins_index(job_data, a):
    '''
    a means attained-service to that job
    gittins_index = P/E
    r_gi = E/P
    '''
    ut_delta = JOBS.gittins_delta

    data = job_data['data']
    if a > (job_data['data'][-1] - 1):
        return 0.0
    else:
        idx = next(x[0] for x in enumerate(data) if x[1] > a)

    next_a = a + ut_delta
    if next_a > (job_data['data'][-1] - 1):
        idx_delta = job_data['num'] - 1
    else:
        idx_delta = next(x[0] for x in enumerate(data) if x[1] > next_a)
    # print(idx, idx_delta)

    p = round(((idx_delta - idx) * 1.0) / (job_data['num'] - idx), 5)

    e_sum = sum(data[idx : idx_delta]) + (ut_delta * (job_data['num'] - idx_delta))
    e = round(e_sum / (job_data['num'] - idx), 5)

    # rank of gittins index = 1/gi
    # r_gi = round(e / p, 4)
    r_gi = round(p * 1000000 / e, 4)

    # print(idx, idx_delta, p, e_sum, e, r_gi)
    return r_gi


def parse_job_dist():
    job_dist_file = os.path.join(os.getcwd(), 'yarn-gput1000.csv')
    fd = open(job_dist_file, 'r')
    reader = csv.DictReader(fd, delimiter = ',') 
    durations = list()
    for row in reader:
        durations.append(int(row['duration']))
    fd.close()
    total_len = len(durations)
    durations.sort()
    print("  %s samples are learned" % total_len)

    job_dict = dict()
    job_dict['num'] = total_len
    job_dict['data'] = durations

    gi = list()
    for v in job_dict['data']:
        gi.append(cal_r_gittins_index(job_dict, int(v-1)))

    # print(gi)
    job_dict['data'].append(sys.maxsize)
    gi.append(0.0)
    job_dict['gittins'] = gi

    return job_dict


def main():

    if FLAGS.schedule == 'multi-dlas-gpu': 
        if FLAGS.scheme != 'count':
            util.print_fn("In Main, multi-dlas-gpu without count")
            exit()
    ''' Parse input'''
    parse_job_file(FLAGS.trace_file)
    parse_cluster_spec()

    ''' prepare logging '''
    LOG.init_log()

    # lp.placement(JOBS.job_list[0])
    ''' Prepare jobs'''
    JOBS.prepare_job_start_events()

    # used gpu number of jobs
    used_gpu = [8,4,2,1]

    # sim_job_events()
    if FLAGS.schedule == 'shortest':
        shortest_first_sim_jobs()
    elif FLAGS.schedule == 'shortest-gpu':
        shortest_first_sim_jobs(True)
    elif FLAGS.schedule == 'dlas-gpu':
        dlas_sim_jobs(True)
    elif FLAGS.schedule == 'multi-resource-blossom-same-gpu':
        multi_resource_blossom_same_sim_jobs(True)
    elif FLAGS.schedule == 'multi-resource-blossom-same-gpu-unaware':
        multi_resource_blossom_same_sim_jobs(True, know_duration=False)
    elif FLAGS.schedule == 'multi-resource-blossom-same-gpu-unaware-worstordering':
        multi_resource_blossom_same_sim_jobs(True, know_duration=False, ordering=2)
    elif FLAGS.schedule == 'multi-resource-gpu-unaware':
        multi_resource_blossom_same_sim_jobs(True, know_duration=False, blossom=False)
    elif FLAGS.schedule == 'antman':
        antman_sim_jobs()
    elif FLAGS.schedule == 'themis':
        themis_sim_jobs()
    else:
        print('Scheduler not supported!')

if __name__ == '__main__':
    # print('Hello world %d' % 2)
    main()
