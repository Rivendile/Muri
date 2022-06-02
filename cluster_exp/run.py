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
import cvxpy as cp
import numpy as np

import utils
import flags
import jobs
import cluster
import log
import lp
from matching import Blossom_Same, _Packing
from scheduler import Scheduler
# import hosts
# import placement_scheme as scheme
# import cmd

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
flags.DEFINE_float('weight_lbd', 0.0, '''The factor of the lower bound of expected weight (i jobs packing of n resources: i/n)''')

flags.DEFINE_boolean('print', False, 
                '''Enable print out information, default is False''')
flags.DEFINE_boolean('flush_stdout', True, 
                '''Flush stdout, default is True''')
flags.DEFINE_integer('scheduler_port', 9011, '''The port of scheduler''')
flags.DEFINE_integer('controller_port', 9012, '''The port of controler''')
flags.DEFINE_integer('schedule_interval', 10, '''The scheduling interval of scheduler, default is 10s''')
flags.DEFINE_integer('fast_forwarding', 0, 
                '''Running iters when fast-forwarding for cluster experiment, default is 0, i.e., no fast-forwarding''')
flags.DEFINE_integer('packing_num', 4, 
                '''maximum number of jobs in one packing''')
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
    utils.print_fn('--------------------------------- Read TF jobs from: %s ---------------------------------' % trace_file) 
    utils.print_fn('    we get the following fields:\n        %s' % keys)
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
    utils.print_fn('---------------------------------- Get %d TF jobs in total ----------------------------------' % job_idx)
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
        utils.print_fn(keys)
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
            # utils.print_fn('num_switch %s' % row['num_switch'])
            FLAGS.num_switch = int(row['num_switch'])
            FLAGS.num_node_p_switch = int(row['num_node_p_switch'])
            FLAGS.num_gpu_p_node = int(row['num_gpu_p_node'])
            FLAGS.num_cpu_p_node = int(row['num_cpu_p_node'])
            FLAGS.mem_p_node = int(row['mem_p_node'])
        fd.close()

    utils.print_fn("num_switch: %d" % FLAGS.num_switch)
    utils.print_fn("num_node_p_switch: %d" % FLAGS.num_node_p_switch)
    utils.print_fn("num_gpu_p_node: %d" % FLAGS.num_gpu_p_node)
    utils.print_fn("num_cpu_p_node: %d" % FLAGS.num_cpu_p_node)
    utils.print_fn("mem_p_node: %d" % FLAGS.mem_p_node)

    '''init infra'''
    CLUSTER.init_infra()
    # utils.print_fn(lp.prepare_cluster_info())
    utils.print_fn('--------------------------------- End of cluster spec ---------------------------------')
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
    elif FLAGS.scheme == 'balance':
        ret = lp.placement(job)
        # ret = lp.min_new_job(job)
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

def shortest_first_sim_jobs(scheduler=None, gputime=False, place=False):
    '''
    new jobs are added to the end of the ending queue
    but in the queue, shortest (gpu) job first be served, until no resource
    place=True: place jobs in the descending order of #GPU
    '''
    # end_events = list()
    scheduler._controller.set_start_time()
    last_check_time = 0
    finished_job_cnt = 0
    if FLAGS.fast_forwarding == 0:
        while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            done_flag = False
            new_flag = False
            # print('before: ', scheduler.get_time())
            while not scheduler._controller.done_queue.empty():
                finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
                e_job = JOBS.find_runnable_job(job_id)
                if returncode==0:
                    tmp = finished_time-e_job['last_check_time']
                    e_job['total_executed_time'] += tmp
                    e_job['last_check_time'] = finished_time
                    CLUSTER.release_job_res(e_job)
                    scheduler._trainers.pop(e_job['job_idx'])
                    LOG.job_complete(e_job, finished_time)
                    finished_job_cnt += 1
                    scheduler._logger.info(f'scheduler finishes {finished_job_cnt} jobs in all!')
                    JOBS.runnable_jobs.remove(e_job)
                else:
                    e_job['status'] = 'PENDING'
                    scheduler._trainers.pop(e_job['job_idx'])
                    del e_job['placements'][:]
                done_flag = True
                print(scheduler.get_time(), 'check: done ', e_job['job_idx'], finished_time)
            while scheduler.has_ready_jobs(scheduler.get_time()):
                event = JOBS.job_events.pop(0)
                assert 'start_jobs' in event
                for s_job in event['start_jobs']:
                    JOBS.move_to_runnable(s_job)
                    s_job['remaining_time'] = s_job['iteration_time'] * s_job['remaining_iterations']
                    s_job['remaining_gputime'] = s_job['remaining_time'] * s_job['num_gpu']
                    utils.print_fn('---- job[%d] is added' % s_job['job_idx'])
                new_flag = True
            
            tmp_time = scheduler.get_time()
            if done_flag or new_flag:
                assert tmp_time - last_check_time > FLAGS.schedule_interval or tmp_time < FLAGS.schedule_interval
                for rjob in JOBS.runnable_jobs:
                    if 'RUNNING' == rjob['status']:
                        tmp = tmp_time - rjob['last_check_time']
                        rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
                        rjob['last_check_time'] = tmp_time
                        finished_iter = scheduler.query_stats([rjob['job_idx']])
                        rjob['remaining_iterations'] -= finished_iter
                        # print(rjob['job_idx'], rjob['remaining_iterations'], finished_iter)
                        rjob['remaining_time'] = rjob['iteration_time'] * rjob['remaining_iterations']
                        if gputime:
                            rjob['remaining_gputime'] = rjob['remaining_time'] * rjob['num_gpu']
                        print(tmp_time, 'check: running ', rjob['job_idx'], rjob['remaining_iterations'], rjob['total_executed_time'])
                    elif 'PENDING' == rjob['status']:
                        tmp = tmp_time - rjob['last_check_time']
                        rjob['pending_time'] += tmp
                        rjob['last_check_time'] = tmp_time
                        print(tmp_time, 'check: pending ', rjob['job_idx'], rjob['remaining_iterations'], rjob['pending_time'])
                    elif 'END' == rjob['status']: #almost impossible
                        JOBS.runnable_jobs.remove(rjob)
                        print(tmp_time, 'check: ending ', rjob['job_idx'], rjob['remaining_iterations'])
                        pass
                #sort jobs with shortest first
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
                        if rjob['job_idx'] in scheduler._trainers:
                            scheduler._trainers.pop(rjob['job_idx'])
                        jobinfo = JOBS.to_jobinfo(rjob)
                        scheduler._controller.kill(jobinfo)
                        assert 'placements' in rjob
                        del rjob['placements'][:]

                    ret = try_get_job_res(rjob) 
                    if True == ret:
                        rjob['job_counter'] += 1
                        if sys.maxsize == rjob['start_time']:
                            rjob['start_time'] = tmp_time
                        if rjob['status'] == 'PENDING':
                            run_jobs.append(rjob)
                        jobinfo = JOBS.to_jobinfo(rjob)
                        scheduler._controller.execute(jobinfo)
                    else:
                        # rjob['status'] = 'PENDING'
                        if rjob['status'] == 'RUNNING':
                            preempt_jobs.append(rjob)
                        continue

                for job in preempt_jobs:
                    job['status'] = 'PENDING'
                    job['preempt'] = int(job['preempt'] + 1)
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, preempt')
                for job in run_jobs:
                    job['status'] = 'RUNNING'
                    job['resume'] = int(job['resume'] + 1)
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, run')

                last_check_time = tmp_time
        
            # print('at end: ', scheduler.get_time())
            time00 = time.time()
            time.sleep(10)
            LOG.checkpoint(tmp_time, scheduler, done_flag or new_flag)
            time01 = time.time()
            time.sleep(FLAGS.schedule_interval-(time01-time00))
    else:
        est_check_time = 0
        running_jobs = 0
        while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            while scheduler.get_time()-last_check_time<FLAGS.schedule_interval and scheduler.has_running_trainers(running_jobs):
                # print("waiting for trainers: ", scheduler.get_time(), running_jobs, scheduler.has_running_trainers(running_jobs), scheduler._trainers.keys())
                time.sleep(5)
            assert len(scheduler._trainers.keys())==running_jobs
            if running_jobs>0:
                time.sleep(2)
            # assert scheduler.has_running_trainers(running_jobs)==False
            tmp_jump_time = max(est_check_time - scheduler.get_time(), 0)
            scheduler._controller._jump_time += tmp_jump_time
            LOG.checkpoint_utils(last_check_time, scheduler)

            # print('deal with done jobs')
            # deal with done jobs
            done_flag = False
            finished_jobs = []
            while not scheduler._controller.done_queue.empty():
                finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
                e_job = JOBS.find_runnable_job(job_id)
                finished_jobs.append(e_job['job_idx'])
                # est_finished_time = finished_time + max(e_job['remaining_iterations']-FLAGS.fast_forwarding, 0)*e_job['real_itertime'][0]
                if returncode==0:
                    e_job['last_finish_time'] = finished_time
                else:
                    e_job['status'] = 'PENDING'
                    del e_job['placements'][:]
                    done_flag = True
                    scheduler._controller._logger.info(f'controller, {rjob["job_idx"]}, error {returncode}')
                scheduler._trainers.pop(e_job['job_idx'])

            # print('deal with new jobs:')
            # deal with new jobs
            new_flag = False
            while scheduler.has_ready_jobs(scheduler.get_time()):
                event = JOBS.job_events.pop(0)
                assert 'start_jobs' in event
                for s_job in event['start_jobs']:
                    JOBS.move_to_runnable(s_job)
                    s_job['remaining_time'] = s_job['iteration_time'] * s_job['remaining_iterations']
                    s_job['remaining_gputime'] = s_job['remaining_time'] * s_job['num_gpu']
                    utils.print_fn('---- job[%d] is added' % s_job['job_idx'])
                    # print(s_job['job_idx'], s_job['last_check_time'], scheduler.get_time())
                new_flag = True
                        
            for rjob in JOBS.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    if running_jobs>0 and rjob['job_idx'] not in finished_jobs:
                        jobinfo = JOBS.to_jobinfo(rjob)
                        scheduler._controller.kill(jobinfo)
                        scheduler._trainers.pop(rjob['job_idx'])
                        done_flag=True
                        scheduler._controller._logger.info(f'controller, {rjob["job_idx"]}, timeout')
                        finished_jobs.append(rjob['job_idx'])
                        if 'placements' in rjob:
                            del rjob['placements'][:]
                        rjob['status'] = 'PENDING'
            
            tmp_time = scheduler.get_time()
            print(tmp_time, '-------------')
            assert tmp_time - last_check_time > FLAGS.schedule_interval or tmp_time < FLAGS.schedule_interval
            done_job_list = list()
            for rjob in JOBS.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    tmp = tmp_time - rjob['last_check_time']
                    # finished_iter = scheduler.query_stats(rjob['job_idx'])
                    # print('request last_finish_time: ', rjob['job_idx'])
                    tmp_error = (rjob['real_itertime'][0]-rjob['iteration_time'])/rjob['iteration_time']
                    JOBS.overhead_list[1].append(tmp_error)
                    if rjob['last_finish_time']<last_check_time:
                        finished_time = rjob['remaining_iterations'] * rjob['real_itertime'][0]
                    else:
                        finished_time = rjob['last_finish_time']-rjob['last_check_time'] + max(rjob['remaining_iterations']-FLAGS.fast_forwarding, 0)*rjob['real_itertime'][0]
                    if finished_time<=tmp:
                        rjob['total_executed_time'] += finished_time
                        rjob['last_check_time'] += finished_time
                        rjob['remaining_iterations'] = 0
                        rjob['remaining_time'] = 0
                        CLUSTER.release_job_res(rjob)
                        LOG.job_complete(rjob, rjob['last_check_time'])
                        finished_job_cnt += 1
                        scheduler._logger.info(f'scheduler finishes {finished_job_cnt} jobs in all!')
                        done_job_list.append(rjob)
                        done_flag=True
                    else:
                        rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
                        if rjob['last_finish_time']<last_check_time:
                            finished_iter = int((tmp_time - rjob['last_check_time'])/rjob['real_itertime'][0])
                        else:
                            finished_iter = int(FLAGS.fast_forwarding + (tmp_time-rjob['last_finish_time'])/rjob['real_itertime'][0])
                        rjob['remaining_iterations'] -= finished_iter
                        # print(rjob['job_idx'], rjob['remaining_iterations'], finished_iter)
                        rjob['remaining_time'] = rjob['iteration_time'] * rjob['remaining_iterations']
                        if gputime:
                            rjob['remaining_gputime'] = rjob['remaining_time'] * rjob['num_gpu']
                        rjob['last_check_time'] = tmp_time
                    # print(tmp_time, 'check: running ', rjob['job_idx'], finished_time, rjob['last_finish_time'], rjob['remaining_iterations'], rjob['last_check_time'], rjob['real_itertime'])
                elif 'PENDING' == rjob['status']:
                    tmp = tmp_time - rjob['last_check_time']
                    rjob['pending_time'] += tmp
                    rjob['last_check_time'] = tmp_time
                    # print(tmp_time, 'check: pending ', rjob['job_idx'], rjob['remaining_iterations'], rjob['last_check_time'])
                elif 'END' == rjob['status']: #almost impossible
                    done_job_list.append(rjob)
                    # print(tmp_time, 'check: ending', rjob['job_idx'], rjob['remaining_iterations'], rjob['last_check_time'])
                    pass
            tmp_list = JOBS.overhead_list[1]
            tmp_n = len(tmp_list)
            if tmp_n>0:
                tmp_mean = sum(tmp_list)/tmp_n
                tmp_ss = sum([(x-tmp_mean)**2 for x in tmp_list])
                tmp_std = (tmp_ss/tmp_n)**0.5
                scheduler._logger.info(f'scheduler, error rate between sim and real: mean {tmp_mean}, std {tmp_std}')
            for djob in done_job_list:
                JOBS.runnable_jobs.remove(djob)
            running_jobs = 0
            if done_flag or new_flag:
                scheduler.clear_src_utils()
                #sort jobs with shortest first
                if gputime:
                    JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_gputime'))
                else:
                    JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_time'))

                # print('---------', tmp_time)
                # for rjob in JOBS.runnable_jobs:
                #     print(rjob['job_idx'], rjob['remaining_time'], rjob['num_gpu'])
                run_jobs = list()
                preempt_jobs = list()
                tmp_runnable_jobs = dict()
                #scan / execute jobs one by one
                # if place==True:
                #     CLUSTER.empty_infra()
                #     for rjob in JOBS.runnable_jobs:
                #         ret = try_get_job_res(rjob, True) 
                #         if ret == True:
                #             tmp_runnable_jobs[rjob['job_idx']]=True
                #     JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('num_gpu'), reverse=True)
                # CLUSTER.empty_infra()
                # for rjob in JOBS.runnable_jobs:
                if place ==True:
                    CLUSTER.empty_infra()
                    tmp_runnable_jobs = list()
                    for rjob in JOBS.runnable_jobs:
                        ret = try_get_job_res(rjob, True) 
                        if ret == True:
                            tmp_runnable_jobs.append(rjob)
                        else:
                            if rjob['status'] == 'RUNNING':
                                assert 'placements' in rjob
                                del rjob['placements'][:]
                                preempt_jobs.append(rjob)
                    tmp_runnable_jobs.sort(key = lambda e:e.__getitem__('num_gpu'), reverse=True)
                else:
                    tmp_runnable_jobs = JOBS.runnable_jobs

                CLUSTER.empty_infra()
                for rjob in tmp_runnable_jobs:
                    if 'RUNNING' == rjob['status']:
                        assert 'placements' in rjob
                        del rjob['placements'][:]

                    # if place==True and rjob['job_idx'] in tmp_runnable_jobs :
                    #     ret = try_get_job_res(rjob) 
                    # else:
                    #     ret = False
                    ret = try_get_job_res(rjob) 
                    assert place==False or ret==True
                    # print('scheduling: ', rjob['job_idx'], rjob['num_gpu'], ret)
                    if True == ret:
                        # print('running: ', rjob['job_idx'], tmp_time, scheduler.get_time())
                        rjob['job_counter'] += 1
                        running_jobs += 1
                        if sys.maxsize == rjob['start_time']:
                            rjob['start_time'] = tmp_time
                        if rjob['status'] == 'PENDING':
                            run_jobs.append(rjob)
                        jobinfo = JOBS.to_jobinfo(rjob)
                        scheduler._controller.execute(jobinfo)
                    else:
                        # rjob['status'] = 'PENDING'
                        if rjob['status'] == 'RUNNING':
                            preempt_jobs.append(rjob)
                        continue

                for job in preempt_jobs:
                    job['status'] = 'PENDING'
                    job['preempt'] = int(job['preempt'] + 1)
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, preempt')
                for job in run_jobs:
                    job['status'] = 'RUNNING'
                    job['resume'] = int(job['resume'] + 1)
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, run')

                time.sleep(20)
            last_check_time = tmp_time
            est_check_time = last_check_time + FLAGS.schedule_interval
            # LOG.checkpoint(tmp_time, scheduler, done_flag or new_flag, secs)


def dlas_sim_jobs(scheduler, gputime=False, solve_starvation=0, place=False):
    '''
    Job's executed time -- priority queue
    Q0:[0, 30min)
    Q1:[30min,1h)
    Q2:[1h, 2h)
    Q3:[2h, 00)

    in each queue, jobs are scheduled in fit-first with FIFO
    how to avoid starvation?

    TODO:  2. add move_back for avoiding starvation;
    '''
    next_job_jump = sys.maxsize
    scheduler._controller.set_start_time()
    last_check_time = 0
    finished_job_cnt = 0
    if FLAGS.fast_forwarding == 0:
        print('Not Implemented!')
    else:
        est_check_time = 0
        running_jobs = 0
        print('dlas info: ', JOBS.num_queue, JOBS.queue_limit)
        while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            while scheduler.get_time()-last_check_time<FLAGS.schedule_interval and scheduler.has_running_trainers(running_jobs):
                # print("waiting for trainers: ", scheduler.get_time(), running_jobs, scheduler.has_running_trainers(running_jobs), scheduler._trainers.keys())
                time.sleep(5)
            assert len(scheduler._trainers.keys())==running_jobs
            if running_jobs>0:
                time.sleep(2)
            # assert scheduler.has_running_trainers(running_jobs)==False
            tmp_jump_time = max(est_check_time - scheduler.get_time(), 0)
            scheduler._controller._jump_time += tmp_jump_time
            LOG.checkpoint_utils(last_check_time, scheduler)

            # print('deal with done jobs')
            # deal with done jobs
            done_flag = False
            finished_jobs = []
            while not scheduler._controller.done_queue.empty():
                finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
                e_job = JOBS.find_runnable_job(job_id)
                finished_jobs.append(e_job['job_idx'])
                # est_finished_time = finished_time + max(e_job['remaining_iterations']-FLAGS.fast_forwarding, 0)*e_job['real_itertime'][0]
                if returncode==0:
                    e_job['last_finish_time'] = finished_time
                else:
                    e_job['status'] = 'PENDING'
                    del e_job['placements'][:]
                    done_flag = True
                scheduler._trainers.pop(e_job['job_idx'])
                # print(scheduler.get_time(), 'last_finish_time', e_job['job_idx'], finished_time, e_job['real_itertime'])

            # print('deal with new jobs:')
            # deal with new jobs
            new_flag = False
            while scheduler.has_ready_jobs(scheduler.get_time()):
                event = JOBS.job_events.pop(0)
                assert 'start_jobs' in event
                for s_job in event['start_jobs']:
                    JOBS.move_to_runnable(s_job)
                    s_job['q_id'] = 0
                    JOBS.queues[0].append(s_job)
                    s_job['remaining_time'] = s_job['iteration_time'] * s_job['remaining_iterations']
                    s_job['remaining_gputime'] = s_job['remaining_time'] * s_job['num_gpu']
                    utils.print_fn('---- job[%d] is added' % s_job['job_idx'])
                    # print(s_job['job_idx'], s_job['last_check_time'], scheduler.get_time())
                new_flag = True
            
            for rjob in JOBS.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    if running_jobs>0 and rjob['job_idx'] not in finished_jobs:
                        jobinfo = JOBS.to_jobinfo(rjob)
                        scheduler._controller.kill(jobinfo)
                        scheduler._trainers.pop(rjob['job_idx'])
                        done_flag=True
                        scheduler._controller._logger.info(f'controller, {rjob["job_idx"]}, timeout')
                        finished_jobs.append(rjob['job_idx'])
                        if 'placements' in rjob:
                            del rjob['placements'][:]
                        rjob['status'] = 'PENDING'
            
            demote_flag = False # have demote/promote or not
            tmp_time = scheduler.get_time()
            assert tmp_time - last_check_time > FLAGS.schedule_interval or tmp_time < FLAGS.schedule_interval
            done_job_list = list()
            for rjob in JOBS.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    tmp = tmp_time - rjob['last_check_time']
                    # finished_iter = scheduler.query_stats(rjob['job_idx'])
                    # print('request last_finish_time: ', rjob['job_idx'])
                    tmp_error = (rjob['real_itertime'][0]-rjob['iteration_time'])/rjob['iteration_time']
                    JOBS.overhead_list[1].append(tmp_error)
                    if rjob['last_finish_time']<last_check_time:
                        finished_time = rjob['remaining_iterations'] * rjob['real_itertime'][0]
                    else:
                        finished_time = rjob['last_finish_time']-rjob['last_check_time'] + max(rjob['remaining_iterations']-FLAGS.fast_forwarding, 0)*rjob['real_itertime'][0]
                    if finished_time<=tmp:
                        rjob['total_executed_time'] += finished_time
                        rjob['last_check_time'] += finished_time
                        rjob['remaining_iterations'] = 0
                        rjob['remaining_time'] = 0
                        CLUSTER.release_job_res(rjob)
                        LOG.job_complete(rjob, rjob['last_check_time'])
                        finished_job_cnt += 1
                        scheduler._logger.info(f'scheduler finishes {finished_job_cnt} jobs in all!')
                        done_job_list.append(rjob)
                        done_flag=True
                    else:
                        rjob['total_executed_time'] += tmp
                        rjob['executed_time'] += tmp
                        if rjob['last_finish_time']<last_check_time:
                            finished_iter = int((tmp_time - last_check_time)/rjob['real_itertime'][0])
                        else:
                            finished_iter = int(FLAGS.fast_forwarding + (tmp_time-rjob['last_finish_time'])/rjob['real_itertime'][0])
                        rjob['remaining_iterations'] -= finished_iter
                        # print(rjob['job_idx'], rjob['remaining_iterations'], finished_iter)
                        # rjob['remaining_time'] = rjob['iteration_time'] * rjob['remaining_iterations']
                        j_gt = 0
                        if gputime:
                            j_gt = rjob['executed_time'] * rjob['num_gpu']
                        else:
                            j_gt = rjob['executed_time']
                        cur_qid = rjob['q_id']
                        while cur_qid<int(JOBS.num_queue - 1) and j_gt >=JOBS.queue_limit[cur_qid]:
                            rjob['q_id'] = cur_qid+1
                            JOBS.queues[rjob['q_id']].append(rjob)
                            JOBS.queues[cur_qid].remove(rjob)
                            # print(f'job {rjob["job_idx"]} demote to Q{rjob["q_id"]}')
                            demote_flag = True
                            cur_qid = rjob['q_id']
                        rjob['last_check_time'] = tmp_time
                    # print(tmp_time, 'check: running ', rjob['job_idx'], finished_time, rjob['last_finish_time'], rjob['remaining_iterations'], rjob['last_check_time'], rjob['real_itertime'])
                elif 'PENDING' == rjob['status']:
                    tmp = tmp_time - rjob['last_check_time']
                    rjob['pending_time'] += tmp
                    rjob['last_check_time'] = tmp_time
                    if rjob['executed_time'] >0:
                        rjob['last_pending_time'] += tmp
                    if solve_starvation>0 and rjob['q_id']>0 and rjob['total_executed_time']>0 and rjob['executed_time']>0:
                        if rjob['last_pending_time']>= rjob['executed_time'] * solve_starvation:
                            rjob['executed_time'] = 0
                            rjob['last_pending_time'] = 0
                            JOBS.queues[0].append(rjob)
                            JOBS.queues[rjob['q_id']].remove(rjob)
                            rjob['q_id'] = 0
                            rjob['promote'] = int(rjob['promote'] + 1)
                            demote_flag = True
                    # print(tmp_time, 'check: pending ', rjob['job_idx'], rjob['remaining_iterations'], rjob['last_check_time'])
                elif 'END' == rjob['status']: #almost impossible
                    done_job_list.append(rjob)
                    # print(tmp_time, 'check: ending', rjob['job_idx'], rjob['remaining_iterations'], rjob['last_check_time'])
                    pass
            tmp_list = JOBS.overhead_list[1]
            tmp_n = len(tmp_list)
            if tmp_n>0:
                tmp_mean = sum(tmp_list)/tmp_n
                tmp_ss = sum([(x-tmp_mean)**2 for x in tmp_list])
                tmp_std = (tmp_ss/tmp_n)**0.5
                scheduler._logger.info(f'scheduler, error rate between sim and real: mean {tmp_mean}, std {tmp_std}')
            for djob in done_job_list:
                JOBS.runnable_jobs.remove(djob)
                JOBS.queues[djob['q_id']].remove(djob)
            running_jobs = 0
            if done_flag or new_flag or demote_flag:
                scheduler.clear_src_utils()
                # print('---------', tmp_time)
                # print('number of runnable jobs:', len(JOBS.runnable_jobs), sum([len(queue) for queue in JOBS.queues]))
                # for queue_id, queue in enumerate(JOBS.queues):
                #     print(queue_id, ":", [rjob['job_idx'] for rjob in queue])
                assert len(JOBS.runnable_jobs) == sum([len(queue) for queue in JOBS.queues])
                run_jobs = list()
                preempt_jobs = list()
                #scan / execute jobs one by one
                if place ==True:
                    CLUSTER.empty_infra()
                    tmp_runnable_jobs = list()
                    for queue in JOBS.queues:
                        for rjob in queue:
                            ret = try_get_job_res(rjob, True) 
                            if ret == True:
                                tmp_runnable_jobs.append(rjob)
                            else:
                                if rjob['status'] == 'RUNNING':
                                    assert 'placements' in rjob
                                    del rjob['placements'][:]
                                    preempt_jobs.append(rjob)
                    tmp_runnable_jobs.sort(key = lambda e:e.__getitem__('num_gpu'), reverse=True)
                else:
                    tmp_runnable_jobs = list()
                    for queue in JOBS.queues:
                        for rjob in queue:
                            tmp_runnable_jobs.append(rjob)

                CLUSTER.empty_infra()
                for rjob in tmp_runnable_jobs:
                    if 'RUNNING' == rjob['status']:
                        assert 'placements' in rjob
                        del rjob['placements'][:]

                    ret = try_get_job_res(rjob) 
                    assert place==False or ret==True
                    # print('scheduling: ', rjob['job_idx'], rjob['num_gpu'], ret)
                    if True == ret:
                        # print('running: ', rjob['job_idx'], tmp_time, scheduler.get_time())
                        rjob['job_counter'] += 1
                        running_jobs += 1
                        if sys.maxsize == rjob['start_time']:
                            rjob['start_time'] = tmp_time
                        if rjob['status'] == 'PENDING':
                            run_jobs.append(rjob)
                        jobinfo = JOBS.to_jobinfo(rjob)
                        scheduler._controller.execute(jobinfo)
                    else:
                        if rjob['status'] == 'RUNNING':
                            preempt_jobs.append(rjob)
                        continue

                for job in preempt_jobs:
                    job['status'] = 'PENDING'
                    job['preempt'] = int(job['preempt'] + 1)
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, preempt')
                for job in run_jobs:
                    job['status'] = 'RUNNING'
                    job['resume'] = int(job['resume'] + 1)
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, run')

                for queue in JOBS.queues:
                    pending_job = list()
                    for job in queue:
                        if job['status'] == 'PENDING':
                            pending_job.append(job)
                    for job in pending_job:
                        queue.remove(job)
                    queue.extend(pending_job)

                time.sleep(20)
            last_check_time = tmp_time
            est_check_time = last_check_time + FLAGS.schedule_interval
            # LOG.checkpoint(tmp_time, scheduler, done_flag or new_flag or demote_flag, secs)

def multi_resource_blossom_same_sim_jobs(scheduler, gputime=False, know_duration=True, ordering=1, blossom=True):
    '''
    new jobs are added to the end of the ending queue
    but in the queue, shortest (gpu) job first be served
    and pack other jobs with the same #GPU according to 
    bipartie graph matching
    NOT FINISHED!!!
    TO DO: unpack if the performance is not improved
    '''
    scheduler._controller.set_start_time()
    last_check_time = 0
    finished_job_cnt = 0
    blossom_cnt = 0
    blossom_sum = 0.
    if FLAGS.fast_forwarding == 0:
        print('Not Implemented!')
    else:
        est_check_time = 0
        running_jobs = 0
        while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            while scheduler.get_time()-last_check_time<FLAGS.schedule_interval and scheduler.has_running_trainers(running_jobs):
                # print("waiting for trainers: ", scheduler.get_time(), running_jobs, scheduler.has_running_trainers(running_jobs), scheduler._trainers.keys())
                time.sleep(5)
            if running_jobs>0:
                time.sleep(2)
            # assert scheduler.has_running_trainers(running_jobs)==False
            tmp_jump_time = max(est_check_time - scheduler.get_time(), 0)
            scheduler._controller._jump_time += tmp_jump_time
            LOG.checkpoint_utils(last_check_time, scheduler)

            # print('deal with done jobs')
            # deal with done jobs
            done_flag = False
            finished_jobs = []
            while not scheduler._controller.done_queue.empty():
                finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
                e_job = JOBS.find_runnable_job(job_id)
                # est_finished_time = finished_time + max(e_job['remaining_iterations']-FLAGS.fast_forwarding, 0)*e_job['real_itertime'][0]
                # e_job['last_finish_time'] = finished_time
                e_packing = e_job['packing']
                for ejob_id in e_packing.packing_jobs:
                    ejob = JOBS.find_runnable_job(ejob_id.job_idx)
                    finished_jobs.append(ejob['job_idx'])
                    if returncode==0:
                        ejob['last_finish_time'] = finished_time
                    else:
                        ejob['status'] = 'PENDING'
                        del ejob['placements'][:]
                        done_flag = True
                    # print('run.py done: ', ejob['job_idx'], ejob['real_itertime'])
                scheduler._trainers.pop(job_id)

            # print('deal with new jobs:')
            # deal with new jobs
            new_flag = False
            while scheduler.has_ready_jobs(scheduler.get_time()):
                event = JOBS.job_events.pop(0)
                assert 'start_jobs' in event
                for s_job in event['start_jobs']:
                    JOBS.move_to_runnable(s_job)
                    s_job['remaining_time'] = s_job['iteration_time'] * s_job['remaining_iterations']
                    s_job['remaining_gputime'] = s_job['remaining_time'] * s_job['num_gpu']
                    s_job['total_executed_time'] = 0
                    s_job['total_executed_gputime'] = 0
                    utils.print_fn('---- job[%d] is added' % s_job['job_idx'])
                    # print('start', s_job['job_idx'], s_job['last_check_time'], scheduler.get_time())
                new_flag = True
            
            tmp_time = scheduler.get_time()
            assert tmp_time - last_check_time > FLAGS.schedule_interval or tmp_time < FLAGS.schedule_interval
            done_job_list = list()
            for rjob in JOBS.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    if rjob['job_idx'] in finished_jobs or running_jobs == 0:
                        finished_time, finished_iter, done_idx = JOBS.calc_packing_finished_info(rjob, tmp_time, last_check_time)
                        rjob['finished_info'] = (finished_time, finished_iter, done_idx)
                    else:
                        job_list = []
                        packing = rjob['packing']
                        packing.calc_iteration_time()
                        max_job_id = -1
                        for pjob in packing.best_permutation:
                            if pjob==None:
                                job_list.append(None)
                            else:
                                tmp_job = JOBS.find_runnable_job(pjob.job_idx)
                                job_list.append(tmp_job)
                                max_job_id = max(max_job_id, tmp_job['job_idx'])
                        while len(job_list)<FLAGS.multi_resource:
                            job_list.append(None)
                        jobinfo = JOBS.to_jobinfo(job_list, True)
                        scheduler._controller.kill(jobinfo)
                        scheduler._trainers.pop(max_job_id)
                        done_flag=True
                        scheduler._controller._logger.info(f'controller, {max_job_id}, timeout')
                        for pjob in packing.best_permutation:
                            if pjob!=None:
                                tmp_job = JOBS.find_runnable_job(pjob.job_idx)
                                finished_jobs.append(tmp_job['job_idx'])
                                if 'placements' in tmp_job:
                                    del tmp_job['placements'][:]
                                tmp_job['status'] = 'PENDING'
                    # print('finished_info: ', rjob['job_idx'], finished_time, finished_iter, done_idx)
            for tmp_num in range(1, FLAGS.packing_num+1):
                # print(tmp_num)
                tmp_list = JOBS.overhead_list[tmp_num]
                tmp_n = len(tmp_list)
                if tmp_n>0:
                    tmp_mean = sum(tmp_list)/tmp_n
                    tmp_ss = sum([(x-tmp_mean)**2 for x in tmp_list])
                    tmp_std = (tmp_ss/tmp_n)**0.5
                    scheduler._logger.info(f'scheduler, error rate between sim and real: {tmp_num} jobs, mean {tmp_mean}, std {tmp_std}')
            for rjob in JOBS.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    tmp = tmp_time - rjob['last_check_time']
                    finished_time, finished_iter, done_idx = rjob['finished_info']
                    for _ in range(done_idx):
                        del rjob['real_itertime'][0]
                    # print('running: ', rjob['job_idx'], rjob['real_itertime'], finished_time, finished_iter)
                    if finished_iter>= rjob['remaining_iterations']:
                        rjob['total_executed_time'] += finished_time-rjob['last_check_time']
                        rjob['last_check_time'] = finished_time
                        rjob['remaining_iterations'] = 0
                        rjob['remaining_time'] = 0
                        packing_ids = [pjob.job_idx for pjob in rjob['packing'].packing_jobs]
                        max_id = max(packing_ids)
                        if max_id == rjob['job_idx']:
                            CLUSTER.release_job_res(rjob)
                        else:
                            rjob['status'] = 'END'
                        LOG.job_complete(rjob, rjob['last_check_time'])
                        finished_job_cnt += 1
                        scheduler._logger.info(f'scheduler finishes {finished_job_cnt} jobs in all!')
                        done_job_list.append(rjob)
                        done_flag=True
                    else:
                        rjob['total_executed_time'] += tmp
                        rjob['remaining_iterations'] -= finished_iter
                        # print(rjob['job_idx'], rjob['remaining_iterations'], finished_iter)
                        rjob['remaining_time'] = rjob['iteration_time'] * rjob['remaining_iterations']
                        if gputime:
                            rjob['remaining_gputime'] = rjob['remaining_time'] * rjob['num_gpu']
                            rjob['total_executed_gputime'] = rjob['total_executed_time'] * rjob['num_gpu']
                        rjob['last_check_time'] = tmp_time
                    # print(tmp_time, 'check: running ', rjob['job_idx'], finished_time, rjob['remaining_iterations'], rjob['last_check_time'], rjob['real_itertime'])
                elif 'PENDING' == rjob['status']:
                    tmp = tmp_time - rjob['last_check_time']
                    rjob['pending_time'] += tmp
                    rjob['last_check_time'] = tmp_time
                    # print(tmp_time, 'check: pending ', rjob['job_idx'], rjob['remaining_iterations'], rjob['last_check_time'], rjob['placements'] if 'placements' in rjob else [])
                elif 'END' == rjob['status']: #almost impossible
                    done_job_list.append(rjob)
                    # print(tmp_time, 'check: ending', rjob['job_idx'], rjob['remaining_iterations'], rjob['last_check_time'])
            for djob in done_job_list:
                JOBS.runnable_jobs.remove(djob)
            running_jobs = 0
            if done_flag or new_flag:
                scheduler.clear_src_utils()
                #sort jobs with shortest first
                for rjob in JOBS.runnable_jobs:
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
                JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('sort_val'))

                run_jobs = list() # pending jobs to running
                preempt_jobs = list() # running jobs to other status
                GPU_num_job = dict() # num_gpu, all runnable jobs
                GPU_chosen_job = dict() # num_gpu, number of chosen jobs for packing
                GPU_nums = dict() # num_gpu, number of possible packings
                required_gpu = 0
                for rjob in JOBS.runnable_jobs:
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
                        # print('select jobs to blossom: ', rjob['job_idx'], rjob['num_gpu'], rjob['sort_val'])
                        up_bd = min(GPU_chosen_job[num_gpu]+FLAGS.packing_num, len(GPU_num_job[num_gpu]))
                        GPU_nums[num_gpu] += 1
                        for tmp_id in range(GPU_chosen_job[num_gpu], up_bd):
                            GPU_num_job[num_gpu][tmp_id]['packing_used']=1
                        GPU_chosen_job[num_gpu] = up_bd
                for key in GPU_num_job.keys():
                    GPU_num_job[key] = GPU_num_job[key][:GPU_chosen_job[key]]
                    required_gpu += GPU_chosen_job[key]*key

                # print("before blossom: ")
                # for key in GPU_num_job.keys():
                #     print([rjob['job_idx'] for rjob in GPU_num_job[key]])
                if blossom==True:
                    time_blossom_0 = time.time()
                    packings = Blossom_Same.run(GPU_num_job, CLUSTER.num_gpu, ordering)
                    time_blossom_1 = time.time()
                    blossom_cnt += 1
                    blossom_sum += time_blossom_1 - time_blossom_0
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

                # print('after blossom: ')
                # for key in packings.keys():
                #     for packing in packings[key]:
                #         print(key, [minijob.job_idx for minijob in packing.packing_jobs])

                CLUSTER.empty_infra()
                vis_flag = dict()
                packings_keys = list(packings.keys()) # possible num_gpu
                packings_keys.sort(reverse=True)
                for packings_key in packings_keys:
                    for packing in packings[packings_key]:
                        job_list = []
                        rjob = None
                        packing.calc_iteration_time(ordering=ordering)
                        for pjob in packing.best_permutation:
                            if pjob==None:
                                job_list.append(None)
                            else:
                                tmp_job = JOBS.find_runnable_job(pjob.job_idx)
                                tmp_job['packing'] = packing
                                # print(tmp_job['job_idx'], tmp_job['placements'], tmp_job['status'])
                                if tmp_job['status']=='RUNNING' and 'placements' in tmp_job:
                                    del tmp_job['placements'][:]
                                vis_flag[pjob.job_idx] = True
                                job_list.append(tmp_job)
                                if rjob == None or tmp_job['job_idx']>rjob['job_idx']:
                                    rjob = tmp_job
                        while len(job_list)<FLAGS.multi_resource:
                            job_list.append(None)
                        # print('placement: ', rjob['job_idx'], [pjob.job_idx if pjob!=None else -1 for pjob in packing.best_permutation], rjob['placements'])
                        ret = try_get_job_res(rjob)
                        if True == ret:
                            running_jobs += 1
                            for tjob in job_list:
                                if tjob!=None:
                                    tjob['job_counter'] += 1
                                    if sys.maxsize == tjob['start_time']:
                                        tjob['start_time'] = tmp_time
                                    if tjob['status'] == 'PENDING':
                                        run_jobs.append(tjob)
                                    if tjob['job_idx'] != rjob['job_idx']:
                                        tjob['placements'] = copy.deepcopy(rjob['placements'])
                            # print('start: ', [tjob['job_idx'] if tjob!=None else -1 for tjob in job_list])
                            jobinfo = JOBS.to_jobinfo(job_list, True)
                            scheduler._controller.execute(jobinfo)
                        else:
                            for tjob in job_list:
                                if tjob!=None and tjob['status'] == 'RUNNING':
                                    preempt_jobs.append(tjob)
                
                for rjob in JOBS.runnable_jobs:
                    if rjob['job_idx'] not in vis_flag and rjob['status'] == 'RUNNING':
                        if 'placements' in rjob:
                            del rjob['placements'][:]
                        preempt_jobs.append(rjob)

                for job in preempt_jobs:
                    job['status'] = 'PENDING'
                    job['preempt'] = int(job['preempt'] + 1)
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, preempt')
                for job in run_jobs:
                    job['status'] = 'RUNNING'
                    job['resume'] = int(job['resume'] + 1)
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, run')

                time.sleep(10)    
            last_check_time = tmp_time
            est_check_time = last_check_time + FLAGS.schedule_interval
            # LOG.checkpoint(tmp_time, scheduler, done_flag or new_flag, secs)
        if blossom==True:
            assert blossom_cnt >0
            blossom_avg = blossom_sum/blossom_cnt
            scheduler._logger.info(f'scheduler, Blossom {blossom_cnt} times, avg time: {blossom_avg}')

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

def themis_sim_jobs(scheduler=None):
    '''
    new jobs are added to the end of the ending queue
    but in the queue, shortest (gpu) job first be served, until no resource
    '''
    # end_events = list()
    num_steps_remaining_prev_iteration, isolated_throughputs_prev_iteration = {}, {}
    cumulative_isolated_time = {} 
    scheduler._controller.set_start_time()
    last_check_time = 0
    finished_job_cnt = 0
    if FLAGS.fast_forwarding == 0:
        while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            done_flag = False
            new_flag = False
            # print('before: ', scheduler.get_time())
            while not scheduler._controller.done_queue.empty():
                finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
                e_job = JOBS.find_runnable_job(job_id)
                if returncode==0:
                    tmp = finished_time-e_job['last_check_time']
                    e_job['total_executed_time'] += tmp
                    e_job['last_check_time'] = finished_time
                    CLUSTER.release_job_res(e_job)
                    scheduler._trainers.pop(e_job['job_idx'])
                    LOG.job_complete(e_job, finished_time)
                    finished_job_cnt += 1
                    scheduler._logger.info(f'scheduler finishes {finished_job_cnt} jobs in all!')
                    JOBS.runnable_jobs.remove(e_job)
                else:
                    e_job['status'] = 'PENDING'
                    scheduler._trainers.pop(e_job['job_idx'])
                    del e_job['placements'][:]
                done_flag = True
                print(scheduler.get_time(), 'check: done ', e_job['job_idx'], finished_time)
            while scheduler.has_ready_jobs(scheduler.get_time()):
                event = JOBS.job_events.pop(0)
                assert 'start_jobs' in event
                for s_job in event['start_jobs']:
                    JOBS.move_to_runnable(s_job)
                    s_job['remaining_time'] = s_job['iteration_time'] * s_job['remaining_iterations']
                    # s_job['remaining_gputime'] = s_job['remaining_time'] * s_job['num_gpu']
                    s_job['deficit'] = 0
                    s_job['time_should_received'] = 0
                    s_job['rounds_received'] = 0
                    utils.print_fn('---- job[%d] is added' % s_job['job_idx'])
                new_flag = True
            
            tmp_time = scheduler.get_time()
            if done_flag or new_flag:
                assert tmp_time - last_check_time > FLAGS.schedule_interval or tmp_time < FLAGS.schedule_interval
                for rjob in JOBS.runnable_jobs:
                    if 'RUNNING' == rjob['status']:
                        tmp = tmp_time - rjob['last_check_time']
                        rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
                        rjob['last_check_time'] = tmp_time
                        finished_iter = scheduler.query_stats([rjob['job_idx']])
                        rjob['remaining_iterations'] -= finished_iter
                        # print(rjob['job_idx'], rjob['remaining_iterations'], finished_iter)
                        rjob['remaining_time'] = rjob['iteration_time'] * rjob['remaining_iterations']
                        print(tmp_time, 'check: running ', rjob['job_idx'], rjob['remaining_iterations'], rjob['total_executed_time'])
                    elif 'PENDING' == rjob['status']:
                        tmp = tmp_time - rjob['last_check_time']
                        rjob['pending_time'] += tmp
                        rjob['last_check_time'] = tmp_time
                        print(tmp_time, 'check: pending ', rjob['job_idx'], rjob['remaining_iterations'], rjob['pending_time'])
                    elif 'END' == rjob['status']: #almost impossible
                        JOBS.runnable_jobs.remove(rjob)
                        print(tmp_time, 'check: ending ', rjob['job_idx'], rjob['remaining_iterations'])
                        pass
                
                if len(JOBS.runnable_jobs)>0:
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
                                r_job['remaining_iterations']) / \
                                isolated_throughputs_prev_iteration[r_job['job_idx']]
                        throughput = r_job['tput']
                        allocation_throughput = throughput * x[job_idx]
                        expected_time_isolated = cumulative_isolated_time[r_job['job_idx']] + \
                        (r_job['remaining_iterations'] / isolated_throughputs[job_idx])
                        expected_time_allocation = tmp_time - r_job['submit_time'] + \
                            (r_job['remaining_iterations'] * cp.inv_pos(allocation_throughput))
                        num_steps_remaining_prev_iteration[r_job['job_idx']] = r_job['remaining_iterations']
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
                    result = cvxprob.solve(solver='ECOS')

                    if cvxprob.status != "optimal":
                        print('WARNING: Allocation returned by policy not optimal!')

                    for i, rjob in enumerate(JOBS.runnable_jobs):
                        if rjob['rounds_received']==0:
                            rjob['sort_val'] = x.value[i]*1e9
                        else:
                            rjob['sort_val'] = x.value[i]/rjob['rounds_received'] #rounds received
                        rjob['allocation'] = x.value[i]
                        rjob['deficit'] = rjob['time_should_received']-rjob['total_executed_time']
                        rjob['time_should_received'] += x.value[i]*FLAGS.schedule_interval

                    JOBS.runnable_jobs.sort(key=lambda e:(e.__getitem__('sort_val'), e.__getitem__('deficit'), e.__getitem__('allocation')), reverse=True)

                    chosen_jobs = list()
                    CLUSTER.empty_infra()
                    for rjob in JOBS.runnable_jobs:
                        if 'RUNNING'==rjob['status']:
                            assert 'placements' in rjob
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
                            if rjob['job_idx'] in scheduler._trainers:
                                scheduler._trainers.pop(rjob['job_idx'])
                            jobinfo = JOBS.to_jobinfo(rjob)
                            scheduler._controller.kill(jobinfo)
                            if 'placements' in rjob:
                                del rjob['placements'][:]
                        if rjob['job_idx'] in chosen_jobs:
                            ret = try_get_job_res(rjob) 
                        else:
                            ret = False
                        if True == ret:
                            rjob['job_counter'] += 1
                            if sys.maxsize == rjob['start_time']:
                                rjob['start_time'] = tmp_time
                            if rjob['status'] == 'PENDING':
                                run_jobs.append(rjob)
                            jobinfo = JOBS.to_jobinfo(rjob)
                            scheduler._controller.execute(jobinfo)
                        else:
                            # rjob['status'] = 'PENDING'
                            if rjob['status'] == 'RUNNING':
                                preempt_jobs.append(rjob)
                            continue

                    for job in preempt_jobs:
                        job['status'] = 'PENDING'
                        job['preempt'] = int(job['preempt'] + 1)
                        scheduler._logger.info(f'scheduler, {job["job_idx"]}, preempt')
                    for job in run_jobs:
                        job['status'] = 'RUNNING'
                        job['resume'] = int(job['resume'] + 1)
                        scheduler._logger.info(f'scheduler, {job["job_idx"]}, run')

                last_check_time = tmp_time
        
            # print('at end: ', scheduler.get_time())
            time00 = time.time()
            time.sleep(10)
            LOG.checkpoint(tmp_time, scheduler, done_flag or new_flag)
            time01 = time.time()
            time.sleep(FLAGS.schedule_interval-(time01-time00))
    else:
        est_check_time = 0
        running_jobs = 0
        while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            while scheduler.get_time()-last_check_time<FLAGS.schedule_interval and scheduler.has_running_trainers(running_jobs):
                # print("waiting for trainers: ", scheduler.get_time(), running_jobs, scheduler.has_running_trainers(running_jobs), scheduler._trainers.keys())
                time.sleep(5)
            assert len(scheduler._trainers.keys())==running_jobs
            if running_jobs>0:
                time.sleep(2)
            # assert scheduler.has_running_trainers(running_jobs)==False
            tmp_jump_time = max(est_check_time - scheduler.get_time(), 0)
            scheduler._controller._jump_time += tmp_jump_time
            LOG.checkpoint_utils(last_check_time, scheduler)

            # print('deal with done jobs')
            # deal with done jobs
            done_flag = False
            finished_jobs = []
            while not scheduler._controller.done_queue.empty():
                finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
                e_job = JOBS.find_runnable_job(job_id)
                finished_jobs.append(e_job['job_idx'])
                # est_finished_time = finished_time + max(e_job['remaining_iterations']-FLAGS.fast_forwarding, 0)*e_job['real_itertime'][0]
                if returncode==0:
                    e_job['last_finish_time'] = finished_time
                else:
                    e_job['status'] = 'PENDING'
                    del e_job['placements'][:]
                    done_flag = True
                    scheduler._controller._logger.info(f'controller, {e_job["job_idx"]}, error {returncode}')
                scheduler._trainers.pop(e_job['job_idx'])

            # print('deal with new jobs:')
            # deal with new jobs
            new_flag = False
            while scheduler.has_ready_jobs(scheduler.get_time()):
                event = JOBS.job_events.pop(0)
                assert 'start_jobs' in event
                for s_job in event['start_jobs']:
                    JOBS.move_to_runnable(s_job)
                    s_job['remaining_time'] = s_job['iteration_time'] * s_job['remaining_iterations']
                    s_job['deficit'] = 0
                    s_job['time_should_received'] = 0
                    utils.print_fn('---- job[%d] is added' % s_job['job_idx'])
                    # print(s_job['job_idx'], s_job['last_check_time'], scheduler.get_time())
                new_flag = True
                        
            for rjob in JOBS.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    if running_jobs>0 and rjob['job_idx'] not in finished_jobs:
                        jobinfo = JOBS.to_jobinfo(rjob)
                        scheduler._controller.kill(jobinfo)
                        scheduler._trainers.pop(rjob['job_idx'])
                        done_flag=True
                        scheduler._controller._logger.info(f'controller, {rjob["job_idx"]}, timeout')
                        finished_jobs.append(rjob['job_idx'])
                        if 'placements' in rjob:
                            del rjob['placements'][:]
                        rjob['status'] = 'PENDING'
            
            tmp_time = scheduler.get_time()
            print(tmp_time, '-------------')
            assert tmp_time - last_check_time > FLAGS.schedule_interval or tmp_time < FLAGS.schedule_interval
            done_job_list = list()
            for rjob in JOBS.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    tmp = tmp_time - rjob['last_check_time']
                    # finished_iter = scheduler.query_stats(rjob['job_idx'])
                    # print('request last_finish_time: ', rjob['job_idx'])
                    tmp_error = (rjob['real_itertime'][0]-rjob['iteration_time'])/rjob['iteration_time']
                    JOBS.overhead_list[1].append(tmp_error)
                    if rjob['last_finish_time']<last_check_time:
                        finished_time = rjob['remaining_iterations'] * rjob['real_itertime'][0]
                    else:
                        finished_time = rjob['last_finish_time']-rjob['last_check_time'] + max(rjob['remaining_iterations']-FLAGS.fast_forwarding, 0)*rjob['real_itertime'][0]
                    if finished_time<=tmp:
                        rjob['total_executed_time'] += finished_time
                        rjob['last_check_time'] += finished_time
                        rjob['remaining_iterations'] = 0
                        rjob['remaining_time'] = 0
                        CLUSTER.release_job_res(rjob)
                        LOG.job_complete(rjob, rjob['last_check_time'])
                        finished_job_cnt += 1
                        scheduler._logger.info(f'scheduler finishes {finished_job_cnt} jobs in all!')
                        done_job_list.append(rjob)
                        done_flag=True
                    else:
                        rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
                        if rjob['last_finish_time']<last_check_time:
                            finished_iter = int((tmp_time - rjob['last_check_time'])/rjob['real_itertime'][0])
                        else:
                            finished_iter = int(FLAGS.fast_forwarding + (tmp_time-rjob['last_finish_time'])/rjob['real_itertime'][0])
                        rjob['remaining_iterations'] -= finished_iter
                        # print(rjob['job_idx'], rjob['remaining_iterations'], finished_iter)
                        rjob['remaining_time'] = rjob['iteration_time'] * rjob['remaining_iterations']
                        rjob['last_check_time'] = tmp_time
                    # print(tmp_time, 'check: running ', rjob['job_idx'], finished_time, rjob['last_finish_time'], rjob['remaining_iterations'], rjob['last_check_time'], rjob['real_itertime'])
                elif 'PENDING' == rjob['status']:
                    tmp = tmp_time - rjob['last_check_time']
                    rjob['pending_time'] += tmp
                    rjob['last_check_time'] = tmp_time
                    # print(tmp_time, 'check: pending ', rjob['job_idx'], rjob['remaining_iterations'], rjob['last_check_time'])
                elif 'END' == rjob['status']: #almost impossible
                    done_job_list.append(rjob)
                    # print(tmp_time, 'check: ending', rjob['job_idx'], rjob['remaining_iterations'], rjob['last_check_time'])
                    pass
            tmp_list = JOBS.overhead_list[1]
            tmp_n = len(tmp_list)
            if tmp_n>0:
                tmp_mean = sum(tmp_list)/tmp_n
                tmp_ss = sum([(x-tmp_mean)**2 for x in tmp_list])
                tmp_std = (tmp_ss/tmp_n)**0.5
                scheduler._logger.info(f'scheduler, error rate between sim and real: mean {tmp_mean}, std {tmp_std}')
            for djob in done_job_list:
                JOBS.runnable_jobs.remove(djob)
            running_jobs = 0
            if (done_flag or new_flag) and len(JOBS.runnable_jobs)>0:
                scheduler.clear_src_utils()
                #sort jobs with shortest first
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
                            r_job['remaining_iterations']) / \
                            isolated_throughputs_prev_iteration[r_job['job_idx']]
                    throughput = r_job['tput']
                    allocation_throughput = throughput * x[job_idx]
                    expected_time_isolated = cumulative_isolated_time[r_job['job_idx']] + \
                    (r_job['remaining_iterations'] / isolated_throughputs[job_idx])
                    expected_time_allocation = tmp_time - r_job['submit_time'] + \
                        (r_job['remaining_iterations'] * cp.inv_pos(allocation_throughput))
                    num_steps_remaining_prev_iteration[r_job['job_idx']] = r_job['remaining_iterations']
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
                result = cvxprob.solve(solver='ECOS')

                if cvxprob.status != "optimal":
                    print('WARNING: Allocation returned by policy not optimal!')

                for i, rjob in enumerate(JOBS.runnable_jobs):
                    if rjob['job_counter']==0:
                        rjob['sort_val'] = x.value[i]*1e9
                    else:
                        rjob['sort_val'] = x.value[i]/rjob['job_counter'] #rounds received
                    rjob['allocation'] = x.value[i]
                    rjob['deficit'] = rjob['time_should_received']-rjob['total_executed_time']
                    rjob['time_should_received'] += x.value[i]*FLAGS.schedule_interval

                JOBS.runnable_jobs.sort(key=lambda e:(e.__getitem__('sort_val'), e.__getitem__('deficit'), e.__getitem__('allocation')), reverse=True)

                chosen_jobs = list()
                CLUSTER.empty_infra()
                for rjob in JOBS.runnable_jobs:
                    if 'RUNNING'==rjob['status']:
                        assert 'placements' in rjob
                        del rjob['placements'][:]
                    ret = try_get_job_res(rjob, True)
                    if True==ret:
                        chosen_jobs.append(rjob['job_idx'])
                
                JOBS.runnable_jobs.sort(key=lambda e: e.__getitem__('num_gpu'), reverse=True)

                # print('---------', tmp_time)
                # for rjob in JOBS.runnable_jobs:
                #     print(rjob['job_idx'], rjob['remaining_time'], rjob['num_gpu'])
                run_jobs = list()
                preempt_jobs = list()
                
                CLUSTER.empty_infra()
                for rjob in JOBS.runnable_jobs:
                    if 'RUNNING' == rjob['status']:
                        if 'placements' in rjob:
                            del rjob['placements'][:]

                    if rjob['job_idx'] in chosen_jobs:
                        ret = try_get_job_res(rjob) 
                    else:
                        ret = False
                    # print('scheduling: ', rjob['job_idx'], rjob['num_gpu'], ret)
                    if True == ret:
                        # print('running: ', rjob['job_idx'], tmp_time, scheduler.get_time())
                        rjob['job_counter'] += 1
                        running_jobs += 1
                        if sys.maxsize == rjob['start_time']:
                            rjob['start_time'] = tmp_time
                        if rjob['status'] == 'PENDING':
                            run_jobs.append(rjob)
                        jobinfo = JOBS.to_jobinfo(rjob)
                        scheduler._controller.execute(jobinfo)
                    else:
                        # rjob['status'] = 'PENDING'
                        if rjob['status'] == 'RUNNING':
                            preempt_jobs.append(rjob)
                        continue

                for job in preempt_jobs:
                    job['status'] = 'PENDING'
                    job['preempt'] = int(job['preempt'] + 1)
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, preempt')
                for job in run_jobs:
                    job['status'] = 'RUNNING'
                    job['resume'] = int(job['resume'] + 1)
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, run')

                time.sleep(20)
            last_check_time = tmp_time
            est_check_time = last_check_time + FLAGS.schedule_interval
            # LOG.checkpoint(tmp_time, scheduler, done_flag or new_flag, secs)


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
            utils.print_fn("This cluster is not large enough to run the job")
            break
        event = JOBS.job_events[0]
        event_time = event['time']
        # utils.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
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
                utils.print_fn('----job[%d] starts from pending' % p_job['job_idx'])
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
                utils.print_fn('----job[%d] move to pending' % s_job['job_idx'])
            else:
                #if job is migratable, add into migratable job list
                # JOBS.add_migratable(s_job)
                JOBS.add_job_end_event(s_job)
                utils.print_fn('----job[%d] starts' % s_job['job_idx'])
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
            utils.print_fn("This cluster is not large enough to run the job")
            break
        event = JOBS.job_events[0]
        event_time = event['time']
        # utils.print_fn('--------------------------------- Handle event[time %d]------------------------------------' % event_time)
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
            utils.print_fn('----job[%d] starts' % s_job['job_idx'])
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
            utils.print_fn("In Main, multi-dlas-gpu without count")
            exit()
    ''' Parse input'''
    parse_job_file(FLAGS.trace_file)
    parse_cluster_spec()

    ''' prepare logging '''
    LOG.init_log()

    ''' scheduler '''
    scheduler = Scheduler(FLAGS.scheduler_port, FLAGS.controller_port)

    # lp.placement(JOBS.job_list[0])
    ''' Prepare jobs'''
    JOBS.prepare_job_start_events()

    # sim_job_events()
    if FLAGS.schedule == 'shortest':
        shortest_first_sim_jobs(scheduler)
    elif FLAGS.schedule == 'shortest-gpu':
        shortest_first_sim_jobs(scheduler, True)
    elif FLAGS.schedule == 'dlas-gpu':
        dlas_sim_jobs(scheduler, True)
    elif FLAGS.schedule == 'multi-resource-blossom-same-gpu':
        multi_resource_blossom_same_sim_jobs(scheduler, True)
    elif FLAGS.schedule == 'multi-resource-blossom-same-gpu-unaware':
        multi_resource_blossom_same_sim_jobs(scheduler, True, know_duration=False)
    elif FLAGS.schedule == 'multi-resource-blossom-same-gpu-unaware-worstordering':
        multi_resource_blossom_same_sim_jobs(scheduler, True, know_duration=False, ordering=2)
    elif FLAGS.schedule == 'multi-resource-gpu-unaware':
        multi_resource_blossom_same_sim_jobs(scheduler, True, know_duration=False, blossom=False)
    elif FLAGS.schedule == 'themis':
        themis_sim_jobs(scheduler, )
    else:
        print('not support scheduler') 

    scheduler._controller.kill_workers()

if __name__ == '__main__':
    # print('Hello world %d' % 2)
    main()