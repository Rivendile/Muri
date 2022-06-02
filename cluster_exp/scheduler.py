from runtime.rpc import scheduler_server, scheduler_client
from controller import Controller
from cluster import CLUSTER

import argparse
import threading
import utils
import copy
from jobs import JOBS


class Scheduler(object):
    def __init__(self, scheduler_port: int, controller_port: int) -> None:
        super().__init__()

        self._logger = utils.make_logger(__name__)

        self._trainers = dict()
        self._server_for_trainer = self.make_server_for_trainer(scheduler_port)

        self._num_workers = CLUSTER.num_node_p_switch
        self._controller = Controller(controller_port, self._num_workers)
        self._src_num = 3
        self._src_utils = [0 for _ in range(self._src_num)]

        # self._start_time = self._controller.get_time()

    def get_time(self):
        return self._controller.get_time()

    def make_server_for_trainer(self, port):
        callbacks = {
            'RegisterTrainer': self._register_trainer_impl,
            'ReportIterTime': self._report_itertime_impl, 
        }

        server_thread = threading.Thread(
            target=scheduler_server.serve,
            args=(port, self._logger, callbacks))
        server_thread.setDaemon(True)
        server_thread.start()

        return server_thread
    

    def _register_trainer_impl(self, trainer_ip, trainer_port, job_id_list):
        success = True
        # self._logger.info(f'scheduler, before register, {job_id} {trainer_ip}:{trainer_port} {self._trainers.keys()}')
        job_id = max(job_id_list)
        # assert job_id not in self._trainers
        tmp_client = scheduler_client.SchedulerClientForTrainer(self._logger, job_id_list, trainer_ip, trainer_port)
        self._trainers[job_id] = tmp_client
        self._logger.info(f'scheduler, register, {job_id}-{job_id_list}, {trainer_ip}:{trainer_port}')

        return success

    def _report_itertime_impl(self, job_id, iter_time, src_utils):
        success = True
        num_gpu = 0
        for rjob_id in job_id:
            if rjob_id>=0:
                rjob = JOBS.find_runnable_job(rjob_id)
                rjob['real_itertime'] = copy.deepcopy(list(iter_time))
                num_gpu = rjob['num_gpu']
        for i in range(self._src_num): # cpu util is approximate
            self._src_utils[i] += src_utils[i]*num_gpu
        self._logger.info(f'scheduler, update job {job_id} iter_time {list(iter_time)}; src_utils {src_utils} -> {self._src_utils}')
        return success

    def query_stats(self, job_id_list):
        job_id = max(job_id_list)
        assert job_id in self._trainers
        finished_iterations = self._trainers[job_id].query_stats()
        return finished_iterations

    def has_ready_jobs(self, tmp_time):
        if len(JOBS.job_events)>0 and JOBS.job_events[0]['time']<=tmp_time:
            return True 
        else:
            return False

    def has_running_trainers(self, running_jobs):
        if running_jobs>self._controller.done_queue.qsize():
            return True
        else:
            return False
    
    def clear_src_utils(self):
        self._src_utils = [0 for _ in range(self._src_num)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler_port', type=int, default=9011)
    parser.add_argument('--controller_port', type=int, default=9012)
    args = parser.parse_args()

    scheduler = Scheduler(args.scheduler_port, args.controller_port)