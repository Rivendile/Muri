from runtime.rpc import trainer_client, trainer_server

import argparse
import utils
import time
import threading


class Trainer(object):
    def __init__(self, scheduler_ip, scheduler_port, trainer_ip, trainer_port, job_id) -> None:
        super().__init__()

        self._trainer_ip = trainer_ip
        self._trainer_port = trainer_port
        self._job_id = job_id
        # self._batch_size = batch_size
        # self._demotion_threshold = demotion_threshold

        self._logger = utils.make_logger(__name__)
        self._start_time = time.time()
        self._finished_iteraions = 0

        self._client_for_scheduler = trainer_client.TrainerClientForScheduler(self._logger, scheduler_ip, scheduler_port)
        self.init_stats()

        self._server_for_scheduler = self.make_server_for_scheduler(self._trainer_port)

        self.register()

        self._logger.info(f'job {self._job_id}, trainer, start, {self._start_time}')

    def register(self):
        success = False
        while success == False:
            success = self._client_for_scheduler.register_trainer(self._trainer_ip, self._trainer_port, self._job_id)

    def report_itertime(self, iter_time, src_utils):
        success = self._client_for_scheduler.report_itertime(self._job_id, iter_time, src_utils)
        self._logger.info(f'job {self._job_id} reported iteration time {iter_time} and resource utils {src_utils}')

    def init_stats(self):
        pass
    

    def update_stats(self, iteration_time):
        self._finished_iteraions += 1
        self._logger.info(f'trainer update_stats: {self._finished_iteraions}, {iteration_time}')


    def record(self, iteration_time):
        self.update_stats(iteration_time)

        # if self.demotion() == True:
        #     self._client_for_scheduler.report_stats(self._job_id, self._finished_iteraions, True)



    def make_server_for_scheduler(self, port: int):
        callbacks = {
            'QueryStats' : self._query_stats_impl,
        }

        server_thread = threading.Thread(
            target=trainer_server.serve,
            args=(port, self._logger, callbacks))
        server_thread.setDaemon(True)
        server_thread.start()

        return server_thread


    def _query_stats_impl(self):
        self._logger.info(f'trainer query stats, {self._finished_iteraions}')
        return self._finished_iteraions


    # def demotion(self) -> bool:
    #     if self._demotion_threshold == None:
    #         return False
        
    #     return (time.time() - self._start_time >= self._demotion_threshold)


if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler_ip', type=str, required=True)
    parser.add_argument('--scheduler_port', type=int, default=9011)
    parser.add_argument('--trainer_port', type=int)
    parser.add_argument('--job_id', type=int, default=-1)
    # parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--demotion_threshold', type=float, default=None)
    args = parser.parse_args()

    trainer = Trainer(args.scheduler_ip, args.scheduler_port, utils.get_host_ip(), args.trainer_port, args.job_id)