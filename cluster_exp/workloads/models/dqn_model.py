# add for RL
from .deep_rl import *


class DQNModel:
    def __init__(self, idx, args, sargs):
        self.idx = idx
        self.args = args
        self.sargs = sargs # specific args for this model
    
    def prepare(self, hvd):
        '''
        prepare dataloader, model, optimizer for training
        '''
        if hvd.local_rank()==0:
            mkdir('log')
            mkdir('tf_log')
        self.device = torch.device("cuda:%d" % (hvd.local_rank()))
        Config.DEVICE = self.device
        kwargs = dict()
        kwargs['log_level'] = 0
        kwargs['n_step'] = 1
        kwargs['replay_cls'] = UniformReplay
        kwargs['async_replay'] = False
        kwargs['game'] = 'BreakoutNoFrameskip-v4'
        kwargs['run'] = 0
        config = Config()
        config.merge(kwargs)
        config.task_fn = lambda: Task(config.game)
        config.eval_env = config.task_fn()
        config.optimizer_fn = lambda params: torch.optim.RMSprop(
            params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
        config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
        config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)
        config.batch_size = self.sargs['batch_size']
        config.discount = 0.99
        config.history_length = 4
        config.max_steps = int(2e7)
        replay_kwargs = dict(
            memory_size=int(1e6),
            batch_size=config.batch_size,
            n_step=config.n_step,
            discount=config.discount,
            history_length=config.history_length,
        )
        config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
        config.replay_eps = 0.01
        config.replay_alpha = 0.5
        config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

        config.state_normalizer = ImageNormalizer()
        config.reward_normalizer = SignNormalizer()
        config.target_network_update_freq = 10000
        config.exploration_steps = config.batch_size
        # config.exploration_steps = 100
        config.sgd_update_frequency = 4
        config.gradient_clip = 5
        config.double_q = False
        config.async_actor = False
        # if hvd.rank()==0:
        #     print(config)
        self.model = DQNAgent(config)
        # print("build model!!!!!")
        self.optimizer = hvd.DistributedOptimizer(self.model.optimizer, named_parameters=self.model.network.named_parameters(prefix='model'+str(self.idx)))
        hvd.broadcast_parameters(self.model.network.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        self.config = config
        
    def get_data(self):
        '''
        get data
        '''
        for _ in range(self.args.play_times):
            transitions = self.model.actor.step()
            for states, actions, rewards, next_states, dones, info in transitions:
                self.model.record_online_return(info)
                self.model.total_steps += 1
                self.model.replay.feed(dict(
                    state=np.array([s[-1] if isinstance(s, LazyFrames) else s for s in states]),
                    action=actions,
                    reward=[self.config.reward_normalizer(r) for r in rewards],
                    mask=1 - np.asarray(dones, dtype=np.int32),
                ))
                states = next_states
        
        transitions = self.model.replay.sample()
        return transitions
    
    def forward_backward(self, thread):
        '''
        forward, calculate loss and backward
        '''
        thread.join()
        transitions = thread.get_result()

        if self.model.total_steps < self.config.exploration_steps:
            return

        if self.config.noisy_linear:
            self.model.target_network.reset_noise()
            self.model.network.reset_noise()
        loss = self.model.compute_loss(transitions)
        if isinstance(transitions, PrioritizedTransition):
            priorities = loss.abs().add(self.config.replay_eps).pow(self.config.replay_alpha)
            idxs = tensor(transitions.idx).long()
            self.model.replay.update_priorities(zip(to_np(idxs), to_np(priorities)))
            sampling_probs = tensor(transitions.sampling_prob)
            weights = sampling_probs.mul(sampling_probs.size(0)).add(1e-6).pow(-self.config.replay_beta())
            weights = weights / weights.max()
            loss = loss.mul(weights)

        loss = self.model.reduce_loss(loss)
        self.optimizer.zero_grad()
        loss.backward()

    def comm(self):
        with self.model.config.lock:
        # self.optimizer.synchronize()
        # nn.utils.clip_grad_norm_(self.model.network.parameters(), self.model.config.gradient_clip)
        # with self.optimizer.skip_synchronize():
            self.optimizer.step()
        if self.model.total_steps / self.model.config.sgd_update_frequency % \
                self.model.config.target_network_update_freq == 0:
            self.model.target_network.load_state_dict(self.model.network.state_dict())

    def print_info(self):
        print("Model ", self.idx, ": ", self.sargs["model_name"], "; batch size: ", self.sargs["batch_size"], "; play times: ", self.args.play_times)

    def data_size(self):
        return 0