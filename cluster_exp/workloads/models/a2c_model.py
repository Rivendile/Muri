# add for RL
from .deep_rl import *


class A2CModel:
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
        self.device = torch.device("cuda")
        Config.DEVICE = self.device
        kwargs = dict()
        kwargs['log_level'] = 0
        kwargs['game'] = 'BreakoutNoFrameskip-v4'
        config = Config()
        config.merge(kwargs)

        config.num_workers = self.sargs["batch_size"]
        config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
        config.eval_env = Task(config.game)
        config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
        config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody())
        config.state_normalizer = ImageNormalizer()
        config.reward_normalizer = SignNormalizer()
        config.discount = 0.99
        config.use_gae = True
        config.gae_tau = 1.0
        config.entropy_weight = 0.01
        config.rollout_length = self.args.rollout_length
        config.gradient_clip = 5
        config.max_steps = int(2e7)

        self.model = A2CAgent(config)
        self.config = self.model.config

        self.optimizer = hvd.DistributedOptimizer(self.model.optimizer, named_parameters=self.model.network.named_parameters(prefix='model'+str(self.idx)))

        hvd.broadcast_parameters(self.model.network.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

    def get_data(self):
        '''
        get data
        '''
        self.storage = Storage(self.config.rollout_length)
        states = self.model.states
        for _ in range(self.model.config.rollout_length):
            prediction = self.model.network(self.model.config.state_normalizer(states))
            next_states, rewards, terminals, info = self.model.task.step(to_np(prediction['action']))
            self.model.record_online_return(info)
            rewards = self.model.config.reward_normalizer(rewards)
            self.storage.feed(prediction)
            self.storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                        'mask': tensor(1 - terminals).unsqueeze(-1)})

            states = next_states
            self.model.total_steps += self.model.config.num_workers
        return states

    
    def forward_backward(self, thread):
        '''
        forward, calculate loss and backward
        '''
        thread.join()
        states = thread.get_result()
        self.model.states = states

        prediction = self.model.network(self.model.config.state_normalizer(states))
        self.storage.feed(prediction)
        self.storage.placeholder()

        advantages = tensor(np.zeros((self.model.config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(self.model.config.rollout_length)):
            returns = self.storage.reward[i] + self.model.config.discount * self.storage.mask[i] * returns
            if not self.model.config.use_gae:
                advantages = returns - self.storage.v[i].detach()
            else:
                td_error = self.storage.reward[i] + self.model.config.discount * self.storage.mask[i] * self.storage.v[i + 1] - self.storage.v[i]
                advantages = advantages * self.model.config.gae_tau * self.model.config.discount * self.storage.mask[i] + td_error
            self.storage.advantage[i] = advantages.detach()
            self.storage.ret[i] = returns.detach()

        entries = self.storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        self.model.optimizer.zero_grad()
        (policy_loss - self.model.config.entropy_weight * entropy_loss +
        self.model.config.value_loss_weight * value_loss).backward()
        
    def comm(self):
        # self.optimizer.synchronize()
        # nn.utils.clip_grad_norm_(self.model.network.parameters(), self.model.config.gradient_clip)
        # with self.optimizer.skip_synchronize():
        self.optimizer.step()

    def print_info(self):
        print("Model ", self.idx, ": ", self.sargs["model_name"], "; batch size: ", self.sargs["batch_size"], "; rollout length: ", self.args.rollout_length)

    def data_size(self):
        return 0