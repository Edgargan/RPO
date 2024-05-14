#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import  torch


class RPO(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.lr_scheduler = lambda params: torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=params)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state)
        action = to_np(prediction['mean'])
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for step in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                         'mask': tensor(1 - terminals).unsqueeze(-1),
                         'state': tensor(states)})
            if step < config.rollout_length-1:
                storage.feed({'mask_next': tensor(np.ones_like(terminals)).unsqueeze(-1)})
            else:
                storage.feed({'mask_next': tensor(np.zeros_like(terminals)).unsqueeze(-1)})

            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        storage.feed(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        entries = storage.extract(['state', 'action', 'log_pi_a', 'ret', 'advantage', 'mask', 'mask_next'])  
        EntryCLS = entries.__class__                                                
        entries = EntryCLS(*list(map(lambda x: x.detach(), entries)))
        entries.advantage.copy_((entries.advantage - entries.advantage.mean()) / entries.advantage.std())

        length = entries.state.size(0)
        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(entries.state.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = EntryCLS(*list(map(lambda x: x[batch_indices], entries)))
                prediction = self.network(entry.state, entry.action)
                ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
                ratio_clip = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                         1.0 + self.config.ppo_ratio_clip)
                obj = ratio * entry.advantage
                obj_clipped = ratio_clip * entry.advantage
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['entropy'].mean()
                value_loss = 0.5 * (entry.ret - prediction['v']).pow(2).mean()

                # second step
                next_batch_indices = batch_indices + 1
                next_batch_indices_mask = (next_batch_indices < length).int()
                next_batch_indices_mask_1 = (next_batch_indices < length).int().unsqueeze(dim=-1)
                next_entry = EntryCLS(*list(map(lambda x: x[next_batch_indices * next_batch_indices_mask], entries)))
                next_prediction = self.network(next_entry.state, next_entry.action)
                next_ratio = (next_prediction['log_pi_a'] - next_entry.log_pi_a).exp()
                next_ratio_clip = next_ratio.clamp(1.0 - self.config.ppo_next_ratio_clip,
                                                   1.0 + self.config.ppo_next_ratio_clip)
                next_mask = next_batch_indices_mask_1 * entry.mask * entry.mask_next
                next_obj = ratio * next_ratio * next_entry.advantage
                next_obj_clip = ratio_clip * next_ratio_clip * next_entry.advantage
                next_policy_loss = -(torch.min(next_obj, next_obj_clip) * next_mask).mean()


                total_policy_loss = policy_loss + config.regu * next_policy_loss
                self.opt.zero_grad()
                (total_policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

            sampler = random_sample(np.arange(entries.state.size(0)), 1024)
            batch_indices = tensor(list(sampler)[0]).long()
            entry = EntryCLS(*list(map(lambda x: x[batch_indices], entries)))
            prediction = self.network(entry.state, entry.action)
            ratio1 = (entry.log_pi_a - prediction['log_pi_a']).exp()
            TV = 0.5 * (ratio1-1).abs().mean().item()

        if self.total_steps % 4096 == 0:
            sampler = random_sample(np.arange(entries.state.size(0)), 256)
            batch_indices = tensor(list(sampler)[0]).long()  
            entry = EntryCLS(*list(map(lambda x: x[batch_indices], entries)))

            prediction = self.network(entry.state, entry.action)
            approx_kl = (entry.log_pi_a - prediction['log_pi_a'])
            true_entropy = prediction['entropy']
            ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.config.ppo_ratio_clip).float()).item()
            info = {
                    'true_entropy_mean': true_entropy.mean(),
                    'true_entropy_median': true_entropy.median(),
                    'true_entropy_max': true_entropy.max(),
                    'approx_kl_mean': approx_kl.mean(),
                    'approx_kl_median': approx_kl.median(),
                    'approx_kl_max': approx_kl.max(),
                    'ratio_mean': ratio.mean(),
                    'ratio_median': ratio.median(),
                    'ratio_max': ratio.max(),
                    'clip_fraction': clip_fraction,
                    'evaluate_return_train': np.mean(list(self.evaluate_return_train)),
                    'lr': self.opt.param_groups[0]['lr']
                    }
            self.record_actionvalue(info)

