import numpy as np

from rpo.common.initializers import init_seeds
from rpo.common.logger import Logger
import gym
import tensorflow as tf
import os
import pickle
class BaseAlg:
    """Base algorithm class for training"""

    def __init__(self,seed,env,actor,critic,runner,params):

        self.seed = seed
        self.env = env
        self.actor = actor
        self.critic = critic
        self.runner = runner
        self.params = params

        self.ac_kwargs = params['ac_kwargs']

        init_seeds(self.seed,env)
        self.logger = Logger(self.params)
    
    def _update(self,sim_total=0):
        raise NotImplementedError # Algorithm specific

    def update_old_weights(self):
        raise NotImplementedError # Algorithm specific

    def eval_episode(self):
        state = self.env.reset()
        state_raw, _, _ = self.env.get_raw()
        episodic_return = 0
        done = False
        while not done:
            action = self.actor.sample(state, mean=True)
            state, reward, done, _ = self.env.step(action)
            _, _, reward_raw_true = self.env.get_raw()

            episodic_return += reward_raw_true
        return episodic_return

    def eval_episodes(self):
        episodic_returns = []
        for i in range(10):
            episodic_returns.append(self.eval_episode())
        avg_returns = np.mean(episodic_returns)
        return avg_returns


    def record_values(self, info, step):
        for k, v in info.items():
            self.logger.add_scalar(k, v, step)


    def learn(self,sim_size,no_op_batches):   


        for _ in range(no_op_batches):                
            self.runner.generate_batch(self.env,self.actor)
            s_raw, rtg_raw = self.runner.get_env_info()
            self.env.update_rms(s_raw,rtg_raw)
            self.runner.reset()

        sim_total = 0
        while sim_total < sim_size:
            self.runner.generate_batch(self.env,self.actor)    
            
            log_ent = {'ent': np.squeeze(self.actor.entropy())}

            info = self._update()

            s_raw, rtg_raw = self.runner.get_env_info()
            self.env.update_rms(s_raw,rtg_raw)

            log_info = self.runner.get_log_info()
            # self.logger.log_train(log_info)

            sim_total += self.runner.steps_total
            kl_verify, length = self.runner.update(self.actor)



            if sim_total % 4096 == 0:
                episodic_return_test = self.eval_episodes()

                info["episodic_return_test"] = episodic_return_test
                info["episodic_return_train"] = log_info['J_tot']
                info["entropy"] = log_ent["ent"]
                info["kl_verify"] = kl_verify
                info["length"] = length

                print(f"Env: {self.params['env_name']} Seed: {self.params['ac_seed']} Total T: {sim_total+1} Reward: {episodic_return_test}")

                self.record_values(info, sim_total)

                current = {
                    'actor_weights': self.actor.get_weights(),
                    'critic_weights': self.critic.get_weights(),
                }
                self.logger.save_current(current, self.params["log_path"]+'/Model', 'steps_%s_seed_%s' % (str(sim_total), str(self.seed)))

        final = {
            'actor_weights':    self.actor.get_weights(),
            'critic_weights':   self.critic.get_weights(),
        }
        self.logger.log_final(final)
        
    def dump(self,params):
        self.logger.log_params(params)
        return self.logger.dump()

    def save(self,params,log_path,log_name):
        self.logger.log_params(params)
        self.logger.save(log_path,log_name)

