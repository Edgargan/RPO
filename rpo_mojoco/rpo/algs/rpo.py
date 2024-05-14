import numpy as np
import tensorflow as tf

from rpo.algs.base_alg import BaseAlg

class RPO(BaseAlg):
    """Algorithm class for rpo"""

    def __init__(self,seed,env,actor,critic,runner,params):
        super(RPO,self).__init__(seed,env,actor,critic,runner,params)
        self._ac_setup()

    def _ac_setup(self):
        self.vf_lr = self.ac_kwargs['vf_lr']
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.vf_lr)

        self.adv_center = self.ac_kwargs['adv_center']
        self.adv_scale = self.ac_kwargs['adv_scale']
        self.actor_lr = self.ac_kwargs['actor_lr']
        if self.ac_kwargs['scaleinitlr']:
            self.actor_lr = self.actor_lr * self.ac_kwargs['eps_mult']
        self.actor_opt_type = self.ac_kwargs['actor_opt_type']
        self.update_it = self.ac_kwargs['update_it']                
        self.nminibatch = self.ac_kwargs['nminibatch']                
        self.eps = self.ac_kwargs['eps_ppo']
        self.eps_next = self.ac_kwargs['eps_ppo_next']
        self.eps_next_reg = self.ac_kwargs['eps_ppo_next_reg']
        self.max_grad_norm = self.ac_kwargs['max_grad_norm']
        
        self.adaptlr = self.ac_kwargs['adaptlr']                      
        self.adapt_factor = self.ac_kwargs['adapt_factor']             
        self.adapt_minthresh = self.ac_kwargs['adapt_minthresh']     
        self.adapt_maxthresh = self.ac_kwargs['adapt_maxthresh']     

        self.early_stop = self.ac_kwargs['early_stop']

        if self.actor_opt_type == 'Adam':
            self.actor_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.actor_lr)
        elif self.actor_opt_type == 'SGD':
            self.actor_optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.actor_lr)
        else:
            raise ValueError('actor_opt_type must be Adam or SGD')
    
    def _get_neg_pg(self,s_active,a_active,adv_active,neglogp_old_active, mask_idx, d_active,
                    s_next_active, a_next_active, adv_next_active, neglogp_old_next_active):


        adv_mean = np.mean(adv_active)
        adv_next_mean = np.mean(adv_next_active)
        adv_std = np.std(adv_active) + 1e-8
        adv_next_std = np.std(adv_next_active) + 1e-8

        if self.adv_center:                     
            adv_active = adv_active - adv_mean 
            adv_next_active = adv_next_active - adv_next_mean
        if self.adv_scale:                       
            adv_active = adv_active / adv_std
            adv_next_actdive = adv_next_active / adv_next_std

        with tf.GradientTape() as tape:
            neglogp_cur_active = self.actor.neglogp(s_active,a_active)
            ratio = tf.exp(neglogp_old_active - neglogp_cur_active)
            ratio_clip = tf.clip_by_value(ratio, 1.-self.eps, 1.+self.eps)
            pg_loss1_1 = ratio * adv_active * -1
            pg_loss1_2 = ratio_clip * adv_active * -1   # shape (32,)
            pg_loss1 = tf.reduce_mean(tf.maximum(pg_loss1_1, pg_loss1_2))

            # next
            neglogp_cur_next_active = self.actor.neglogp(s_next_active,a_next_active)
            ratio_next = tf.exp(neglogp_old_next_active - neglogp_cur_next_active)
            ratio_clip_next = tf.clip_by_value(ratio_next, 1. - self.eps_next, 1. + self.eps_next)
            pg_loss2_1 = ratio * ratio_next * adv_next_active * -1
            pg_loss2_2 = ratio_clip * ratio_clip_next * adv_next_active * -1
            pg_loss2 = tf.reduce_mean(tf.maximum(pg_loss2_1, pg_loss2_2)* mask_idx * d_active)

            pg_loss = pg_loss1 + pg_loss2 * self.eps_next_reg

        neg_pg = tape.gradient(pg_loss,self.actor.trainable)
        
        return neg_pg

    def _update(self, sim_total=0):
        data_all = self.runner.get_update_info(self.actor,self.critic)
        s_all, a_all, adv_all, rtg_all, neglogp_old_all, weights_all, state_f,\
        s_reg_m, s_reg_v, r_reg_m, r_reg_v, d_all = data_all

        n_samples = s_all.shape[0]
        # print('n_samples', n_samples)
        n_batch = int(n_samples / self.nminibatch)
        # print('n_batch', n_batch)

        mean_old = self.actor.sample(s_all,mean=True,clip=False)
        logstd_old = self.actor.logstd.numpy()

        pg_norm_all_pre = 0
        pg_norm_all = 0
        v_loss_all = 0
        vg_norm_all = 0


        for sweep_it in range(self.update_it):
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            sections = np.arange(0,n_samples,n_batch)[1:]


            batches = np.array_split(idx,sections)

            if (n_samples % n_batch != 0):
                batches = batches[:-1]

            for batch_idx in batches:
                # Active data
                s_active = s_all[batch_idx]
                a_active = a_all[batch_idx]
                adv_active = adv_all[batch_idx]
                rtg_active = rtg_all[batch_idx]
                d_active = 1 - d_all[batch_idx]
                neglogp_old_active = neglogp_old_all[batch_idx]

                batch_idx_next = (batch_idx+1) % 2048
                mask_idx = ((batch_idx+1) < 2048).astype(float)
                s_next_active = s_all[batch_idx_next]
                a_next_active = a_all[batch_idx_next]
                adv_next_active = adv_all[batch_idx_next]
                rtg_next_active = rtg_all[batch_idx_next]
                neglogp_old_next_active = neglogp_old_all[batch_idx_next]


                # Critic Update
                with tf.GradientTape() as tape:
                    V = self.critic.value(s_active, False)
                    v_loss = 0.5 * tf.reduce_mean(tf.square(rtg_active - V))
                
                vg = tape.gradient(v_loss, self.critic.trainable)
                self.critic_optimizer.apply_gradients(zip(vg, self.critic.trainable))

                v_loss_all += v_loss
                vg_norm_all += tf.linalg.global_norm(vg)

                # Actor Update
                neg_pg = self._get_neg_pg(s_active, a_active, adv_active, neglogp_old_active, mask_idx, d_active,
                                          s_next_active, a_next_active, adv_next_active, neglogp_old_next_active)
                
                if self.max_grad_norm is not None:
                    neg_pg, pg_norm_pre = tf.clip_by_global_norm(neg_pg, self.max_grad_norm)
                else:
                    pg_norm_pre = tf.linalg.global_norm(neg_pg)
                
                self.actor_optimizer.apply_gradients(zip(neg_pg, self.actor.trainable))
                
                pg_norm_all_pre += pg_norm_pre
                pg_norm_all += tf.linalg.global_norm(neg_pg)

            neglogp_cur_all = self.actor.neglogp(s_all, a_all)
            ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
            ratio_diff = tf.abs(ratio - 1)
            tv = 0.5 * tf.reduce_mean(ratio_diff)

        log_actor = {
            'tv': tv.numpy()
        }
        # self.logger.log_train(log_actor)

        info = log_actor
        
        self.actor.update_old_weights()

        return info
