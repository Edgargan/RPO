import numpy as np

from rpo.common.runner_utils import aggregate_data, gae_all, reward_calc, gae, gae1

class Runner:
    """Class for running simulations and storing recent simulation data"""

    def __init__(self,T,gamma,lam,b_size,is_trunc,M,weights): 
        self.T = T
        self.gamma = gamma
        self.lam = lam
        self.b_size = b_size

        self.is_trunc = is_trunc

        self.noldpols = M - 1    
        self.weights = weights

        self.reset()
    
    def reset(self):
        self.reset_buffer()
        self.reset_cur()
    
    def update(self, actor):
        if self.noldpols > 0:
            self.update_buffer()
        self.reset_cur()
        return 0, 0
    
    def reset_cur(self):
        self.s_batch = []
        self.s_raw_batch = []
        self.a_batch = []
        self.r_batch = []
        self.sp_batch = []
        self.d_batch = []
        self.neglogp_batch = []
        self.k_batch = []
        self.rtg_raw_batch = []
        self.J_tot_batch = []
        self.J_disc_batch = []

        self.traj_total = 0
        self.steps_total = 0

        self.state_f_batch = []
        self.s_reg_m_batch = []
        self.s_reg_v_batch = []
        self.r_reg_m_batch = []
        self.r_reg_v_batch = []

    def reset_buffer(self):
        self.s_buffer = []
        self.a_buffer = []
        self.r_buffer = []    
        self.sp_buffer = []
        self.d_buffer = []
        self.neglogp_buffer = []
        self.k_buffer = []

        self.k = 0
        self.next_idx = 0
        self.full_buffer = False

        self.state_f_buffer = []
        self.s_reg_m_buffer = []
        self.s_reg_v_buffer = []
        self.r_reg_m_buffer = []
        self.r_reg_v_buffer = []
    
    def update_buffer(self):
        if self.next_idx >= len(self.s_buffer):
            self.s_buffer.append(self.s_batch)
            self.a_buffer.append(self.a_batch)
            self.r_buffer.append(self.r_batch)
            self.sp_buffer.append(self.sp_batch)
            self.d_buffer.append(self.d_batch)
            self.neglogp_buffer.append(self.neglogp_batch)
            self.k_buffer.append(self.k_batch)

            self.state_f_buffer.append(self.state_f_batch)
            self.s_reg_m_buffer.append(self.s_reg_m_batch)
            self.s_reg_v_buffer.append(self.s_reg_v_batch)
            self.r_reg_m_buffer.append(self.r_reg_m_batch)
            self.r_reg_v_buffer.append(self.r_reg_v_batch)
        else:
            self.s_buffer[self.next_idx] = self.s_batch
            self.a_buffer[self.next_idx] = self.a_batch
            self.r_buffer[self.next_idx] = self.r_batch
            self.sp_buffer[self.next_idx] = self.sp_batch
            self.d_buffer[self.next_idx] = self.d_batch
            self.neglogp_buffer[self.next_idx] = self.neglogp_batch
            self.k_buffer[self.next_idx] = self.k_batch

            self.state_f_buffer[self.next_idx] = self.state_f_batch
            self.s_reg_m_buffer[self.next_idx] = self.s_reg_m_batch
            self.s_reg_v_buffer[self.next_idx] = self.s_reg_v_batch
            self.r_reg_m_buffer[self.next_idx] = self.r_reg_m_batch
            self.r_reg_v_buffer[self.next_idx] = self.r_reg_v_batch

        self.k += 1
        self.next_idx = (self.next_idx + 1) % self.noldpols

        if len(self.s_buffer) == self.noldpols:
            self.full_buffer = True
        else:
            self.full_buffer = False

    def _generate_traj(self,env,actor):
        """Generates single trajectory"""
        s_traj = []
        s_raw_traj = []
        a_traj = []
        r_traj = []
        r_raw_traj = []
        r_raw_true_traj = []
        sp_traj = []
        d_traj = []
        neglogp_traj = []

        state_f_traj = []
        s_reg_m_traj = []
        s_reg_v_traj = []
        r_reg_m_traj = []
        r_reg_v_traj = []

        S_M = []

        full = True

        s = env.reset()
        s_raw,_,_ = env.get_raw()
        s_m, s_v, r_m, r_v = env.get_regu()


        state_f = env.get_state_latten()

        for t in range(self.T):
            s_old = s
            s_old_raw = s_raw
            state_f_old = state_f
            s_m_old, s_v_old, r_m_old, r_v_old = s_m, s_v, r_m, r_v

            a = actor.sample(s_old,clip=False)
            neglogp = actor.neglogp(s_old,a)

            s, r, d, _ = env.step(actor.clip(a))
            s_raw, r_raw, r_raw_true = env.get_raw()

            state_f = env.get_state_latten()

            S_M.append(s_m)
            s_m, s_v, r_m, r_v = env.get_regu()



            if t == (self.T-1):
                d = False

            # Store
            s_traj.append(s_old)
            s_raw_traj.append(s_old_raw)
            a_traj.append(a.numpy())
            r_traj.append(r)
            r_raw_traj.append(r_raw)
            r_raw_true_traj.append(r_raw_true)
            sp_traj.append(s)
            d_traj.append(d)
            neglogp_traj.append(neglogp.numpy())

            state_f_traj.append(state_f_old)
            s_reg_m_traj.append(s_m_old)
            s_reg_v_traj.append(s_v_old)
            r_reg_m_traj.append(r_m_old)
            r_reg_v_traj.append(r_v_old)

            self.steps_batch += 1
            if self.steps_batch >= self.b_size:
                if t < (self.T-1):
                    full = False
                break
            
            if d:
                break

        s_traj = np.array(s_traj)
        s_raw_traj = np.array(s_raw_traj)
        a_traj = np.array(a_traj)
        r_traj = np.array(r_traj)
        r_raw_traj = np.array(r_raw_traj)
        r_raw_true_traj = np.array(r_raw_true_traj)
        sp_traj = np.array(sp_traj)
        d_traj = np.array(d_traj)
        neglogp_traj = np.array(neglogp_traj)
        k_traj = np.ones_like(r_traj,dtype='int') * self.k

        s_reg_m_traj = np.array(s_reg_m_traj)
        s_reg_v_traj = np.array(s_reg_v_traj)
        r_reg_m_traj = np.array(r_reg_m_traj)
        r_reg_v_traj = np.array(r_reg_v_traj)



        rtg_raw_traj, J_tot, J_disc = reward_calc(
            r_raw_traj,r_raw_true_traj,self.gamma)

        return (s_traj, s_raw_traj, a_traj, r_traj, sp_traj, d_traj, 
            neglogp_traj, k_traj, rtg_raw_traj, J_tot, J_disc, full,
                state_f_traj, s_reg_m_traj, s_reg_v_traj, r_reg_m_traj, r_reg_v_traj)

    def generate_batch(self,env,actor):
        """Generates batch of trajectories"""

        traj_batch = 0
        self.steps_batch = 0
        criteria = 0

        while criteria < self.b_size:
            res = self._generate_traj(env,actor)
            
            (s_traj, s_raw_traj, a_traj, r_traj, sp_traj, d_traj, 
                neglogp_traj, k_traj, rtg_raw_traj, J_tot, J_disc, full,
             state_f, s_reg_m_traj, s_reg_v_traj, r_reg_m_traj, r_reg_v_traj) = res

            # Store
            self.s_batch.append(s_traj)
            self.s_raw_batch.append(s_raw_traj)
            self.a_batch.append(a_traj)
            self.r_batch.append(r_traj)
            self.sp_batch.append(sp_traj)
            self.d_batch.append(d_traj)
            self.neglogp_batch.append(neglogp_traj)
            self.k_batch.append(k_traj)

            self.state_f_batch.append(state_f)
            self.s_reg_m_batch.append(s_reg_m_traj)
            self.s_reg_v_batch.append(s_reg_v_traj)
            self.r_reg_m_batch .append(r_reg_m_traj)
            self.r_reg_v_batch.append(r_reg_v_traj)

            if full:
                self.rtg_raw_batch.append(rtg_raw_traj)
                self.J_tot_batch.append(J_tot)
                self.J_disc_batch.append(J_disc)

            traj_batch += 1
            criteria = self.steps_batch
        
        self.traj_total += traj_batch
        self.steps_total += self.steps_batch
    
    def get_log_info(self):
        J_tot_ave = np.mean(self.J_tot_batch)
        J_disc_ave = np.mean(self.J_disc_batch)
        log_info = {
            'J_tot':    J_tot_ave,
            'J_disc':   J_disc_ave,
            'traj':     self.traj_total,
            'steps':    self.steps_total
        }
        return log_info

    def get_update_info(self,actor,critic):
        if (self.noldpols > 0) and (len(self.s_buffer) > 0):
            s_all = [self.s_batch] + self.s_buffer
            a_all = [self.a_batch] + self.a_buffer
            neglogp_all = [self.neglogp_batch] + self.neglogp_buffer

            k_all = [self.k_batch] + self.k_buffer
            M_active = len(self.s_buffer) + 1

            weights_active = self.weights[:M_active]
            weights_active = weights_active / np.sum(weights_active)
            weights_active = weights_active * M_active


            r_all = [self.r_batch] + self.r_buffer
            sp_all = [self.sp_batch] + self.sp_buffer
            d_all = [self.d_batch] + self.d_buffer

            state_f = [self.state_f_batch] + self.state_f_buffer
            s_reg_m = [self.s_reg_m_batch] + self.s_reg_m_buffer
            s_reg_v = [self.s_reg_v_batch] + self.s_reg_v_buffer
            r_reg_m = [self.r_reg_m_batch] + self.r_reg_m_buffer
            r_reg_v = [self.r_reg_v_batch] + self.r_reg_v_buffer

            adv_all, rtg_all = gae_all(s_all,a_all,sp_all,r_all,d_all,
                neglogp_all,self.gamma,self.lam,True,self.is_trunc,actor,critic)

        else:
            s_all = [self.s_batch]
            a_all = [self.a_batch]
            neglogp_all = [self.neglogp_batch]

            k_all = [self.k_batch]
            weights_active = np.array([1.])

            r_all = [self.r_batch]
            sp_all = [self.sp_batch]
            d_all = [self.d_batch]

            state_f = [self.state_f_batch]
            s_reg_m = [self.s_reg_m_batch]
            s_reg_v = [self.s_reg_v_batch]
            r_reg_m = [self.r_reg_m_batch]
            r_reg_v = [self.r_reg_v_batch]

            # Uses GAE: no old data
            adv_all, rtg_all = gae_all(s_all,a_all,sp_all,r_all,d_all,
                neglogp_all,self.gamma,self.lam,False,self.is_trunc,actor,critic)


        s_all = aggregate_data(s_all)
        a_all = aggregate_data(a_all)
        adv_all = aggregate_data(adv_all)
        d_all = aggregate_data(d_all)
        rtg_all = aggregate_data(rtg_all)
        neglogp_all = aggregate_data(neglogp_all)

        state_f = aggregate_data(state_f)
        s_reg_m = aggregate_data(s_reg_m)
        s_reg_v = aggregate_data(s_reg_v)
        r_reg_m = aggregate_data(r_reg_m)
        r_reg_v = aggregate_data(r_reg_v)

        k_all = self.k - aggregate_data(k_all)


        weights_all = weights_active[k_all]


        return s_all, a_all, adv_all, rtg_all, neglogp_all, weights_all, state_f, s_reg_m, s_reg_v, r_reg_m, r_reg_v, d_all

    def get_env_info(self):
        s_raw = aggregate_data([self.s_raw_batch])
        rtg_raw = aggregate_data([self.rtg_raw_batch])

        return s_raw, rtg_raw


