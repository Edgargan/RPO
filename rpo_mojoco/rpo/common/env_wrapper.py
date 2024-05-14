import numpy as np

class RunningNormalizer:
    """Tracks running statistics to use for normalization"""

    def __init__(self,dim):
        self.dim = dim
        self.t_last = 0

        if dim == 1:
            self.mean = 0.0
            self.var = 0.0
            self.std = 1.0
        else:            
            self.mean = np.zeros(dim,dtype=np.float32)
            self.var = np.zeros(dim,dtype=np.float32)
            self.std = np.ones(dim,dtype=np.float32)
    
    def normalize(self,data,center=True,clip=10.):
        if center:
            stand = (data - self.mean) / np.maximum(self.std,1e-8)
        else:
            stand = data / np.maximum(self.std,1e-8)
        
        return np.clip(stand,-clip,clip)

    def denormalize(self,data_norm,center=True):
        if center:
            data = data_norm * np.maximum(self.std,1e-8) + self.mean
        else:
            data = data_norm * np.maximum(self.std,1e-8)
        
        return data

    def update(self,data):
        t_batch = data.shape[0]
        M_batch = data.mean(axis=0)
        S_batch = np.sum(np.square(data - M_batch), axis=0)

        t = t_batch + self.t_last

        self.var = ((S_batch + self.var * np.maximum(1,self.t_last-1)  
            + (t_batch / t) * self.t_last * np.square(M_batch-self.mean)) 
            / np.maximum(1,t-1))

        self.mean = (t_batch * M_batch + self.t_last * self.mean) / t

        self.mean = self.mean.astype('float32')
        self.var = self.var.astype('float32')

        if t==1:
            self.std = np.abs(self.mean)
        else:
            self.std = np.sqrt(self.var)

        self.t_last = t
    
    def reset(self):
        self.t_last = 0

        if self.dim == 1:
            self.mean = 0.0
            self.var = 0.0
            self.std = 1.0
        else:            
            self.mean = np.zeros(self.dim,dtype=np.float32)
            self.var = np.zeros(self.dim,dtype=np.float32)
            self.std = np.ones(self.dim,dtype=np.float32)

    def instantiate(self,t,mean,var):
        self.t_last = t
        self.mean = mean
        self.var = var
        if self.t_last==0:
            self.reset()
        elif self.t_last==1:
            self.std = np.abs(self.mean)
        else:
            self.std = np.sqrt(self.var)

class NormEnv:
    """
    Environment wrapper that handles normalization of observations and rewards.

    Attributes:
        env: Gym environment
        observation_space: observation space of Gym environment
        action_space: action space of Gym environment
        s_normalize (bool): normalize observations if true
        r_normalize (bool): normalize rewards if true
        r_shift (float): decrease raw rewards by constant value
        s_rms (RunningNormalizer): running filter for observations
        r_rms (RunningNormalizer): running filter for rewards
        s_raw (np.ndarray): most recent unnormalized observation
        r_raw (float): most recent unnormalized reward
        r_raw_true (float): most recent unnormalized true reward
    """

    def __init__(self,env,s_normalize,r_normalize,r_shift=False):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.s_normalize = s_normalize
        self.r_normalize = r_normalize

        if r_shift:
            try:
                self.r_shift = self.env.healthy_reward
            except:
                self.r_shift = 0.0
        else:
            self.r_shift = 0.0

        self.s_rms = RunningNormalizer(self.observation_space.shape[0])
        self.r_rms = RunningNormalizer(1)
    
    def step(self,a):
        s_raw, r_raw_true, d, info = self.env.step(a)
        
        s_norm = self.s_rms.normalize(s_raw)

        r_raw_adj = r_raw_true - self.r_shift
        r_norm = self.r_rms.normalize(r_raw_adj,center=False)

        self.s_raw = s_raw
        self.r_raw = r_raw_adj
        self.r_raw_true = r_raw_true

        return s_norm, r_norm, d, info
    
    def reset(self):      
        s_raw = self.env.reset()
        s_norm = self.s_rms.normalize(s_raw)

        self.s_raw = s_raw
        self.r_raw = None
        self.r_raw_true = None

        return s_norm

    def seed(self,seed):
        self.env.seed(seed)

    def get_raw(self):
        return self.s_raw, self.r_raw, self.r_raw_true

    def update_rms(self,s_data,r_data):
        if self.s_normalize:
            self.s_rms.update(s_data)
        if self.r_normalize:
            self.r_rms.update(r_data)

    def reset_rms(self):
        self.s_rms = RunningNormalizer(self.observation_space.shape[0])
        self.r_rms = RunningNormalizer(1)

    def set_rms(self,s_t,s_mean,s_var,r_t,r_mean,r_var):
        if self.s_normalize and s_t:
            assert self.s_rms.mean.shape == s_mean.shape, 's_mean shape incorrect'
            assert self.s_rms.var.shape == s_var.shape, 's_var shape incorrect'
            self.s_rms.instantiate(s_t,s_mean,s_var)
        if self.r_normalize and r_t:
            self.r_rms.instantiate(r_t,r_mean,r_var)

    def set_rms1(self,s_mean,s_var,r_mean,r_var):
        self.s_rms.mean, self.r_rms.mean = s_mean,r_mean
        self.s_rms.var, self.r_rms.var = s_var,r_var
        self.s_rms.std, self.r_rms.std = np.sqrt(s_var), np.sqrt(r_var)

    def get_regu(self):
        return self.s_rms.mean, self.s_rms.var, self.r_rms.mean, self.r_rms.var


    def get_state_latten(self):
        return self.env.env.sim.get_state().flatten()

    def state_from_flattened(self,s):
        # print('1111', s)
        self.env.env.sim.set_state_from_flattened(s)
        self.env.env.sim.forward()
