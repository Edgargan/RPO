import numpy as np
import tensorflow as tf
import random
import os
import gym
import pickle

from rpo.common.env_wrapper import NormEnv
from rpo.common.actor import GaussianActor
from rpo.common.critic import Critic

def init_seeds(seed,env=None):
    """Sets random seed"""
    seed = int(seed)
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)

def init_env(env_name,s_normalize,r_normalize,r_shift,
    s_t,s_mean,s_var,r_t,r_mean,r_var):
    """Creates environment with NormEnv wrapper"""
    env_raw = gym.make(env_name)
    env = NormEnv(env_raw,s_normalize,r_normalize,r_shift)
    env.set_rms(s_t,s_mean,s_var,r_t,r_mean,r_var)
    
    return env

def init_actor(env,layers,activations,gain,std_mult,actor_weights):
    """Initializes actor"""
    actor = GaussianActor(env,layers,activations,gain,std_mult)

    if actor_weights is not None:
        actor.set_weights(actor_weights)
    
    return actor

def init_critic(env,vf_layers,vf_activations,vf_gain,
    r_min,r_max,gamma,critic_weights):
    """Initializes critic"""
    
    r_range = np.array([r_min,r_max],dtype=np.float32)
    vf_range = r_range / (1-gamma)
    
    critic = Critic(env,vf_layers,vf_activations,vf_gain,vf_range)

    if critic_weights is not None:
        critic.set_weights(critic_weights)
    
    return critic

def import_params(import_path,import_file,import_idx):
    """Imports parameter info from previous simulations"""
    import_filefull = os.path.join(import_path,import_file)
    with open(import_filefull,'rb') as f:
        import_data = pickle.load(f)

    if isinstance(import_data,list):
        assert import_idx < len(import_data), 'import_idx too large'
        import_final = import_data[import_idx]['final']
        import_params = import_data[import_idx]['param']
    elif isinstance(import_data,dict):
        import_final = import_data['final']
        import_params = import_data['param']
    else:
        raise TypeError('imported data not a list or dict')
    
    imported = dict()
    
    # Environment info
    imported['env_name'] = import_params['env_name']
    imported['s_normalize'] = import_params['s_normalize']
    imported['r_normalize'] = import_params['r_normalize']
    imported['r_shift'] = import_params['r_shift']
    imported['s_t'] = import_final['s_t']
    imported['s_mean'] = import_final['s_mean']
    imported['s_var'] = import_final['s_var']
    imported['r_t'] = import_final['r_t']
    imported['r_mean'] = import_final['r_mean']
    imported['r_var'] = import_final['r_var']

    # Actor info
    imported['actor_layers'] = import_params['actor_layers']
    imported['actor_activations'] = import_params['actor_activations']
    imported['actor_gain'] = import_params['actor_gain']
    imported['actor_std_mult'] = import_params['actor_std_mult']
    imported['actor_weights'] = import_final['actor_weights']

    # Critic info
    imported['vf_layers'] = import_params['vf_layers']
    imported['vf_activations'] = import_params['vf_activations']
    imported['vf_gain'] = import_params['vf_gain']
    imported['r_min'] = import_params['r_min']
    imported['r_max'] = import_params['r_max']
    imported['gamma'] = import_params['gamma']
    imported['critic_weights'] = import_final['critic_weights']

    return imported