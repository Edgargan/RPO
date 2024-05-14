import gym
import numpy as np
import tensorflow as tf

from rpo.common.ac_utils import transform_features, create_nn 
from rpo.common.ac_utils import flat_to_list, list_to_flat

class GaussianActor:
    """
    Multivariate Gaussian policy with diagonal covariance.

    Attributes:
        act_low (np.ndarray): minimum action values in each dimension
        act_high (np.ndarray): maximum action values in each dimension
        logstd (tf.Variable): log standard deviation variable
        trainable (list): list of trainable variables
    """

    def __init__(self,env,layers,activations,gain,std_mult=1.0):

        assert isinstance(env.action_space,gym.spaces.Box), (
            'Only Box action space supported')
        
        in_dim = env.observation_space.shape[0]
        out_dim = env.action_space.shape[0]
        self._nn = create_nn(in_dim,out_dim,layers,activations,gain,
            name='actor')

        self.act_low = env.action_space.low
        self.act_high = env.action_space.high
        if any(np.isinf(self.act_low)) or any(np.isinf(self.act_high)):
            logstd_init = np.zeros((1,)+env.action_space.shape)
        else:
            logstd_mult = np.log(std_mult*((self.act_high - self.act_low) / 2))
            logstd_init = np.ones((1,)+env.action_space.shape) * logstd_mult

        self.logstd = tf.Variable(logstd_init,dtype=tf.float32,name='logstd')

        self.trainable = self._nn.trainable_variables + [self.logstd]

        self._nn_old = create_nn(in_dim,out_dim,layers,activations,gain,
            name='actor_old')
        self.logstd_old = tf.Variable(logstd_init,dtype=tf.float32,
            name='logstd_old')
        self.update_old_weights()
    
    def sample(self,s,mean=False,clip=True):
        s_feat = transform_features(s)

        a_mean = self._nn(s_feat)

        if mean:
            act = a_mean
        else:
            u = tf.random.normal(tf.shape(a_mean), dtype=a_mean.dtype)
            act = a_mean + tf.exp(self.logstd) * u

        if clip:
            act = tf.clip_by_value(act,self.act_low,self.act_high)

        if act.shape[0] == 1:
            act = tf.squeeze(act,axis=0)

        return act

    def sample_off(self,s,mean=False,clip=True):
        s_feat = transform_features(s)

        a_mean = self._nn(s_feat)           # (1, actor.shape)

        if mean:
            act = a_mean
        else:
            u = tf.random.normal(tf.shape(a_mean), dtype=a_mean.dtype)
            act = a_mean + tf.exp(self.logstd) * u

        if clip:
            act = tf.clip_by_value(act,self.act_low,self.act_high)

        if a_mean.shape[0] == 1:
            a_mean = tf.squeeze(a_mean,axis=0)
        if act.shape[0] == 1:
            act = tf.squeeze(act,axis=0)

        return act, a_mean, tf.squeeze(self.logstd,axis=0)

    def clip(self,a):
        return tf.clip_by_value(a,self.act_low,self.act_high)
    
    def neglogp(self,s,a):
        s_feat = transform_features(s)

        a_mean = self._nn(s_feat)

        a_vec = (tf.square((a - a_mean) / tf.exp(self.logstd)) 
            + 2*self.logstd + tf.math.log(2*np.pi))

        return 0.5 * tf.squeeze(tf.reduce_sum(a_vec,axis=-1))

    def neglogp_old(self,s,a):
        s_feat = transform_features(s)

        a_mean = self._nn_old(s_feat)

        a_vec = (tf.square((a - a_mean) / tf.exp(self.logstd_old)) 
            + 2*self.logstd_old + tf.math.log(2*np.pi))

        return 0.5 * tf.squeeze(tf.reduce_sum(a_vec,axis=-1))

    def get_kl(self, s):
        s_feat = transform_features(s)

        mean = self._nn(s_feat)
        logstd = self.logstd
        std = tf.exp(logstd)

        mean_old_1 = self._nn_old(s_feat)
        logstd_old_1 = self.logstd_old
        std_old_1 = tf.exp(logstd_old_1)
        mean_old = tf.stop_gradient(mean_old_1)
        logstd_old = tf.stop_gradient(logstd_old_1)
        std_old = tf.stop_gradient(std_old_1)

        kl = (-1 / 2 + logstd - logstd_old +
              (tf.square(std_old) + tf.square(mean_old - mean))  / (2 * tf.square(std))
        )
        return tf.reduce_sum(kl, axis=-1, keepdims=True)


    def kl(self,s,mean_old,logstd_old):
        s_feat = transform_features(s)
        a_mean = self._nn(s_feat)

        num = tf.square(a_mean-mean_old) + tf.exp(2*logstd_old)
        vec = num / tf.exp(2*self.logstd) + 2*self.logstd - 2*logstd_old - 1


        return 0.5 * tf.reduce_sum(vec,axis=-1)
    
    def entropy(self):
        vec = 2*self.logstd + tf.math.log(2*np.pi) + 1
        return 0.5 * tf.reduce_sum(vec,axis=-1)

    def get_weights(self,flat=False):
        weights = self._nn.get_weights() + [self.logstd.numpy()]
        if flat:
            weights = list_to_flat(weights)
        
        return weights
    
    def set_weights(self,weights,from_flat=False,increment=False):
        if from_flat:
            weights = flat_to_list(self.trainable,weights)
        
        if increment:
            weights = list(map(lambda x,y: x+y,
                weights,self.get_weights(flat=False)))
        
        model_weights = weights[:-1]
        logstd_weights = weights[-1]
        logstd_weights = np.maximum(logstd_weights,np.log(1e-3))
        
        self._nn.set_weights(model_weights)
        self.logstd.assign(logstd_weights)
    
    def update_old_weights(self):
        model_weights = self._nn.get_weights() 
        logstd_weights = self.logstd.numpy()

        self._nn_old.set_weights(model_weights)
        self.logstd_old.assign(logstd_weights)