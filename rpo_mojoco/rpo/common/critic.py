import tensorflow as tf

from rpo.common.ac_utils import transform_features
from rpo.common.ac_utils import create_nn

class Critic:
    """
    Value function used for bootstrapping and advantage function estimation.

    Attributes:
        vf_range (np.ndarray): range of value function, if known
        trainable (list): list of trainable variables
    """

    def __init__(self,env,vf_layers,vf_activations,vf_gain,vf_range):

        self.vf_range = vf_range
        
        in_dim = env.observation_space.shape[0]
        self._nn = create_nn(in_dim,1,vf_layers,vf_activations,vf_gain,
            name='critic')

        self.trainable = self._nn.trainable_variables

    def value(self,s,predict=True):
        s_feat = transform_features(s)

        values = self._nn(s_feat)

        if predict:
            V = tf.clip_by_value(tf.squeeze(values),
                self.vf_range[0],self.vf_range[1])
        else:
            V = tf.squeeze(values)

        return V

    def get_weights(self):
        return self._nn.get_weights()

    def set_weights(self,weights):
        self._nn.set_weights(weights)