import numpy as np
import tensorflow as tf

def create_activations(layers,activations_in):
    """Creates list of Tensorflow activations from string inputs"""
    if len(layers) > 1 and len(activations_in) == 1:
        activations_in = [activations_in[0] for layer in layers]
    activations = []
    for act_str in activations_in:
        if act_str == 'tanh':
            act = tf.keras.activations.tanh
        elif act_str == 'relu':
            act = tf.keras.activations.relu
        elif act_str == 'elu':
            act = tf.keras.activations.elu
        else:
            raise ValueError('activations must be tanh, relu or elu')

        activations.append(act)
    
    return activations

def transform_features(s):
    """Updates dtype and shape of inputs to neural networks"""
    # print('s1', s)
    s = tf.cast(s,dtype=tf.float32)
    # print('s2', s)

    if len(s.shape) == 1:
        s = tf.expand_dims(s,axis=0)

    return s

def create_nn(in_dim,out_dim,layers,activations,gain,name=None):
    """Creates neural network"""
    nn = tf.keras.Sequential(name=name)

    activations = create_activations(layers,activations)
    assert len(activations) == len(layers), (
        'activations must be single tf function or list of length len(layers)')
    
    for layer_idx in range(len(layers)):
        if layer_idx == 0:
            nn.add(
                tf.keras.layers.Dense(
                    units=layers[layer_idx],
                    kernel_initializer=tf.keras.initializers.Orthogonal(
                        gain=np.sqrt(2)),
                    activation=activations[layer_idx],
                    name=('hid%d'%layer_idx),
                    input_shape=(in_dim,)
                )
            )
        else:
            nn.add(
                tf.keras.layers.Dense(
                    units=layers[layer_idx],
                    kernel_initializer=tf.keras.initializers.Orthogonal(
                        gain=np.sqrt(2)),
                    activation=activations[layer_idx],
                    name=('hid%d'%layer_idx)
                )
            )

    nn.add(
        tf.keras.layers.Dense(
            units=out_dim,
            kernel_initializer=tf.keras.initializers.Orthogonal(
                gain=gain),
            name='out'
        )
    )

    return nn

def flat_to_list(trainable,weights):
    """Converts flattened array back into list"""
    shapes = [tf.shape(theta).numpy() for theta in trainable ]
    sizes = [tf.size(theta).numpy() for theta in trainable ]
    idxs = np.cumsum([0]+sizes)

    weights_list = []

    for i in range(len(shapes)):
        elem_flat = weights[idxs[i]:idxs[i+1]]
        elem = elem_flat.reshape(shapes[i])
        weights_list.append(elem)
    
    return weights_list

def list_to_flat(weights):
    """Flattens list into array"""
    weights_flat = np.concatenate(list(map(
                lambda y: np.reshape(y,[-1]),weights)),-1)
    
    return weights_flat