from keras import Input
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class BatchNorm():
    def __init__(self, eps, momentum, dim, name="batch_norm"):
        self.eps = eps
        self.momentum = momentum
        self.gamma = tf.Variable(tf.constant(1.0, shape=[1, dim]), trainable=True)
        self.beta = tf.Variable(tf.constant(0.0, shape=[1, dim]), trainable=True)
        self.global_mean = tf.Variable(tf.constant(0.0, shape=[1, dim]), trainable=False)
        self.global_var = tf.Variable(tf.constant(1.0, shape=[1, dim]), trainable=False)

    def forward(self, x):
        return (x - tf.exp(self.beta)) * tf.exp(-self.gamma) * tf.sqrt(self.global_var + self.eps) + self.global_mean

    def inverse(self, x):
        batch_mean, batch_var = tf.nn.moments(x, axes=[0], keepdims=True)
        global_mean_updated = tf.compat.v1.assign_add(self.global_mean, self.momentum * (self.global_mean - batch_mean))
        global_var_updated = tf.compat.v1.assign_add(self.global_var, self.momentum * (self.global_var - batch_var))
        with tf.control_dependencies([global_mean_updated, global_var_updated]):
            return ((x - batch_mean) * 1. / tf.sqrt(batch_var + self.eps)) * tf.exp(self.gamma) + tf.exp(self.beta)

    def inverse_log_det(self, x):
        _, batch_var = tf.nn.moments(x, axes=[0], keepdims=True)
        total_log_gamma = tf.reduce_sum(tf.math.log(self.gamma))
        inv_log_det = total_log_gamma - 0.5 * tf.reduce_sum(tf.math.log(batch_var + self.eps))
        return inv_log_det


class NN(layers.Layer):
    def __init__(self, n_out, n_hidden, n_layer):
        super().__init__()
        self.s_layer = []
        self.t_layer = []
        for i in range(n_layer):
            self.s_layer.append(layers.Dense(n_hidden, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(0.1), name='Dense_s_{}'.format(i)))
            self.t_layer.append(layers.Dense(n_hidden, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1),name='Dense_t_{}'.format(i)))
        self.s_layer.append(layers.Dense(n_out, activation="tanh", name='Output_s'))
        self.t_layer.append(layers.Dense(n_out, activation="linear", name='Output_t'))

    def call(self, x, *args, **kwargs):
        if args[1]==True:
            input = tf.concat([x, args[0]], axis=1)
        else:
            input = x
        s = input
        t = input
        for s_layer in self.s_layer:
            s = s_layer(s)
        for t_layer in self.t_layer:
            t = t_layer(t)
        return [s, t]


class RealNVP(tf.keras.Model):
    """
    realNVP function implementation based on Dinh explanantion https://arxiv.org/pdf/1605.08803.pdf
    adapted from: https://keras.io/examples/generative/real_nvp/
    """
    def __init__(self, num_coupling_layers, dim, conditional, n_hidden, n_layer, batchnorm=False):
        super(RealNVP, self).__init__()
        self.num_coupling_layers = num_coupling_layers #number of coupling layers
        self.conditional = conditional  # True if conditional distribution
        self.dim = dim #number of dimension of the data distribution 
        self.batchnorm = batchnorm
        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(loc=[0.0] * dim, scale_diag=[1.0] * dim)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.layers_list = [NN(n_out=self.dim, n_hidden=n_hidden, n_layer=n_layer) for i in
                            range(num_coupling_layers)]
        self.batch_norm_list = [BatchNorm(eps=1e-5, momentum=0.05, dim=self.dim) for i in range(num_coupling_layers)]

    def mask(self):
        mask = np.zeros((self.num_coupling_layers, self.dim), dtype='float32')
        mask[1::2, ::2] = 1
        mask[::2, 1::2] = 1
        return mask

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, x, training=True, mask=None, *args):
        """ the call function takes the full data set and splits the distribution variables and
        conditionning variables before applying mask. Data x has to have the disribution variables
        in the first self.dim columns"""
        y = x[:, self.dim:]
        x = x[:, 0:self.dim]
        log_det_inv = 0
        direction = 1
        mask = self.mask()
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = tf.math.multiply(x, mask[i])
            reversed_mask = 1 - mask[i]
            s, t = self.layers_list[i](x_masked, y, self.conditional) 
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                    reversed_mask
                    * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                    + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])
        return x, log_det_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)
            loss = loss + tf.compat.v1.losses.get_regularization_loss()

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def build_graph(self):
        x = Input(shape=(19))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))