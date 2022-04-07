from msilib.schema import Verb
from tabnanny import verbose
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from pingouin import multivariate_normality
import numpy as np
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from download_prices import setseed
setseed()
from keras import Input


data = make_moons(3000, noise=0.05)[0].astype("float32")
norm = layers.Normalization()
norm.adapt(data)
normalized_data = norm(data)
# Creating a custom layer with keras API.
output_dim = 256
reg = 0.01


def Coupling(input_shape):
    input = keras.layers.Input(shape=input_shape)

    t_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    t_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_1)
    t_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_2)
    t_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_3)
    t_layer_5 = keras.layers.Dense(
        input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_4)

    s_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    s_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_1)
    s_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_2)
    s_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_3)
    s_layer_5 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_4)

    return keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])
class RealNVP(keras.Model):
    def __init__(self, num_coupling_layers):
        super(RealNVP, self).__init__()

        self.num_coupling_layers = num_coupling_layers

        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0, 0.0], scale_diag=[1.0, 1.0]
        )
        self.masks = np.array(
            [[0, 1], [1, 0]] * (num_coupling_layers // 2), dtype="float32"
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(2) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, x, training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
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
        print("train step")
        with tf.GradientTape() as tape:

            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        print("test_step")
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}
    def build_graph(self):
        x = Input(shape=(2))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

model = RealNVP(num_coupling_layers=2)
model.build((None, 2))
model.build_graph().summary()
tf.keras.utils.plot_model(model.build_graph(),                      # here is the trick (for now)
                                to_file='model_twomoon.png', dpi=96,              # saving  
                                show_shapes=True, show_layer_names=True,  # show shapes and layer name
                                expand_nested=False                       # will show nested block
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

history = model.fit(
    normalized_data, batch_size=256, epochs=200, verbose=1, validation_split=0.2
)
# From data to latent space.
z, trainloss = model(normalized_data)
loss_train = model.log_loss(normalized_data)
# From latent space to data.
samples = model.distribution.sample(3000)

x, test_loss = model(samples, training=False)
loss_test = model.log_loss(x)
print("loss test log_loss \n", loss_test)
print("loss withe evaluation", model.evaluate(samples, verbose=0))
from keras import backend as K

f, axes = plt.subplots(2, 2)
f.set_size_inches(20, 15)

axes[0, 0].scatter(normalized_data[:, 0], normalized_data[:, 1], color="r")
axes[0, 0].set(title="Inference data space X", xlabel="x", ylabel="y")
axes[0, 1].scatter(z[:, 0], z[:, 1], color="r")
axes[0, 1].set(title="Inference latent space Z", xlabel="x", ylabel="y")
axes[0, 1].set_xlim([-3.5, 4])
axes[0, 1].set_ylim([-4, 4])
axes[1, 0].scatter(samples[:, 0], samples[:, 1], color="g")
axes[1, 0].set(title="Generated latent space Z", xlabel="x", ylabel="y")
axes[1, 1].scatter(x[:, 0], x[:, 1], color="g")
axes[1, 1].set(title="Generated data space X", label="x", ylabel="y")
axes[1, 1].set_xlim([-2, 2])
axes[1, 1].set_ylim([-2, 2])

del model
K.clear_session()
tf.compat.v1.reset_default_graph()
plt.show()
# model.evaluate(samples)
# print(multivariate_normality(z, alpha=.05))

# grdevices = importr('grDevices')
# packages_to_install_if_needed = ("MVN",)
# utils = rpackages.importr("utils")
# utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
# names_to_install = [x for x in packages_to_install_if_needed if not rpackages.isinstalled(x)]
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))
# # load the package
# mvn = importr("MVN")
# numpy2ri.activate()
# grdevices.png(file="C:/Users/Sezen/MA_THESIS/plots/qq.png", width=512, height=512)
# res_mard = mvn.mvn(z.numpy(), mvnTest = "mardia", multivariatePlot = "qq")
# grdevices.dev_off()
# print(res_mard)
# res_hz = mvn.mvn(z.numpy(), mvnTest = "hz", multivariatePlot = "qq")
# # res_roy = mvn.mvn(z.numpy(), mvnTest = "royston", multivariatePlot = "qq")
# res_dh = mvn.mvn(z.numpy(), mvnTest = "dh", multivariatePlot = "qq")

# plt.show()
# print(res_mard, "\n", res_hz, "\n", res_dh)