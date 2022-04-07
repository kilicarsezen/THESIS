import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from Benchmark import Copulas_lqr, Copulas_pinball_ANN
from RealNVP import *
from Training import train_realnvp
from evaluation import *
import matplotlib.pyplot as plt
from download_prices import setseed
from pingouin import multivariate_normality
pd.set_option('display.max_columns', 50)
def getnormdata(data):
    norm = layers.experimental.preprocessing.Normalization()
    norm.adapt(data)
    normalized_data = norm(data)
    return normalized_data

def Two_moon_visual(conditioned_train, z, samples,  x, data_y):

    f, axes = plt.subplots(2, 2)
    f.set_size_inches(20, 15)
    colors = {0: 'red', 1: 'blue'}
    for idx, yval in enumerate(np.unique(data_y)):
        axes[0, 0].scatter(x=conditioned_train[data_y == yval, 0], y=conditioned_train[data_y == yval, 1],
                           c=colors[idx], label=yval)
    axes[0, 0].scatter(conditioned_train[:, 0], conditioned_train[:, 1], color="r")
    axes[0, 0].set(title="Inference data space X", xlabel="x", ylabel="y")
    # axes[0, 0].legend(loc='upper right')

    axes[0, 1].scatter(z[:, 0], z[:, 1], color="y")
    axes[0, 1].set(title="Inference latent space Z", xlabel="x", ylabel="y")
    axes[0, 1].set_xlim([-4, 4])
    axes[0, 1].set_ylim([-3.5, 4])

    axes[1, 0].scatter(samples[:, 0], samples[:, 1], color="g")
    axes[1, 0].set(title="Generated latent space Z", xlabel="x", ylabel="y")

    axes[1, 1].scatter(x[:, 0], x[:, 1], color="g")
    axes[1, 1].set(title="Generated data space X", label="x", ylabel="y")
    axes[1, 1].set_xlim([-4, 4])
    axes[1, 1].set_ylim([-2, 2])

    plt.show()

data, label = make_moons(3000, noise=0.05)
normalized_data = getnormdata(data.astype("float32"))

data_cond = np.concatenate((normalized_data, label[:,None]), axis=1)

model, history = train_realnvp(data_realnvp=data_cond, dim=2, num_couples=3, conditional=True, batchnorm=False, lr=0.001, 
                            batch_size=256, n_epochs=200, n_hidden=256, n_layer=6)
# From data to latent space.
z, _ = model(data_cond)

# From latent space to data.
samples = model.distribution.sample(3000)
# subteam = np.array([0] * 1500 + [1] * 1500)
subteam = np.array([0] * 3000 )
np.random.shuffle(subteam)
labels=subteam.reshape(3000,)

data_cond_test = np.concatenate((samples, labels[:,None]), axis=1)
x, _ = model.predict(data_cond_test)

Two_moon_visual(data_cond_test,z, samples, x, labels)

fig = plt.gcf()
fig.set_size_inches(10, 10)
fig.set_dpi(100)
colors = {0: 'red', 1: 'blue'}
for idx, yval in enumerate(np.unique(labels)):
    plt.scatter(x=x[labels == yval, 0], y=x[labels == yval, 1],
                    c=colors[idx], label=yval)
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
fig.tight_layout()
plt.show()