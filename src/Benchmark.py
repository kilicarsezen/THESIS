import scipy.interpolate as interpolate
import statsmodels.formula.api as smf
from functools import partial, update_wrapper
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow import subtract, reduce_mean, maximum
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf
tfk = tf.keras
from scipy.stats import multivariate_normal, norm
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from keras.utils.vis_utils import plot_model

def uncond_benchmark(data):
    mu = data.mean()
    cov = data.cov()
    unc_bench = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
    return unc_bench


def cond_benchmark(input_shape, output_shape, n_hidden):
    """
    Gaussian Regression model with two dense layers and an output layer.
    Output layer returns a mulviariate normal distribution

    input_shape : number of features in the exogenous data
    output_shape : number of dimension of the target density
    """
    neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
    model_cond_bench = tfk.Sequential([
        tfk.layers.InputLayer(input_shape=(input_shape,), name="input"),
        tfk.layers.Dense(6, activation="tanh", name="dense_1"),
        tfk.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(output_shape),activation="relu", name='dense_3'),
        tfp.layers.MultivariateNormalTriL(output_shape, name="output")])
    plot_model(model_cond_bench, to_file="C:/Users/Sezen/MA_THESIS/src/my_model.png", show_shapes=True, show_layer_names=True,
                 rankdir='LR', dpi=96, show_layer_activations=False) #plot network structure
    return model_cond_bench, neg_log_likelihood

 #Copula models are adapted from the master thesis "Scenario forecasts of renewable power generation 
 # using Normalizing Flows for day-ahead operation of energy systems under uncertainty" (Author Leonar Paeleke)
class Copulas_lqr:
    """
    Gaussian copula with linear quantile regression (LQR) class
    """

    def __init__(self):
        """
        initialization of copulas class with linear quantile regression for the inverse cdf
        """
        self._dim = None
        self._con_dim = None
        self.corr = None
        self._y_min = None
        self._y_max = None
        self.mvnorm = None
        self.mod = None
        self.models = []
        self.inv_cdf = []
        self.quantiles = np.arange(0.01, 1.0, 0.01) #quantiles to predict
        self.__x_vals =np.arange(0.01, 1.0, 0.01) #x axis values for the inverse cdf function, same as quantiles as I 
                                                    #extrapolate outside of the range 0.01 and 0.99
    def fit(self, X, y):
        """
        train copula function on data
        :param X : array-like, shape (n_observations, n_features)
            Training vector(exogenous variables), where n_observations is the number of data points in the train set and
            n_features is the number of exogenous features. 
        :param y : array-like, shape (n_observations, dim )
            Target vector(price deltas) relative to X, where dim is the dimension of the target, in our case it is 4.
        """
        # dimensionality of data and samples
        self._dim = y.shape[1]
        self._price_cols = y.columns
        self._con_dim = X.shape[1]
        # explain stochastic dependence structure of the original data by its correlation matrix
        self.corr = y.corr()
        # get min and max values in data
        self._y_min = y.min()
        self._y_max = y.max()
        # prepare multivariate normal distribution to sample from
        self.mvnorm = multivariate_normal(mean=np.zeros(self.dim), cov=self.corr)

        # create formula used to build linear quantile regression model
        for q, quarter in enumerate(self._price_cols):
            formula = quarter + '~'
            for col in X.columns[:-1]:
                formula += col + '+'
            formula += X.columns[-1]
            print(formula)
            self.mod = smf.quantreg(formula, pd.concat([X, y[quarter]], axis=1))
            self.models.append([self._fit_model(q) for q in self.quantiles])

    def sample(self, n, X):
        """
        generate samples
        :param n: number of samples to generate
        :param X: array-like, exogene information e.g. forecast error, shape (n_observations_test, n_features), where n_observations_test is the number of
            data points in the test set 
        :return: n samples
        """
        # generate n multivariate normal distribution samples
        x = self.mvnorm.rvs(n * X.shape[0])
        # transform drawn samples into uniform distributed samples
        x_uniform = norm.cdf(x)
        samples = np.zeros(x_uniform.shape)
        # create inverse cdf from linear quantile regression and exogen information X
        for quarter, quarter_name in enumerate(self._price_cols):
            print("sampling for {}".format(quarter_name))
            inv_cdf = self._inv_cdf(X, quarter) 
            for i, interpolation in enumerate(inv_cdf):
                samples[n * i:(n * (i + 1)), quarter] = interpolation(x_uniform[n * i:(n * (i + 1)), quarter])
        return samples

    def _inv_cdf(self, data, timestep):
        """
        fit inverse cdf function by linearly interpolating the quantile values for exog informations in data
        :param data: pandas dataframe containing the conditioning information
        :param timestep: indicates the dimension of prediction, e.g. 1st quarter, 2nd quarter ..
        :return: scipy interpolate1d object
        """
        y_hat = [(self.models[timestep][i].predict(data)) for i, _ in enumerate(self.quantiles)]
        y_hat = np.array(y_hat).transpose(1, 0)
        y_vals = y_hat
        # interpolate between predictions, extrapolate on the tails
        return np.apply_along_axis(self.interpolation, 1, y_vals)

    def interpolation(self, y_vals):
        x_vals = self._x_vals
        return interpolate.interp1d(x=x_vals, y=y_vals, fill_value='extrapolate')

    def _fit_model(self, q):
        """
        fit model instance to a specific quantile
        :param q: quantile
        :return:
        """
        res = self.mod.fit(q=q, p_tol=1e-5)
        return res

    def predict(self, n, X):
        return self.sample(n, X)

    @property
    def dim(self):
        return self._dim

    @property
    def con_dim(self):
        return self._con_dim

    @property
    def _x_vals(self):
        return self.__x_vals

class Copulas_pinball_ANN:
    """
    Gaussian copula with pinball ANN class
    """
    def __init__(self, n_hidden=32, n_epochs=100, n_batch=64):
        """
        initialization of copulas class
        """
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self._dim = None
        self.corr = None
        self._y_min = None
        self._y_max = None
        self.mvnorm = None
        self.inv_cdf = []
        self.quantiles = np.arange(0.01, 1.0, 0.01)
        self.__x_vals =  np.arange(0.01, 1.0, 0.01)

    def fit(self, X, y):
        """
        train copula function on data
        :param X: exogenous data, shape (n_observations, n_features)
        :paran y: target data, shape (n_observations, n_dim)
        """
        # dimensionality of data and samples
        self._dim = y.shape[1]
        # explain stochastic dependence structure of the original data by its correlation matrix
        self.corr = y.corr()
        # get min and max values in target data
        self._y_min = y.min().min()
        self._y_max = y.max().max()
        # prepare multivariate normal distribution to sample from
        self.mvnorm = multivariate_normal(mean=np.zeros(self.dim), cov=self.corr)

        # train model for inv cdf
        self.model = PinballKerasRegressor(build_fn=create_pinball_ann,
                                      input_dim=X.shape[1],
                                      quantiles=self.quantiles,
                                      hidden_neurons=self.n_hidden, nb_epoch=self.n_epochs, batch_size=self.n_batch, verbose=2)
        self.model.fit(X, y)

    def sample(self, n, X):
        """
        generate samples
        :param n: number of samples to generate
        :return: n samples
        """
        # generate n multivariate normal distribution samples of shape (num_test_days, num_scenarios, num_data_dimension) (365, 20, 24)

        x = self.mvnorm.rvs((X.shape[0], n))
        # transform drawn samples into uniform distibuted samples
        x_uniform = norm.cdf(x)

        samples = np.zeros(x_uniform.shape)
        x_uniform_swapped = np.swapaxes(x_uniform, -1, -2)

        # calculate inv cdf
        x_vals = self._x_vals
        y_hat = self.model.predict(X) # shape (num_test_days, num_dimension, quantiles) (365, 24, 19)
        y_vals= y_hat
        for i in range(len(X)):
            inv_cdf = interpolate.interp1d(x_vals, y_vals[i, :, :], fill_value='extrapolate') # inverse cdf of a day
            samples[i] = inv_cdf(x_uniform_swapped[i, :, :])[range(self.dim), range(self.dim)].T # generated samples of a day
        return samples

    def predict(self, n, X):
        return self.sample(n, X)
    
    @property
    def dim(self):
        return self._dim
    @property
    def _x_vals(self):
        return self.__x_vals


def _pinball_loss(y_true, y_pred, q):
    ''' Pinball loss for Tensorflow Backend '''
    error = subtract(y_true, y_pred)
    return reduce_mean(maximum(q * error, (q - 1) * error), axis=-1)


def _wrapped_partial(func, *args, **kwargs):
    # Needed as discussed here:
    # http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def create_pinball_ann(quantiles, input_dim, hidden_neurons):
    ''' Pinball loss minimizing ANN model '''
    inputs = Input(shape=(input_dim,), name='Input')
    x = Dense(32, activation='relu', name="Hidden1")(inputs)
    x = Dense(32, activation='relu', name="Hidden2")(x)
    outputs = []
    losses = []
    for q in quantiles:
        outputs.append(Dense(1, name="Output_q_{}".format(q))(x))
        losses.append(_wrapped_partial(_pinball_loss, q=q))
    model = Model(inputs=inputs, outputs=outputs, name="QuantileANN")
    model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    return model


class PinballKerasRegressor():
    '''  Wrapper following the sklearn interface '''

    def __init__(self, **args):
        self.args = args
        self.quantiles = args["quantiles"]
        self.models = []

    def fit(self, X, y):
        self.H = y.shape[1]
        for i in range(self.H):
            model = KerasRegressor(**self.args)
            model.fit(X, [y.iloc[:, i].to_numpy().flatten(), ] * len(self.quantiles), batch_size=32, validation_split=0.2)
            self.models.append(model)

    def predict(self, X):
        y_hat = np.zeros((X.shape[0], self.H, len(self.quantiles)))
        for i in range(self.H):
            y_hat[:, i, :] = np.array(self.models[i].predict(X))[:, :].T
        return y_hat