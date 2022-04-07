from pickle import TRUE
from RealNVP import *
from Benchmark import Copulas_lqr, cond_benchmark, uncond_benchmark, Copulas_pinball_ANN
from tensorflow import keras
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from download_prices import setseed
from keras import backend as K

def train_realnvp(data_realnvp, num_couples=2, dim=4, conditional=True, batchnorm=False, lr=0.001, batch_size=128,
                 n_epochs=100, val_split=0.2, n_hidden=2, n_layer=3):
    print("training realnvp")
    model_realnvp = RealNVP(num_coupling_layers=num_couples, dim=dim, conditional=conditional, batchnorm=batchnorm,
                    n_hidden=n_hidden, n_layer=n_layer)
    model_realnvp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    history_realnvp = model_realnvp.fit(data_realnvp, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_split=val_split)
    return model_realnvp, history_realnvp


def get_unconditional_benchmark(data):
    return uncond_benchmark(data)

def train_conditional_benchmark(conditions, target, n_epochs=150, n_hidden=2, val_split=0.2, lr=0.001):
    print("training Gaussian regression")
    model, loss = cond_benchmark(input_shape=conditions.shape[1], output_shape=target.shape[1], n_hidden=n_hidden)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss=loss)
    history = model.fit(conditions, target, epochs=100, validation_split=0.2, verbose=1)

    return model, history


def train_copula_benchmark(conditions, target):
    print("training copula lqr")
    lqr_copula = Copulas_lqr()
    history = lqr_copula.fit(X=conditions, y=target)
    return lqr_copula, history


def train_copula_ann_benchmark(conditions, target, n_hidden, n_epochs, n_batch):
    print("training copula ann")
    ann_copula = Copulas_pinball_ANN(n_hidden=32, n_epochs=100, n_batch=64)
    history = ann_copula.fit(X=conditions, y=target)
    return ann_copula, history

"""generate samples from the trained models, return the sampes, loss on test set and history"""
def test_models(model, test_set, observations, modelName, n):
    if modelName == 'copula LQR': 
        predictions = model.predict(n, test_set)
        log_loss =  None
        test_set_gaussian = None

    elif modelName == 'copula ANN':
        predictions = model.predict(n, test_set)
        predictions = predictions.reshape((test_set.shape[0]*n, 4))
        log_loss = None
        test_set_gaussian = None
        
    elif modelName == 'conditional realnvp' :
        print("test conditional realnvp")
        predictions, _ = model.predict(test_set[modelName])
        test_set_gaussian, _ = model(test_set['full'].values) 
        log_loss = model.evaluate(test_set['full'])

    elif modelName=='unconditional realnvp':
        print("test unconditional realnvp")
        predictions, _ = model.predict(test_set[modelName])
        test_set_gaussian, _ = model(test_set['real'].values) 
        log_loss = model.evaluate(test_set['real'].values)

    elif modelName == 'conditional gaussian':
        predicted_distribution = model(test_set)
        samples_tm = predicted_distribution.sample(n)
        predictions = np.zeros((n*samples_tm.numpy().shape[1], samples_tm.numpy().shape[2]))
        for i in range(samples_tm.numpy().shape[1]):
            predictions[i*n:n*(i+1),:] = samples_tm.numpy()[:,i,:]
        neg_log_likelihood_gaus = lambda x, rv_x: -rv_x.log_prob(x) 
        log_loss =np.mean(neg_log_likelihood_gaus(observations, predicted_distribution))
        test_set_gaussian = None

    elif modelName == 'unconditional gaussian':
        predictions = model.sample(n)
        neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)   
        log_loss = np.mean(neg_log_likelihood(observations.to_numpy(), model))
        test_set_gaussian = None
    return predictions, log_loss, test_set_gaussian


""" Hyperparameter tuning """
logdir="C:/Users/Sezen/MA_THESIS/src/hypertune"
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([2, 8]))
HP_NUM_COUPLES = hp.HParam('num_couples', hp.Discrete([3, 6, 10]))
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-3, 1e-2]))
HP_NUM_LAYER = hp.HParam('n_layer', hp.Discrete([2, 5, 10]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([8, 16, 32, 64]))
HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([50,100]))

HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))

METRIC = 'loss'

with tf.summary.create_file_writer(logdir).as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_NUM_COUPLES, HP_LR, HP_NUM_LAYER, HP_BATCH_SIZE, HP_NUM_EPOCHS, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC, display_name='Log Loss')],
  )

def train_test_model(hparams, data, test_set):
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    setseed()
    model = RealNVP(num_coupling_layers=hparams[HP_NUM_COUPLES], dim=4, conditional=TRUE, batchnorm=False,
                    n_hidden=hparams[HP_NUM_UNITS], n_layer=hparams[HP_NUM_LAYER])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hparams[HP_LR]))
    model.fit(data, batch_size=hparams[HP_BATCH_SIZE], epochs=hparams[HP_NUM_EPOCHS], verbose=1, validation_split=0.2)
    loss_tm = model.evaluate(test_set['full'].values)
    del model
    return loss_tm

def run(run_dir, hparams, data, test):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    logloss = train_test_model(hparams, data, test)
    tf.summary.scalar(METRIC, logloss, step=1)

def hyper_tune(data, test):
    
    
    session_num = 0

    for num_units in HP_NUM_UNITS.domain.values:
        for num_couples in HP_NUM_COUPLES.domain.values:
            for lr in HP_LR.domain.values:
                for num_layers in HP_NUM_LAYER.domain.values:
                    for batch_size in HP_BATCH_SIZE.domain.values:
                        for num_epochs in HP_NUM_EPOCHS.domain.values:
                            for optimizer in HP_OPTIMIZER.domain.values:
                                hparams = {
                                    HP_NUM_UNITS: num_units,
                                    HP_NUM_COUPLES: num_couples,
                                    HP_LR:lr,
                                    HP_NUM_LAYER:num_layers,
                                    HP_BATCH_SIZE:batch_size,
                                    HP_NUM_EPOCHS:num_epochs,
                                    HP_OPTIMIZER: optimizer,
                                }
                                run_name = "/run-%d" % session_num
                                print('--- Starting trial: %s' % run_name)
                                print({h.name: hparams[h] for h in hparams})
                                run(logdir+run_name, hparams, data, test)
                                     
                                session_num += 1