from Training import *
from download_prices import get_train_test_sets, ID3_quarterly, setseed
import matplotlib.pyplot as plt
from evaluation import *
import tensorflow_probability as tfp
tfd = tfp.distributions
import pandas as pd
from keras import backend as K



pd.set_option('display.max_columns', 50)


def graph_loss(history, mdl):
    if history==None:
        print("history plot")
    else:
        plt.figure()
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.legend(loc="upper right")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path +'history {}.png'.format(mdl))
"""
Calls the train and test functions for the given models. Returns the generated samples, 
history objects and loss functions on the test set for the log loss trained models,
and the test_inference for the realnvp model.
"""
def run_models(trainsets, test_sets, modelName, n_sample=100, **params):
    train = trainsets[modelName]
    if modelName == 'unconditional realnvp':
        model, history = train_realnvp(train, conditional=False, num_couples=params['n_couples'], dim=train.shape[1], batchnorm=False, 
                                lr=params['lr'], batch_size=params['n_batch'], n_epochs=params['n_epochs'],
                                n_hidden=params['n_hidden'], n_layer=params['n_layers'],val_split=0.3)
        samples, loss, test_inference = test_models(model, test_set=test_sets[modelName], modelName=modelName,
                                    observations=test_sets['real'], n=n_sample)

    elif modelName == 'conditional realnvp':        
        model, history = train_realnvp(train, conditional=True, num_couples=params['n_couples'], dim=test_sets['real'].shape[1],
         batchnorm=False, lr=params['lr'], batch_size=params['n_batch'], n_epochs=params['n_epochs'],
                                n_hidden=params['n_hidden'], n_layer=params['n_layers'], val_split=0.3)
        train_inference, _ = model(trainsets[modelName].values) #pass the train set in the direction of training
        multi_norm_check(pd.DataFrame(train_inference.numpy(), columns=test_sets['real'].columns), 
                        "C:/Users/Sezen/MA_THESIS/",'dh','train_set_qq.png', False)#check if realnvp converts train set into 
                                                                                    #mv normal
        samples, loss, test_inference = test_models(model, test_set=test_sets, modelName=modelName,
                                    observations=test_sets['real'], n=n_sample)

    elif modelName == 'unconditional guassian':
        model = get_unconditional_benchmark(train)
        samples, loss, test_inference = test_models(model, modelName=modelName, n=n_sample)

    elif modelName == 'conditional gaussian':
        model, history = train_conditional_benchmark(train[0], train[1], n_hidden=params['n_hidden'],n_epochs=params['n_epochs'],
        lr=params['lr']) 
        samples, loss, test_inference = test_models(model, test_set=test_sets[modelName].to_numpy(), modelName=modelName, n=n_sample,
                                    observations=test_sets['real'])  
    elif modelName == 'copula LQR':
        model, history = train_copula_benchmark(train[0], train[1])
        samples, loss, test_inference = test_models(model, test_set=test_sets[modelName], modelName=modelName, n=n_sample,
                                    observations=test_sets['real'])
    elif modelName == 'copula ANN':
        model, history = train_copula_ann_benchmark(conditions=train[0], target=train[1], n_hidden=params['n_hidden'],
                                                    n_batch=params['n_batch'], n_epochs=params['n_epochs'])

        samples, loss, test_inference = test_models(model, test_set=test_sets[modelName], modelName=modelName, n=n_sample,
                                    observations=test_sets['real'])
    del model
    return samples, loss, history, test_inference


if __name__ == '__main__':
    setseed()
    data = ID3_quarterly #data to be trained
    n_sample = 500
     #number of samples to generate
    dim = 4 #number of dimension of the target data
    train_sets, test_sets = get_train_test_sets(dataset=data,
                                                to_be_categorized_features=[],
                                                numeric_features = ['Wind_Q1_error', 'Wind_Q2_error', 'Wind_Q3_error', 'Wind_Q4_error',
                            'Solar_Q1_error', 'Solar_Q2_error', 'Solar_Q3_error', 'Solar_Q4_error',
                            'load_Q1_forecast', 'load_Q2_forecast', 'load_Q3_forecast', 'load_Q4_forecast',
                            'Conventional'],
                                                cyclical_features=['Hour'],
                                                n_sample=n_sample) #preprocess the data and get train test set
    parameters ={'n_couples':6,
                'n_layers':2,
                'n_hidden':3,
                'n_epochs':500,
                'n_batch':128,
                'lr':1e-3}   
    # model_names = ['unconditional realnvp','conditional realnvp','conditional gaussian',
    #               'copula LQR','copula ANN'] 
    # hyper_tune(train_sets['conditional realnvp'], test_sets) #tunes the hyperparameters of normalizing flows
    model_names =['conditional realnvp','conditional gaussian','copula LQR','copula ANN'] #models to train
    samples = {} #stores the samples generated by the models
    loss = {} #stores the loss values on test
    history = {} #stores the history objects
    inference_test = {} #stores the price delta transformed into mv gaussian by realnvp
    es = {} #energy score for density
    es_ = {} #energy score for ensemble
    vs = {} #variogram score 
    rmse_ = {} 
    for mdl in model_names:
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        setseed()
        samples[mdl], loss[mdl], history[mdl], inference_test[mdl] = run_models(trainsets=train_sets, test_sets=test_sets, modelName=mdl, 
            n_sample=n_sample, **parameters)
        es[mdl], es_[mdl], vs[mdl], rmse_[mdl] = evaluate_(samples[mdl], test_sets['real'], n_samples=n_sample, 
                                                            model=mdl, dim=dim)
    res_dh = multi_norm_check(pd.DataFrame(inference_test['conditional realnvp'].numpy(), columns=test_sets['real'].columns), 
                        "C:/Users/Sezen/MA_THESIS/",'dh','test_set_qq.png', False)

    # """outliers in the test set will be evaluated seperately"""
    """calculate the mahalanobis distance of the samples in the test set and keep the 100 samples with highest 
    mahalanobis distance in test_sets_ext"""
    mvn_res_test = multi_norm_check(test_sets['real'],"C:/Users/Sezen/MA_THESIS/EDA_plots",'dh', 'noplot', True) 
    sorted_mvn_res_test_3 = sorted(mvn_res_test[3], key=lambda x: x[1], reverse=True)[0:100]
    index_ext = pd.to_datetime([str(x[0]) for x in sorted_mvn_res_test_3]).sort_values()
    test_sets_ext = test_sets['real'].loc[index_ext, :]
    index_ext_loc = [test_sets['real'].index.get_loc(i) for i in index_ext]

    samples_ext = {} # samples drawn for the rare observations
    es_ext = {} #energy score for density of rare obs  
    es_ext_ = {} #energy score for enseble of rare obs
    vs_ext = {} #variogram score for rare obs
    rmse_ext_ = {}
    for mdl in model_names:
        samples_ext[mdl]= np.zeros((len(index_ext)*n_sample, dim))
        for i, idx in enumerate(index_ext_loc):
            samples_ext[mdl][n_sample*i:n_sample*(i+1),:]=samples[mdl][idx*n_sample:n_sample*(idx+1),:]
        es_ext[mdl], es_ext_[mdl], vs_ext[mdl], rmse_ext_[mdl] = evaluate_(samples_ext[mdl],
                                                                    test_sets_ext, n_samples=n_sample, model=mdl, dim=4)
    
    es_and_models_to_plot = []
    vs_and_models_to_plot = []
    es_and_models_to_plot_ext = []
    vs_and_models_to_plot_ext = []
    
    for mdl in model_names:
        print(mdl + "log loss on test {}".format(loss[mdl]))
        print(mdl + " Mean ES {}, ES ensemble {} Variogram {} RMSE {}".format(np.mean(es[mdl]), np.mean(es_[mdl]),
                            np.mean(vs[mdl]), np.mean(rmse_[mdl])))
        print(mdl + " Std ES {}, ES ensemble {} Variogram {} RMSE {}".format(np.std(es[mdl]), np.std(es_[mdl]),
                            np.std(vs[mdl]), np.std(rmse_[mdl])))
        print(mdl + " Outlier Mean ES {}, ES ensemble {} Variogram {} RMSE {}".format(np.mean(es_ext[mdl]),
                            np.mean(es_ext_[mdl]), np.mean(vs_ext[mdl]), np.mean(rmse_ext_[mdl])))
        print(mdl + " Outlier Std ES {}, ES ensemble {} Variogram {} RMSE {}".format(np.std(es_ext[mdl]),
                            np.std(es_ext_[mdl]), np.std(vs_ext[mdl]), np.std(rmse_ext_[mdl])))
        # pairplot_evaluation(test_sets['real'], samples[mdl], mdl, 'test set')
        # pairplot_evaluation(test_sets_ext, samples_ext[mdl], mdl, 'test set outliers')
        rank_hist(observations=test_sets['real'].values, preds=samples[mdl], n_samples=n_sample, 
                    model_name=mdl, num_bins=30)
        graph_loss(history=history[mdl], mdl=mdl)

        es_and_models_to_plot.append(es[mdl]) 
        vs_and_models_to_plot.append(vs[mdl]) 
        es_and_models_to_plot_ext.append(es_ext[mdl]) 
        vs_and_models_to_plot_ext.append(vs_ext[mdl]) 
    # plot_scenarios(test_sets['real'], samples, n_sample=n_sample, models=model_names, outliers='')
    # plot_scenarios(test_sets_ext, samples_ext, n_sample=n_sample, models=model_names, outliers='outliers')
        
    # plot_kde(samples, test_sets['real'], model_names, 'test set')
    # # plot_kde(samples_ext, test_sets_ext, model_names, 'test set outliers')
    # plot_kde_dim(samples, test_sets['real'], model_names, 'test set')
    # plot_kde_dim(samples_ext, test_sets_ext, model_names, 'test set outliers')
    # lineplot_(test_sets['real'], samples, model_names=model_names)

    plot_score(es_and_models_to_plot, test_sets['real'].index, model_names, labels=list(('Energy Score', 'Date')), 
                outliers='test set',max_ytick = 14)    
    plot_score(vs_and_models_to_plot, test_sets['real'].index, model_names, labels=list(('Variogram Score', 'Date')),
                outliers='test set', max_ytick=24, 
                 tick_freq=4)    
    plot_score(es_and_models_to_plot_ext, index_ext, model_names, labels=list(('Energy Score', 'Date')),
                 outliers='test set outliers', max_ytick=14)    
    plot_score(vs_and_models_to_plot_ext, index_ext, model_names, labels=list(('Variogram Score', 'Date')),
                outliers='test set outliers', max_ytick=28, tick_freq=4) 
    plt.close("all")