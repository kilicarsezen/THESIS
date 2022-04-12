import numpy as np
import pandas as pd
from pyparsing import alphas
from scipy.stats import rankdata
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import random
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import matplotlib.dates as mdates
mpl.rcParams.update(mpl.rcParamsDefault)
path ="C:/Users/Sezen/MA_THESIS/Evaluation Plots/"
def energy_score_ensemble(obs, predictions):
    m = predictions.shape[0]
    es1 = (1 / m) * np.sum(np.linalg.norm(predictions - obs, axis=1))
    es2 = 1 / (2 * m**2) * (np.sum(np.linalg.norm(predictions[:, np.newaxis] - predictions, axis=2)))
    return es1-es2

def energy_score(obs, predictions):
    m = predictions.shape[0]
    es1 = (1 / m) * np.sum(np.linalg.norm(predictions - obs, axis=1))
    es2 = 1 / (2 * (m - 1)) * (np.sum(np.linalg.norm(predictions[0:-1, :] - predictions[1:, :], axis=1)))
    return es1 - es2

def variogram_Score(obs, preds, w_i_j, p=0.5):
    y_i_j = np.abs(obs - obs[:, np.newaxis])**p
    X_i_j = np.sum(np.abs(preds - preds[:, np.newaxis])**p, axis=2)/preds.shape[1]
    diff = (y_i_j - X_i_j)**2
    v_s = np.sum(np.multiply(w_i_j, diff))
    return v_s

def rmse(obs, samples):
    return np.sqrt(np.mean((obs - np.average(samples, axis=1))**2))


def evaluate_(samples, observations, n_samples, dim, model):
    print("evaluation has started for " + model)
    samples_ev = samples.copy()
    observations_ev = observations.copy()
    n_observations = observations_ev.shape[0]
    energyScore = np.empty(shape=(n_observations))
    energyScore_ensemble = np.empty(shape=(n_observations))
    variogramScore = np.empty(shape=(n_observations))
    samples_ev = np.reshape(samples_ev, (n_observations, n_samples, dim))
    for i in range(n_observations):
        energyScore[i] = energy_score(observations_ev.values[i], samples_ev[i, :, :])
        energyScore_ensemble[i] = energy_score_ensemble(observations_ev.values[i], samples_ev[i, :, :])
        variogramScore[i] = variogram_Score(observations_ev.values[i], np.transpose(samples_ev[i, :, :]), np.ones((4,4)))
    rmse_ev = rmse(observations_ev.values, samples_ev)
    return energyScore, energyScore_ensemble, variogramScore, rmse_ev

def plot_score(scores, index, mdl_name, labels, outliers, max_ytick,tick_freq=2):
    print("plotting {} for {} - {}".format(labels[0], mdl_name, outliers))
    fig, axs = plt.subplots(len(mdl_name), sharex=True, figsize=(12,10), dpi=80, num=labels[0]+outliers)
    colors=['red','green','goldenrod','darkmagenta']
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    for i, ax in enumerate(axs):
        ax.set_yticks(np.arange(0, max_ytick+tick_freq, tick_freq))
        ax.hlines(y=np.arange(0, max_ytick+1, tick_freq), xmin=index[0], xmax=index[-1], colors='grey', linestyles='--')
        ax.plot(index, scores[i], colors[i])
        if mdl_name[i]=='conditional realnvp':
            ax.set_title('Normalizing Flows', fontsize=24)
        elif mdl_name[i] == 'conditional gaussian':
            ax.set_title('Gaussian Probability', fontsize=24)
        elif mdl_name[i] == 'copula LQR':
            ax.set_title('Copula LQR', fontsize=24)
        elif mdl_name[i] == 'copula ANN':
            ax.set_title('Copula ANN', fontsize=24)
        else:
            ax.set_title(mdl_name[i])
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        ax.margins(x=0)
        plt.setp(ax.get_xticklabels(), fontsize=16, rotation=45)
        plt.setp(ax.get_yticklabels(), fontsize=16)
    fig.supxlabel(labels[1],fontsize=24)
    fig.supylabel(labels[0],fontsize=24)
    fig.tight_layout()
    plt.savefig(path +"{} {}.png".format(labels[0], outliers))
    plt.close(fig)

def pairplot_evaluation(test_set, samples, mdl, outliers):
    print("pair plot for {} {}".format(mdl, outliers))
    fig=plt.figure()
    obs_pp = test_set.copy()
    samples_pp = samples.copy()
    target_cols = obs_pp.keys()

    obs_pp['type'] = 'observations'    
    all_cols = obs_pp.keys()

    obs_pp=obs_pp.to_numpy().reshape(obs_pp['type'].shape[0],5)
    
    samples_pp = np.c_[samples_pp, ['samples']*samples_pp.shape[0]]

    data_pp=np.concatenate((samples_pp, obs_pp), axis=0)
    data_pp = pd.DataFrame(data_pp, columns=all_cols)
    data_pp[target_cols] = data_pp[target_cols].astype(np.float64)
    
    g1 = sns.pairplot(data_pp, hue='type', x_vars=target_cols, y_vars=target_cols[0], diag_kind=None)
    g1.set(xlim=(-20,23))
    g1.set(ylim=(-20,23))
    plt.savefig(path +'PairPlots/Pair Plot for {} of {} {}.png'.format(target_cols[0], mdl, outliers))

    g2=sns.pairplot(data_pp, hue='type', x_vars=target_cols, y_vars=target_cols[1], diag_kind=None)
    g2.set(xlim=(-20,23))
    g2.set(ylim=(-20,23))
    plt.savefig(path +'PairPlots/Pair Plot for {} of {} {}.png'.format(target_cols[1], mdl, outliers))
    
    g3=sns.pairplot(data_pp, hue='type', x_vars=target_cols, y_vars=target_cols[2], diag_kind=None)
    g3.set(xlim=(-20,23))
    g3.set(ylim=(-20,23))
    plt.savefig(path +'PairPlots/Pair Plot for {} of {} {}.png'.format(target_cols[2], mdl, outliers))

    g4=sns.pairplot(data_pp, hue='type', x_vars=target_cols, y_vars=target_cols[3], diag_kind=None)
    g4.set(xlim=(-20,23))
    g4.set(ylim=(-20,23))
    plt.savefig(path +'PairPlots/Pair Plot for {} of {} {}.png'.format(target_cols[3], mdl, outliers))
    plt.close("all")
    
def getdimensions(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    c = data[:, 3]
    return x,y,z,c

def plot_4D(observations, predictions):
    x1, y1, z1, c1 = getdimensions(observations)
    x2, y2, z2, c2 = getdimensions(predictions)
    fig = plt.figure()
    fig.set_size_inches(20, 15)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('observations')
    img1 = ax1.scatter(x1, y1, z1, c=c1, cmap=plt.hot())
    ax1.set_xlabel("DA_Q1")
    ax1.set_ylabel("DA_Q2")
    ax1.set_zlabel("DA_Q3")
    ax1.set_xlim3d(-15, 10)
    ax1.set_ylim3d(-15, 10)
    ax1.set_zlim3d(-15, 10)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('samples')
    img2 = ax2.scatter(x2, y2, z2, c=c2, cmap=plt.hot(), alpha=0.3)
    cbar = fig.colorbar(img2, ax=ax2, orientation='vertical')
    cbar.ax.set_ylabel("DA_Q4", rotation = 270)
    ax2.set_xlim3d(-15, 10)
    ax2.set_ylim3d(-15, 10)
    ax2.set_zlim3d(-15, 10)
    plt.close(fig)

def rank_(obs, preds):
    obs_preds = np.vstack((obs, preds))
    rank = np.zeros(obs_preds.shape[0])
    for s in range(obs_preds.shape[0]):
        rank[s] = np.sum(np.apply_along_axis(all, 1, obs_preds <= obs_preds[s, :]))
    return rankdata(rank, method="min")

def rank_hist(observations, preds, n_samples, model_name, num_bins=30):
    print("plotting rank histograms for {}".format(model_name))
    preds_rh = preds.copy()
    preds_rh = np.reshape(preds_rh, (observations.shape[0], n_samples, observations.shape[1]))
    preds_rh = np.delete(preds_rh,  random.sample(range(1, n_samples), n_samples-num_bins), axis=1)
    ranks = np.zeros(observations.shape[0])
    for i in range(observations.shape[0]):
        ranks[i] = rank_(observations[i], preds_rh[i])[0]
    plt.figure(num="Rank Hisogram of {}".format(model_name))
    bins=np.arange(1,num_bins+3) -0.5
    if model_name=='conditional realnvp':
        label='Normalizing Flows'
    elif model_name == 'conditional gaussian':
        label='Gaussian Regression'
    elif model_name== 'copula LQR':
        label='Copula LQR'
    elif model_name == 'copula ANN':
        label='Copula ANN'
    else:
        label=model_name
    plt.hist(ranks, bins = bins, density=True, alpha=1, label=label)
    plt.hlines(1/(num_bins+1), color = 'r', xmin = 0.5, xmax= num_bins+1.5, linestyles='--')
    plt.xlabel("Rank", fontsize=20)    
    plt.ylabel("Relative Frequency", fontsize=20)
    plt.margins(x=0, tight=True)
    if model_name=='conditional realnvp':
        plt.title('Normalizing Flows',fontsize=20)
    elif model_name == 'conditional gaussian':
        plt.title('Gaussian Regression',fontsize=20)
    elif model_name== 'copula LQR':
        plt.title('Copula LQR',fontsize=20)
    elif model_name == 'copula ANN':
        plt.title('Copula ANN',fontsize=20)
    else:
        plt.title(model_name)
    plt.tight_layout()
    plt.savefig(path +'Rank Histogram of {}.png'.format(model_name))
    plt.close("all")

def plot_scenarios(reality, samples, n_sample, models, outliers):
    observations = reality.sample(20)
    observations = observations.sample(20)
    observations_grouped={'1st':observations.iloc[0:2,:], '2nd':observations.iloc[2:4,:],'3rd':observations.iloc[4:6,:],
    '4th':observations.iloc[6:8,:], '5th':observations.iloc[8:10,:],'6th':observations.iloc[10:12,:],
    '7th':observations.iloc[12:14,:], '8th':observations.iloc[14:16,:],'9th':observations.iloc[16:18,:],
    '10th':observations.iloc[18:20,:]}
    delete_samples= random.sample(range(1, n_sample), 985)
    for key, observation in observations_grouped.items():    
        idx_locs = [reality.index.get_loc(i) for i in observation.index]        
        fig, axs = plt.subplots(nrows=observation.shape[0], ncols=len(models),figsize=(40, 30), sharex=True, sharey=True)
        for i, idx_loc in enumerate(idx_locs):
            for m, mdl in enumerate(models):
                scenarios = samples[mdl][idx_loc*n_sample:(idx_loc+1)*n_sample]
                scenarios_ = np.delete(scenarios,  delete_samples, axis=0)
                for scenario in range(0, scenarios_.shape[0]):
                    axs[i, m].plot(scenarios_.T[:, scenario], color='dimgrey',linewidth=3)
                axs[i, m].plot(reality.iloc[idx_loc,:].values.T, color='blue', linewidth=5)
                
                axs[i, m].set_xticks([0,1,2,3]) 
                axs[i, m].set_xticklabels(["Q1","Q2","Q3","Q4"], fontsize=40)
                plt.setp(axs[i,m].get_yticklabels(), fontsize=30)
            
                if mdl=='conditional realnvp':
                    axs[0,m].set_title('Normalizing \n Flows',fontdict={'fontsize': 50}, pad=20)
                elif mdl == 'conditional gaussian':
                    axs[0,m].set_title('Gaussian \n Regression',fontdict={'fontsize': 50}, pad=20)
                elif mdl== 'copula LQR':
                    axs[0,m].set_title('Copula \n LQR',fontdict={'fontsize': 50}, pad=20)
                elif mdl == 'copula ANN':
                    axs[0,m].set_title('Copula \n ANN',fontdict={'fontsize': 50}, pad=20)
                else:
                    axs[0,m].set_title(mdl,fontdict={'fontsize': 30}, pad=30)
            axs[i,0].set_ylabel("Price Delta on \n"+observation.index[i].strftime("%m/%d/%Y, %H:%M") + " \n €/MWh" ,
                                fontdict={'fontsize': 50}, ha='center', rotation='horizontal')
            axs[i,0].yaxis.set_label_coords(-0.55, 0.5)
        plt.close(fig)
        colors = ['grey', 'green']
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
        labels = ['Realization', 'Samples']
        fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(1, 1),fontsize=40)
        plt.tight_layout()
        plt.savefig(path +'scenarios for the {} group - {}.png'.format(key, outliers))

def multi_norm_check(data, plot_path, test_name, nameOfPlot, outlier=False):
    if isinstance(data, pd.DataFrame):
        with localconverter(ro.default_converter + pandas2ri.converter):
            data = ro.conversion.py2rpy(data)
    grdevices = importr('grDevices')
    packages_to_install_if_needed = ("MVN",)
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
    names_to_install = [x for x in packages_to_install_if_needed if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packagescc(StrVector(names_to_install))
    # load the package
    mvn = importr("MVN")
    numpy2ri.activate()
    grdevices.png(file=plot_path + nameOfPlot, width=512, height=512)
    mvn_plt = mvn.mvn(data, mvnTest = test_name, multivariatePlot = "qq")
    grdevices.dev_off()
    mvn_ = mvn.mvn(data, mvnTest = test_name, multivariateOutlierMethod = "adj", showOutliers=outlier)
    return mvn_
    
def plot_kde(samples, observations, models, outliers):
    plt.figure(num="KDE Plot {}".format(outliers), figsize=(6,6))
    colors=['red','green','goldenrod','darkmagenta','blue']
    for idx, mdl in enumerate(models):
        ax=sns.kdeplot(samples[mdl].reshape(samples[mdl].shape[0]*4,), bw=0.5, label=mdl, color=colors[idx], ls='--')
        ax.vlines(x=[np.min(samples[mdl].reshape(samples[mdl].shape[0]*4,)), 
                        np.max(samples[mdl].reshape(samples[mdl].shape[0]*4,))], ymin=0, ymax=0.02, colors=colors[idx], linestyles='-', lw=2)
    ax = sns.kdeplot(observations.values.reshape(observations.shape[0]*4,), bw=0.5, label="observations", color=colors[-1], ls='--')
    ax.vlines(x=[np.min(observations.values.reshape(observations.shape[0]*4,)),
                 np.max(observations.values.reshape(observations.shape[0]*4,))], ymin=0, ymax=0.02, colors=colors[-1], 
                 linestyles='-', lw=2)
    ax.legend()
    plt.xlabel(xlabel="Scaled delta price", fontsize=14)
    plt.ylabel(ylabel="Probability Density", fontsize=14)
    plt.tight_layout()
    plt.savefig(path +'KDE Plot {}.png'.format(outliers))
    plt.close("all")

def plot_kde_dim(samples, observations, models, outliers):
    # plt.figure()
    colors=['red','green','goldenrod','darkmagenta','blue']
    for dim in range(observations.shape[1]):
        fig = plt.figure(num="KDE Plot {} for Dimension {}".format(outliers, dim))
        
        for idx, mdl in enumerate(models):
            if mdl=='conditional realnvp':
                label='Normalizing Flows'
            elif mdl == 'conditional gaussian':
                label='Gaussian Regression'
            elif mdl== 'copula LQR':
                label='Copula LQR'
            elif mdl == 'copula ANN':
                label='Copula ANN'
            else:
                label='model'
            sns.kdeplot(samples[mdl][:, dim], bw=0.5, label=label, color=colors[idx], ls='--')
            print("mdl \n",plt.gca().get_xlim()[0])
            plt.vlines(x=[min(samples[mdl][:, dim]),max(samples[mdl][:, dim])], ymin=0, ymax=0.02, colors=colors[idx], linestyle='-', lw=2)
        sns.kdeplot(observations.iloc[:, dim], bw=0.5, label="observations", color=colors[-1], ls='--')
        plt.vlines(x=[min(observations.iloc[:, dim]), max(observations.iloc[:, dim])], ymin=0, ymax=0.02, colors=colors[-1], linestyles='-', lw=2)
        plt.legend()
        plt.xlabel(xlabel="Scaled delta price in quarter {}".format(dim+1), fontsize=14)
        plt.ylabel(ylabel="Probability Density", fontsize=14)
        plt.tight_layout()
        plt.savefig(path +'KDE Plot {} for Dimension {} .png'.format(outliers, dim))
        plt.close(fig)

def lineplot_(observations,samples, model_names): 
    print("mean value for observations \n {}".format(observations.mean()))
    print("std value for observations \n {}".format(observations.std()))
    samples_all_dict = {}   
    for idx, mdl in enumerate(model_names):
        if mdl=='conditional realnvp':
            label='Normalizing Flows'
        elif mdl == 'conditional gaussian':
            label='Gaussian Regression'
        elif mdl== 'copula LQR':
            label='Copula LQR'
        elif mdl == 'copula ANN':
            label='Copula ANN'
        else:
            label='model'
        samples_df = pd.DataFrame(samples[mdl], columns=['Q1','Q2','Q3','Q4'])
        print("mean value for {} {}".format(mdl, samples_df.mean()))
        print("std value for {} {}".format(mdl, samples_df.std()))
        samples_melt = pd.melt(samples_df)
        # grouped = samples_melt.groupby('variable')
        # samples_melt_grouped = grouped.apply(lambda x: x.sample(400))
        samples_melt_grouped = samples_melt
        samples_melt_grouped['model_name'] = label
        samples_all_dict[mdl] = samples_melt_grouped
    observations = observations.rename(columns={"DA_Q1": "Q1", "DA_Q2": "Q2","DA_Q3": "Q3","DA_Q4": "Q4"})
    observations_melt = pd.melt(observations)
    observations_melt['model_name'] = 'Observations'
    observations_melt.groupby('variable').size()
    data_all = observations_melt
    for mdl in model_names:
        data_all = data_all.append(samples_all_dict[mdl])
    data_all.reset_index(inplace=True)
    fig, ax = plt.subplots(figsize=(12,15))
    colors=['blue','red','green','goldenrod','darkmagenta']
    sns.boxplot(ax=ax, data=data_all, x="variable", y="value", hue="model_name", palette=colors)
    
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    labels = ['observations', 'Normalizing Flows', 'Gaussian Probability','Copule LQR', 'Copula ANN']
    plt.legend(lines, labels,loc='upper left',bbox_to_anchor=[0.7, 1])
    plt.setp(ax.get_legend().get_texts(), fontsize=14)
    ax.set_xlabel('Quarters', fontsize=20)
    ax.set_ylabel('Scaled Price Delta', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(path +'Lineplot.png')
    plt.close(fig)