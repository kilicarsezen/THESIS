import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from download_prices import get_train_test_sets, ID3_quarterly_EDA
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import pandas as pd
from evaluation import multi_norm_check
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter
pd.set_option('display.max_rows', 100)

sns.set_theme()
from scipy.stats import norm, stats

target_col_names = ['DA_Q1', 'DA_Q2', 'DA_Q3', 'DA_Q4']

train_sets, test_sets = get_train_test_sets(dataset=ID3_quarterly_EDA,
                                            numeric_features=[],
                                            to_be_categorized_features=['load_Q1_forecast',
                                                                        'load_Q2_forecast',
                                                                        'load_Q3_forecast',
                                                                        'load_Q4_forecast'],
                                            cyclical_features=['Hour'],
                                            n_sample=1000)
# sns.boxplot(data=ID3_quarterly_EDA[target_col_names])
# plt.show()
# print(ID3_quarterly_EDA[target_col_names].describe())
# sns.pairplot(ID3_quarterly_EDA[target_col_names])
# plt.tight_layout()
# plt.savefig("C:/Users/Sezen/PycharmProjects/MA_THESIS/DA_ID_original_pairplot.png")
# plt.show()
path_eda = "C:/Users/Sezen/MA_THESIS/EDA_plots/"
ID3_quarterly_EDA = ID3_quarterly_EDA.loc[
    (ID3_quarterly_EDA['DA_Q1'] > -150)  & (ID3_quarterly_EDA['DA_Q2'] > -190) & (ID3_quarterly_EDA['DA_Q3'] > -100) & (
            ID3_quarterly_EDA['DA_Q4'] > -150) & (ID3_quarterly_EDA['DA_Q2'] < ID3_quarterly_EDA['DA_Q2'].max()) & (
                ID3_quarterly_EDA['DA_Q3'] < ID3_quarterly_EDA['DA_Q3'].max())& (
                ID3_quarterly_EDA['DA_Q4'] < ID3_quarterly_EDA['DA_Q4'].max())]

multi_norm_check(ID3_quarterly_EDA[target_col_names], path_eda,'dh',"eda_qq_plot.png")

def plot_marginals(data):
    f, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes[0, 0].tick_params(labelsize=10)
    axes[0, 1].tick_params(labelsize=10)
    axes[1, 0].tick_params(labelsize=10)
    axes[1, 1].tick_params(labelsize=10)

    axes[0, 0].set_xlabel("DA_Q1", fontsize=30)
    axes[0, 0].set_ylabel("density", fontsize=30)
    sns.histplot(data.iloc[:, 0], kde=True, stat='density', label='samples', ax=axes[0, 0])
    mu1, std1 = norm.fit(data.iloc[:, 0])
    x1_0, x1_1 = axes[0, 0].get_xlim()
    x1_pdf = np.linspace(x1_0, x1_1, 100)
    y1_pdf = norm.pdf(x1_pdf, mu1, std1)
    axes[0, 0].plot(x1_pdf, y1_pdf, 'r', lw=1, label='pdf')

    axes[0, 1].set_xlabel("DA_Q2", fontsize=30)
    axes[0, 1].set_ylabel("density", fontsize=30)
    sns.histplot(data.iloc[:, 1], kde=True, stat='density', label='samples', ax=axes[0, 1])
    mu2, std2 = norm.fit(data.iloc[:, 1])
    x2_0, x2_1 = axes[0, 1].get_xlim()
    x2_pdf = np.linspace(x2_0, x2_1, 100)
    y2_pdf = norm.pdf(x2_pdf, mu2, std2)
    axes[0, 1].plot(x2_pdf, y2_pdf, 'r', lw=1, label='pdf')

    axes[1, 0].set_xlabel("DA_Q3", fontsize=30)
    axes[1, 0].set_ylabel("density", fontsize=30)
    sns.histplot(data.iloc[:, 2], kde=True, stat='density', label='samples', ax=axes[1, 0])
    mu3, std3 = norm.fit(data.iloc[:, 2])
    x3_0, x3_1 = axes[1, 0].get_xlim()
    x3_pdf = np.linspace(x3_0, x3_1, 100)
    y3_pdf = norm.pdf(x3_pdf, mu3, std3)
    axes[1, 0].plot(x3_pdf, y3_pdf, 'r', lw=1, label='pdf')

    axes[1, 1].set_xlabel("DA_Q4", fontsize=30)
    axes[1, 1].set_ylabel("density", fontsize=30)
    sns.histplot(data.iloc[:, 3], kde=True, stat='density', label='samples', ax=axes[1, 1])
    mu4, std4 = norm.fit(data.iloc[:, 3])
    x4_0, x4_1 = axes[1, 1].get_xlim()
    x4_pdf = np.linspace(x4_0, x4_1, 100)
    y4_pdf = norm.pdf(x4_pdf, mu4, std4)
    axes[1, 1].plot(x4_pdf, y4_pdf, 'r', lw=1, label='pdf')
    for i, ax in enumerate(axes.reshape(-1)):
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15 )
        ax.text(x=0, y=0.97, transform=ax.transAxes, s="Skewness: %.2f" % data.iloc[:, i].skew(),
                fontweight='demibold', fontsize=18, verticalalignment='top', horizontalalignment='left',
                backgroundcolor='white', color='xkcd:poo brown')
        ax.text(x=0, y=0.89, transform=ax.transAxes, s="Kurtosis: %.2f" % data.iloc[:, i].kurt(),
                fontweight='demibold', fontsize=18, verticalalignment='top', horizontalalignment='left',
                backgroundcolor='white', color='xkcd:dried blood')
    plt.tight_layout()
    plt.savefig(path_eda+"distplotdeltaprices.png")
    # plt.show()
# plot_marginals(ID3_quarterly_EDA[target_col_names])


# explore the corr between cat variables and price delta
def box_plot(data, colname):
    f1, (ax1, ax2, ax3, ax4) = plt.subplots(4, gridspec_kw={'hspace': 0}, sharex=True, sharey=True)
    f1.set_size_inches(10, 5)
    # f1.suptitle('Box plot of the delta ID3 prices across' + colname)
    sns.boxplot(x=colname, y="DA_Q1", data=data, ax=ax1, showfliers=False)
    sns.boxplot(x=colname, y="DA_Q2", data=data, ax=ax2, showfliers=False)
    sns.boxplot(x=colname, y="DA_Q3", data=data, ax=ax3, showfliers=False)
    sns.boxplot(x=colname, y="DA_Q4", data=data, ax=ax4, showfliers=False)
    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    ax1.set_ylabel('Delta Price Q1',fontsize = 10)
    ax2.set_ylabel('Delta Price Q2',fontsize = 10)
    ax3.set_ylabel('Delta Price Q3',fontsize = 10)
    ax4.set_ylabel('Delta Price Q4',fontsize = 10)

    # for ax in [ax1, ax2, ax3, ax4]:
    #     ax.label_outer()
    # plt.subplots_adjust(hspace=2)
    plt.tight_layout()
    # plt.savefig("C:/Users/Sezen/PycharmProjects/MA_THESIS/plots/hourbox.png")
    plt.show()

# sns.boxplot(data=ID3_quarterly_EDA[target_col_names])
# box_plot(ID3_quarterly_EDA, 'Hour')
# box_plot(ID3_quarterly_EDA, 'Month')
# box_plot(ID3_quarterly_EDA, 'Day')

def boxplot_categorized_features(data, colname, target_col_names, n_bins=3, strategy='kmeans', encode='ordinal'):
    # data[target_col_names]=data[target_col_names].abs()
    data[colname] = data[colname].abs()
    f1, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    Binizer = KBinsDiscretizer(n_bins=n_bins, strategy=strategy, encode=encode)
    Categories = Binizer.fit_transform(data[[colname]])
    data[colname + '_Bins'] = Categories[:, 0]
    sns.boxplot(x=colname + '_Bins', y=target_col_names[0], data=data, ax=ax1, palette="Set3")
    sns.boxplot(x=colname + '_Bins', y=target_col_names[1], data=data, ax=ax2, palette="Set3")
    sns.boxplot(x=colname + '_Bins', y=target_col_names[2], data=data, ax=ax3, palette="Set3")
    sns.boxplot(x=colname + '_Bins', y=target_col_names[3], data=data, ax=ax4, palette="Set3")
    sns.set(rc={'figure.figsize': (50, 50)})
    plt.subplots_adjust(left=0.1, right=0.15, top=0.5, bottom=0.1, hspace=5)
    plt.tight_layout()
    plt.show()


# boxplot_categorized_features(ID3_quarterly,'load_Q4_error', target_col_names)


def summary(x, **kwargs):
    label = x.describe()[['mean', 'std']].round()
    label['skewness'] = x.skew().round()
    label['kurtosis'] = x.kurt().round()
    ax = plt.gca()
    ax.set_axis_off()
    # use .to_string()
    ax.annotate(label[['skewness', 'kurtosis']].to_string(), xy=(0.4, 1), xycoords=ax.transAxes,
                fontweight='demibold', fontsize=24, verticalalignment='top', horizontalalignment='right',
                backgroundcolor='white', color='xkcd:dark blue'
                )
    ax.annotate(label[['mean', 'std']].to_string(), xy=(1, 1), xycoords=ax.transAxes,
                fontweight='demibold', fontsize=24, verticalalignment='top', horizontalalignment='right',
                backgroundcolor='white', color='xkcd:dark blue'
                )


def plot_grid(data):
    g = sns.PairGrid(data)
    g.map_diag(sns.histplot, kde=True, stat='density', label='samples')
    g.map_diag(summary)
    g.map_upper(sns.scatterplot, edgecolor='w')
    g.map_lower(sns.kdeplot)
    for ax in g.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), rotation=0, fontsize=34)
        ax.set_xticks(np.arange(-400,200,100))
        ax.tick_params(axis='both', labelsize=28)
        ax.set_ylabel(ax.get_ylabel(), rotation=90, fontsize=34)
        ax.set_yticks(np.arange(-400,200,100))

        loc = plticker.MultipleLocator(100)  # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)
        loc = plticker.MultipleLocator(100)  # this locator puts ticks at regular intervals

        ax.yaxis.set_major_locator(loc)

        # set y labels alignment
        ax.yaxis.get_label().set_horizontalalignment('right')
    g.fig.set_size_inches(30, 23)

    # g.fig.suptitle('Pair plot of delta ID prices', y=1)
    plt.tight_layout()
    plt.savefig(path_eda +"DA_ID_pairplot.png")


# plot_grid(ID3_quarterly_EDA[target_col_names])


def scatter(data, col_name, figname):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,5),dpi=150)
    # fig.suptitle('Scatter Plot of Delta Prices and ' + col_name, size=20)
    # ax1.set_title('Delta Price in Qurater 1 \n vs ' + col_name[0])
    sns.scatterplot(y='DA_Q1', x=col_name[0], data=data, ax=ax1)
    ax1.set_xlabel(col_name[0],fontsize = 16)
    ax1.set_ylabel('Price Delta Q1',fontsize = 16)

    # ax2.set_title('Delta Price in Qurater 2 \n vs ' + col_name[1])
    sns.scatterplot(y='DA_Q2', x=col_name[1], data=data, ax=ax2)
    ax2.set_xlabel(col_name[1],fontsize = 16)
    ax2.set_ylabel('Price Delta Q2',fontsize = 16)

    # ax3.set_title('Delta Price in Qurater 3 \n vs ' + col_name[2])
    sns.scatterplot(y='DA_Q3', x=col_name[2], data=data, ax=ax3)
    ax3.set_xlabel(col_name[2],fontsize = 16)
    ax3.set_ylabel('Price Delta Q3',fontsize = 16)

    # ax4.set_title('Delta Price in Qurater 4 \n vs ' + col_name[3])
    sns.scatterplot(y='DA_Q4', x=col_name[3], data=data, ax=ax4)
    ax4.set_xlabel(col_name[3],fontsize = 16)
    ax4.set_ylabel('Price Delta Q4',fontsize = 16)


    # ax1.tick_params(axis='both', labelsize=15)
    # ax2.tick_params(axis='both', labelsize=15)
    # ax3.tick_params(axis='both', labelsize=15)
    # ax4.tick_params(axis='both', labelsize=15)

    # plt.subplots_adjust(left=0.1, right=0.15, top=0.5, bottom=0.1, hspace=0.3)

    plt.tight_layout()
    plt.savefig(path_eda+ figname + ".png")

scatter(ID3_quarterly_EDA,['Wind_Q1_error','Wind_Q2_error','Wind_Q3_error','Wind_Q4_error'],"DA_ID_wind1234_scatter")
scatter(ID3_quarterly_EDA,['Solar_Q1_error','Solar_Q2_error','Solar_Q3_error','Solar_Q4_error'],"DA_ID_solar1234_scatter")
scatter(ID3_quarterly_EDA,['load_Q1_error','load_Q2_error','load_Q3_error','load_Q4_error'],"DA_ID_demand1234_scatter")
# scatter(ID3_quarterly_EDA[target_col_names],'DA_Q1')
scatter(ID3_quarterly_EDA, ['Conventional Q1', 'Conventional Q2','Conventional Q3','Conventional Q4'], "DA_ID_conventional_scatter")
# scatter(ID3_quarterly_EDA, ['Conventional', 'Conventional','Conventional','Conventional'])
# print(ID3_quarterly_EDA.columns)

def plot_DA_ID(data, col_QH, col_DA):
    mean_ID_prices = data.groupby(data.index.hour)[col_QH].mean()
    df = mean_ID_prices.stack().reset_index(name='Value').rename(columns={'level_2': 'Quarter'})
    print("index print")
    print(df.index)
    mean_DA = data.groupby(by=data.index.hour)[col_DA].mean()
    mean_DA_repeated = mean_DA.loc[mean_DA.index.repeat(4)]
    mean_DA_repeated.reset_index(inplace=True, drop=True)
    ticks = list(range(1, 100, 4))
    plt.figure(figsize=(9,5), dpi=150)
    plt.plot(df['Value'], drawstyle='steps-pre', label='Intraday')
    plt.plot(mean_DA_repeated, drawstyle='steps-pre', label='Day Ahead')
    plt.xticks(np.arange(0, 97, 4), ticks, rotation=90, fontsize=10)
    plt.yticks()
    plt.xlabel('Quarter of the Day')
    plt.ylabel(' Price (€/MWh)')
    plt.legend(loc="upper left", prop={'size': 12})
    plt.savefig(path_eda +"IDfollowsdayahead.png")
    plt.show()


# plot_DA_ID(ID3_quarterly_EDA, ['Q1', 'Q2', 'Q3','Q4'], 'Day Ahead Auction')
# print(multivariate_normality(ID3_quarterly_EDA[target_col_names], alpha=.05))
def plot_DA_ID_hour(data, col_QH, col_DA):
    mean_ID_prices = data.groupby(data.index.hour)[col_QH].mean()
    df = mean_ID_prices.stack().reset_index(name='Value').rename(columns={'level_2': 'Quarter'})
    mean_DA = data.groupby(by=data.index.hour)[col_DA].mean()
    mean_DA=mean_DA.to_frame()
    print(mean_DA)
    mean_DA_index = np.arange(1,97, 4)
    mean_DA.index=mean_DA_index
    print(mean_DA)
    # mean_DA_repeated = mean_DA.loc[mean_DA.index.repeat(4)]
    # mean_DA_repeated.reset_index(inplace=True, drop=True)
    ticks = list(range(1, 100, 4))
    plt.figure(dpi=150)
    plt.plot(df['Value'], label='Intraday')
    plt.plot(mean_DA, label='Day Ahead')
    plt.xticks(np.arange(0, 97, 4), ticks, rotation=90, fontsize=10)
    plt.yticks()
    plt.xlabel('Quarter of the Day')
    plt.ylabel(' Price (€/MWh)')
    plt.legend(loc="upper left", prop={'size': 12})
    plt.savefig("C:/Users/Sezen/PycharmProjects/MA_THESIS/plots for the paper/IDfollowsdayahead.png")
    plt.show()

# plot_DA_ID_hour(ID3_quarterly_EDA, ['Q1', 'Q2', 'Q3','Q4'], 'Day Ahead Auction')
ID3_quarterly_EDA.loc[:, 'Average Load Forecast'] = ID3_quarterly_EDA.loc[:,
                                                    ['load_Q1_forecast', 'load_Q2_forecast', 'load_Q3_forecast',
                                                     'load_Q4_forecast']].mean(axis=1)
ID3_quarterly_EDA.loc[:, 'Average Load Actual'] = ID3_quarterly_EDA.loc[:,
                                                  ['load_Q1_actual', 'load_Q2_actual', 'load_Q3_actual',
                                                   'load_Q4_actual']].mean(axis=1)
ID3_quarterly_EDA.loc[:, 'Average Load Error'] = ID3_quarterly_EDA.loc[:,
                                                 'Average Load Forecast'] - ID3_quarterly_EDA.loc[:,
                                                                            'Average Load Actual']
# scatter(ID3_quarterly_EDA,'Average Load Actual', 'Average Load Actual')
# scatter(ID3_quarterly_EDA,'Average Load Forecast', 'Average Load Forecast')
# scatter(ID3_quarterly_EDA,['Average Load Error','Average Load Error','Average Load Error','Average Load Error' ],
# 'Average Load Error')
#
ID3_quarterly_EDA.loc[:, 'average_wind_forecast'] = ID3_quarterly_EDA.loc[:,
                                                    ['Wind_Q1_forecast', 'Wind_Q2_forecast', 'Wind_Q3_forecast',
                                                     'Wind_Q4_forecast']].mean(axis=1)
ID3_quarterly_EDA.loc[:, 'average_wind_actual'] = ID3_quarterly_EDA.loc[:,
                                                  ['Wind_Q1_actual', 'Wind_Q2_actual', 'Wind_Q3_actual',
                                                   'Wind_Q4_actual']].mean(axis=1)
ID3_quarterly_EDA.loc[:, 'Average Wind Error'] = ID3_quarterly_EDA.loc[:,
                                                 'average_wind_actual'] - ID3_quarterly_EDA.loc[:,
                                                                          'average_wind_forecast']
# scatter(ID3_quarterly_EDA, ['Average Wind Error','Average Wind Error','Average Wind Error','Average Wind Error'],
# 'Average Wind Error')
#

ID3_quarterly_EDA.loc[:, 'Conventional/Wind Forecast Error Q1'] =  ID3_quarterly_EDA.loc[ID3_quarterly_EDA['Wind_Q1_error']!=0,
                                                                      'Conventional']  / ID3_quarterly_EDA.loc[ID3_quarterly_EDA['Wind_Q1_error']!=0,
                                                                                        'Wind_Q1_error']
ID3_quarterly_EDA.loc[:, 'Conventional/Wind Forecast Error Q2'] = ID3_quarterly_EDA.loc[ID3_quarterly_EDA['Wind_Q2_error']!=0,
                                                                      'Conventional']  / ID3_quarterly_EDA.loc[ID3_quarterly_EDA['Wind_Q2_error']!=0,
                                                                                        'Wind_Q2_error'] 
ID3_quarterly_EDA.loc[:, 'Conventional/Wind Forecast Error Q3'] = ID3_quarterly_EDA.loc[ID3_quarterly_EDA['Wind_Q3_error']!=0,
                                                                      'Conventional']  / ID3_quarterly_EDA.loc[ID3_quarterly_EDA['Wind_Q3_error']!=0,
                                                                                        'Wind_Q3_error'] 
ID3_quarterly_EDA.loc[:, 'Conventional/Wind Forecast Error Q4'] = ID3_quarterly_EDA.loc[  ID3_quarterly_EDA['Wind_Q4_error']!=0,
                                                                      'Conventional']   / ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['Wind_Q4_error']!=0,
                                                                                        'Wind_Q4_error']          
scatter(ID3_quarterly_EDA, ['Conventional/Wind Forecast Error Q1','Conventional/Wind Forecast Error Q2',
'Conventional/Wind Forecast Error Q3','Conventional/Wind Forecast Error Q4'], "DA_ID_conventional-wind1234_scatter")


ID3_quarterly_EDA.loc[:, 'average_solar_forecast'] = ID3_quarterly_EDA.loc[:,
                                                     ['Solar_Q1_forecast', 'Solar_Q2_forecast', 'Solar_Q3_forecast',
                                                      'Solar_Q4_forecast']].mean(axis=1)
ID3_quarterly_EDA.loc[:, 'average_solar_actual'] = ID3_quarterly_EDA.loc[:,
                                                   ['Solar_Q1_actual', 'Solar_Q2_actual', 'Solar_Q3_actual',
                                                    'Solar_Q4_actual']].mean(axis=1)
ID3_quarterly_EDA.loc[:, 'average_solar_error'] = ID3_quarterly_EDA.loc[:,
                                                  'average_solar_actual'] - ID3_quarterly_EDA.loc[:,
                                                                            'average_solar_forecast']
# scatter(ID3_quarterly_EDA, 'average_solar_error', 'average_solar_error')
#
ID3_quarterly_EDA.loc[:, 'Conventional/Solar Forecast Error Q1'] = ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['Solar_Q1_error']!=0,
                                                                      'Conventional'] / ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['Solar_Q1_error']!=0,
                                                                                        'Solar_Q1_error']
ID3_quarterly_EDA.loc[:, 'Conventional/Solar Forecast Error Q2'] = ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['Solar_Q2_error']!=0,
                                                                      'Conventional'] / ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['Solar_Q2_error']!=0,
                                                                                        'Solar_Q2_error']
ID3_quarterly_EDA.loc[:, 'Conventional/Solar Forecast Error Q3'] = ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['Solar_Q3_error']!=0,
                                                                      'Conventional'] / ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['Solar_Q3_error']!=0,
                                                                                        'Solar_Q3_error']
ID3_quarterly_EDA.loc[:, 'Conventional/Solar Forecast Error Q4'] = ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['Solar_Q4_error']!=0,
                                                                      'Conventional'] / ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['Solar_Q4_error']!=0,
                                                                                        'Solar_Q4_error']                                
scatter(ID3_quarterly_EDA, ['Conventional/Solar Forecast Error Q1','Conventional/Solar Forecast Error Q2',
'Conventional/Solar Forecast Error Q3','Conventional/Solar Forecast Error Q4'], "DA_ID_conventional-solar1234_scatter")

ID3_quarterly_EDA.loc[:, 'Conventional/Demand Forecast Error Q1'] = ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['load_Q1_error']!=0,
                                                                      'Conventional'] / ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['load_Q1_error']!=0,
                                                                                        'load_Q1_error']
ID3_quarterly_EDA.loc[:, 'Conventional/Demand Forecast Error Q2'] = ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['load_Q2_error']!=0,
                                                                      'Conventional'] / ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['load_Q2_error']!=0,
                                                                                        'load_Q2_error']
ID3_quarterly_EDA.loc[:, 'Conventional/Demand Forecast Error Q3'] = ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['Solar_Q3_error']!=0,
                                                                      'Conventional'] / ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['load_Q3_error']!=0,
                                                                                        'load_Q3_error']
ID3_quarterly_EDA.loc[:, 'Conventional/Demand Forecast Error Q4'] = ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['load_Q4_error']!=0,
                                                                      'Conventional'] / ID3_quarterly_EDA.loc[
                                                                          ID3_quarterly_EDA['load_Q4_error']!=0,
                                                                                        'load_Q4_error']                                
scatter(ID3_quarterly_EDA, ['Conventional/Demand Forecast Error Q1','Conventional/Demand Forecast Error Q2',
'Conventional/Demand Forecast Error Q3','Conventional/Demand Forecast Error Q4'],"DA_ID_conventional-demand1234_scatter")


ID3_quarterly_EDA.loc[:, 'Average Renewable Actual'] = ID3_quarterly_EDA.loc[:,
                                                       'average_wind_actual'] + ID3_quarterly_EDA.loc[:,
                                                                                'average_solar_actual']
ID3_quarterly_EDA.loc[:, 'Average Renewable Forecast'] = ID3_quarterly_EDA.loc[:,
                                                         'average_wind_forecast'] + ID3_quarterly_EDA.loc[:,
                                                                                    'average_solar_forecast']
ID3_quarterly_EDA.loc[:, 'Average Renewable Error'] = ID3_quarterly_EDA.loc[:,
                                                      'Average Renewable Actual'] - ID3_quarterly_EDA.loc[:,
                                                                                    'Average Renewable Forecast']
# scatter(ID3_quarterly_EDA,'Average Renewable Error','Average Renewable Error')

#
ID3_quarterly_EDA.loc[:, 'Renewable/Average Load Actual'] = ID3_quarterly_EDA.loc[:,
                                                            'Average Renewable Actual'] / ID3_quarterly_EDA.loc[:,
                                                                                          'Average Load Actual']
ID3_quarterly_EDA.loc[:, 'Renewable Forecast/Average Load Forecast'] = ID3_quarterly_EDA.loc[:,
                                                                       'Average Renewable Forecast'] / \
                                                                       ID3_quarterly_EDA.loc[:,
                                                                       'Average Load Forecast']
# scatter(ID3_quarterly_EDA,'Renewable Forecast/Average Load Forecast','Renewable Forecast/Average Load Forecast')

#
ID3_quarterly_EDA.loc[:, 'Conventional/Average Load Actual'] = ID3_quarterly_EDA.loc[:,
                                                               'Conventional'] / ID3_quarterly_EDA.loc[:,
                                                                                 'Average Load Actual']
ID3_quarterly_EDA.loc[:, 'Conventional/Average Load Forecast'] = ID3_quarterly_EDA.loc[:,
                                                                 'Conventional'] / ID3_quarterly_EDA.loc[:,
                                                                                   'Average Load Forecast']
# scatter(ID3_quarterly_EDA,'Conventional/Average Load Actual','Conventional/Average Load Actual')
# scatter(ID3_quarterly_EDA,'Conventional/Average Load Forecast','Conventional/Average Load Forecast')

#
ID3_quarterly_EDA.loc[:, 'Conventional/Average Renewable Actual'] = ID3_quarterly_EDA.loc[:,
                                                                    'Conventional'] / ID3_quarterly_EDA.loc[:,
                                                                                      'Average Renewable Actual']
ID3_quarterly_EDA.loc[:, 'Conventional/Average Renewable Forecast'] = ID3_quarterly_EDA.loc[:,
                                                                      'Conventional'] / ID3_quarterly_EDA.loc[:,
                                                                                        'Average Renewable Forecast']
# scatter(ID3_quarterly_EDA,'Conventional/Average Renewable Forecast','Conventional/Average Renewable Forecast')


ID3_quarterly_EDA.loc[:, 'Conventional/Average Renewable Error'] = ID3_quarterly_EDA.loc[:,
                                                                   'Conventional'] / ID3_quarterly_EDA.loc[:,
                                                                                     'Average Renewable Error']
ID3_quarterly_EDA.loc[:, 'Conventional/Average Load Error'] = ID3_quarterly_EDA.loc[:, 'Conventional'] / \
                                                              ID3_quarterly_EDA.loc[:,
                                                              'Average Load Error']

# scatter(ID3_quarterly_EDA,'Conventional/Average Renewable Error','Conventional/Average Renewable Error')
# scatter(ID3_quarterly_EDA.loc[ID3_quarterly_EDA['Conventional/Average Renewable Error']<100],'Conventional/Average
# Renewable Error','Conventional/Average Renewable Error')
# scatter(ID3_quarterly_EDA.loc[ID3_quarterly_EDA['Conventional/Average Load Error']<100],
# 'Conventional/Average Load Error','Conventional/Average Load Error')

# print(ID3_quarterly_EDA.loc[ID3_quarterly_EDA['Solar_Q1_forecast']>0].corr()[target_col_names])
# print(ID3_quarterly_EDA.corr()[target_col_names])

# plot_marginals(ID3_quarterly_EDA[target_col_names])

plt.show()
