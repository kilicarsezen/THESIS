# THESIS
This folder contains the scripts, data, plots, and results
for the master thesis "Multivariate Probabilistic Modeling of
Intraday Elctricity Prices Using Normalizing Flows" from Sezen
Kilicatslan for her degree in Information Systems at the Humboldt-Universität 
zu Berlin (2022).

### Content of the files
download_prices.py downloads the price data from the website
https://energy-charts.info/?l=en&c=DE and loads the other data from 
user's local system. Data preprocessing and train-test split is also
performed in the same file.
helper.py is a helper function that names the columns when the get_feature_names() 
method of ColumnTransformer fails.
EDA.py is the explanatory data analysis
RealNVP.py implements the realnvp approach.
Benchmark.py file includes the Gaussian Regression and Copula models
Two_moon.py is the MWE for RealNVP.py
Training.py has functions that train the models and perform prediction. Moreover,
hyperparameter tuning is also under this file.
evaluation.py applies the evaluation methods described in the thesis 
testing.py is the main script where training, prediction and evaluation functions are called. 

