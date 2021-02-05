import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# loading the data
train = pd.read_csv("train/train.csv")

# get the average of three columns
train['cc_cons_avg'] = train[['cc_cons_apr', 'cc_cons_may', 'cc_cons_jun']].mean(axis=1)

# ------------------------------------------------------------------ #
# -----------------Analyzing data----------------------------------- #
# ------------------------------------------------------------------ #
# get the correlation between the data attributes and the target variable
train[train.columns[1:]].corr()['cc_cons'][:]

# ------------------------------------------------------------------ #
# -----------------Data cleaning and preparation-------------------- #
# ------------------------------------------------------------------ #

# Preparing the feature matrix  and label values to give input to the model for training
x = train[['cc_cons_avg', 'card_lim']].copy()
y = train[['cc_cons']].copy()

# Fill the null values with the median values
# This is continuous data, The box plot determined this data has outliers,
# Hence filling the null values with median value is apt
x.card_lim.fillna(x.card_lim.median(), inplace=True)

# Applying box-cox transformation to smothen the skewed data
transform = np.asarray(x[['card_lim']].values)
tcl = stats.boxcox(transform)[0]

transform = np.asarray(x[['cc_cons_avg']].values)
tca = stats.boxcox(transform)[0]

transform = np.asarray(x[['cc_cons']].values)
tc = stats.boxcox(transform)[0]

# Prepare feature matrix
x_t = np.concatenate((tca, tcl), axis=1)
y_t = tc

# ------------------------------------------------------------------ #
# -----------------Data cleaning and preparation-------------------- #
# ------------------------------------------------------------------ #
# Model from data
model11 = sm.OLS(y_t, x_t).fit()

# model summary
# use model11.params to get param values
model11.summary()

# ------------------------------------------------------------------ #
# -----------------Testing and validation--------------------------- #
# ------------------------------------------------------------------ #

# load testing data
test = pd.read_csv("test/test.csv")
test['cc_cons_avg'] = test[['cc_cons_apr', 'cc_cons_may', 'cc_cons_jun']].mean(axis=1)
px = test[['cc_cons_avg', 'card_lim']].copy()

# Check if any columns has null values and if any null values impute data with median values
# Repeat for other features
pd.isna(test.card_lim).sum()
test.card_lim.fillna(test.card_lim.median(), inplace=True)

# Predict the data
y = model11.predict(px)

# Get the result - ID and respective predicted credit card average for the next three months
# and save as csv
res = test['id'].copy()
result = pd.DataFrame({'id': res, 'cc_cons': y})
result.to_csv(r'result.csv')
