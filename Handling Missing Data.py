
# coding: utf-8

# # Handling Missing Data
# 
# Will Badart <badart_william@bah.com>
# 
# created: **SEP 2018**
# 
# This notebook is an explanation of different techniques for handling missing data (particularly, large swaths of missing data). We will compare how each technique affects models' performance. I'll be using [this][SO] Stack Overflow post as an outline for the different techniques we'll explore.
# 
# [SO]: https://stackoverflow.com/a/35684975/4025659

# ## The Dataset
# 
# I'll be using the [breast cancer][dataset] dataset from `sklearn` as a base, and performing a (reverse?) pre-processing step of removing the values from a random sampling of cells to simulate the problem of missing data.
# 
# [dataset]: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer

# In[1]:


import numpy as np

from random import (
    choices, random, randint, seed as seed_py)
from sklearn.datasets import load_breast_cancer

RANDOM_STATE = 0xdecafbad
PROB_MISSING = 0.9

seed_py(RANDOM_STATE)

def should_i_do_it():
    return random() < PROB_MISSING

def stomp_indexes(x):
    options = list(range(len(x)))
    return choices(options,
                   # See NOTE below
                   weights=[6, 3, 1] * 10,
                   k=randint(0, len(x)))

def append_col(arr, newcol):
    assert len(arr) == len(newcol)
    return np.append(
        arr, newcol.reshape(len(newcol), 1), axis=1)

X, y = load_breast_cancer(return_X_y=True)

for x in X:
    if should_i_do_it():
        targets = stomp_indexes(x)
        x[targets] = np.nan

Xy = append_col(X, y)


# **NOTE:** The weights here are a cycle of 30 values which alternate between `6`, `3`, and `1`. The acheived effect is that 10 of the features are missing, on average, more than 20% of the time, 10 of the features 15-20% of the time, and the remaining 10 less than 10% of the time.
# 
# This creates a distinction between "good" features (with fewer missing values) and bad ones.

# ## Assessing the Damage
# 
# Below, I note a few summary statistics to give an idea of the distribution of missing values.
# 
# First, we note the proportion of values which have been squashed.

# In[2]:


import matplotlib.pyplot as plt

def proportion_na(col):
    return sum(np.isnan(col)) / len(col)

proportions = [proportion_na(col) for col in X.T]
ax = plt.gca()
ax.bar(range(len(proportions)), proportions)
ax.set_ylim([0, 1])
plt.show()


# Please review the above proportions and adjust `PROB_MISSING` and the `weights` keyword argument to `choices` to your preference.
# 
# Here, the `count` row shows the number of non-missing values in the column.

# In[10]:


import pandas as pd

def describe(A):
    return pd.DataFrame(A).describe()
describe(Xy)


# Below is the proportion of data objects with no missing values at all:

# In[4]:


len([x for x in X if not any(np.isnan(x))]) / len(X)


# ## Dealing with it
# 
# The five strategies I'm going to try are:
# 
# 1. *Drop rows with missing data:* it's exactly what it sounds like
# 2. *Mean/ mode:* fill missing cells with the mean (if we had categorical attributes, mode) of the present values of the column
# 3. *Conditional mean/mode:* same as (2) but only take the mean of rows which share your label
# 4. *Hot-decking:* use a distance metric to find the closest row which has a value in your missing column, and use that
# 5. *KNN:* same as hot-deck but *K* > 1.
# 
# ### 1. Drop rows with missing values
# 
# Since (1) affects the shape of `X`, there also a little extra handling that needs to be done for `y`:

# In[11]:


def filter_na(A):
    combined = np.array([
        x for x in A if not any(np.isnan(x))])
    return combined

Xy_dropped = filter_na(Xy)
describe(Xy_dropped)


# The rest of the strategies don't change `y`, but some need to consider it.
# 
# ### 2. Fill with column mean

# In[6]:


def fill_mean(ax):
    ax[np.isnan(ax)] = ax[~np.isnan(ax)].mean()
    return ax

X_mean = np.copy(X)
np.apply_along_axis(fill_mean, 0, X_mean)
Xy_mean = append_col(X_mean, y)


# ### 3. Conditional fill with column mean

# In[12]:


def fill_cond_mean(A):
    for row in A:
        same_class = A[A[:, -1] == row[-1]]
        for j, v in enumerate(row):
            if np.isnan(v):
                col = same_class[:, j]
                row[j] = col[~np.isnan(col)].mean()
    return A

Xy_cond = np.copy(Xy)
fill_cond_mean(Xy_cond)
describe(Xy_cond)


# ### 4. Hot-decking

# ### 5. KNN

# In[13]:


from functools import partial
from heapq import nsmallest
from scipy.spatial.distance import euclidean

def skip_by_index(target_idx, a):
    return a[[i for i, _ in enumerate(a) if i != target_idx]]

def euc_with_missing(missing_idx, x, y):
    """
    Compute the euclidean distance between x and y. This will
    get called when x is missing its value at missing_idx. If
    y is also missing this value, it is disqualified. Also, y
    must have a value for all of x's non-missing values.
    """
    x_nan, y_nan = np.isnan(x), np.isnan(y)
    if y_nan[missing_idx] or np.isnan(y[~x_nan]).any():
        return float('inf')
    x_nonan = x[~x_nan]
    return euclidean(x_nonan, y[~x_nan])
    
def knn(A, x, skip, k=5):
    return np.array(nsmallest(
        k, A, key=partial(euc_with_missing, skip, x)))

def fill_knn(A, k=5):
    for row in A:
        for j, v in enumerate(row):
            if np.isnan(v):
                neighbors = knn(A, row, j, k)
                col = neighbors[:, j]
                assert not np.isnan(col).any()
                row[j] = col.mean()
    return A

X_knn = np.copy(X)
fill_knn(X_knn, int(len(X) ** .5))
Xy_knn = append_col(X_knn, y)
describe(Xy_knn)


# ## Showdown
# 
# So which enriched dataset yields the best model? Let's find out.

# In[16]:


from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

MAX_DEPTH = 12

def split_labels(Xy):
    return Xy[:, :-1], Xy[:, -1]

data = {
    'dropped': split_labels(Xy_dropped),
    'mean': split_labels(Xy_mean),
    'cond': split_labels(Xy_cond),
    'knn': split_labels(Xy_knn) }

for name, (X, y) in data.items():
    print(name)
    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('accuracy:', accuracy_score(y_test, y_pred))
    print('f1-score:', f1_score(y_test, y_pred))
    print()

