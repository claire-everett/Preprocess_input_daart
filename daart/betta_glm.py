#!/usr/bin/env python
# coding: utf-8

# ## fit GLMs on fish data

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
from glob import glob
import os 
os.chdir('/Users/claireeverett/Desktop/Process_input/scripts/')

from daart.data import load_feature_csv, load_label_csv, split_trials
from daart.eval import get_precision_recall

from IPython.display import Audio


# ### helper functions



def expand_with_lags(data, n_lags=0):
    """Construct a design matrix with lags for a regression model.
    
    Parameters
    ----------
    data : array-like
        shape of (n_samples, n_features)
    n_lags : int
        number of lags *preceding* and *proceeding* each time bin
        
    Returns
    -------
    array-like
    
    """
    data_w_shifts = []
    #for shift in np.arange(-n_lags, n_lags + 1): # if. you want lag in both directions
    for shift in np.arange(-n_lags, 1): # lag leading up to predicted event
        if shift == 0:
            data_tmp = data.copy()
        else:
            data_tmp = np.roll(data, shift=shift, axis=0)
        if shift > 0:
            data_tmp[:shift] = 0
        elif shift < 0:
            data_tmp[shift:] = 0
        data_w_shifts.append(data_tmp)
    return np.hstack(data_w_shifts)


def logistic_regression_search(
        features_train, targets_train, features_val=None, targets_val=None, l2_reg_list=[0],
        max_iter=5000, verbose=True
):
    """Fit a series of logistic regression binary classifiers.

    Parameters
    ----------
    features_train : np.ndarray
        shape (n_samples, n_features)
    targets_train : np.ndarray
        shape (n_samples,)
    features_val : np.ndarray, optional
        shape (n_samples, n_features); if None, results dict will contain evaluation on train data
    targets_val : np.ndarray, optional
        shape (n_samples,); if None, results dict will contain evaluation on train data
    l2_reg_list : list of int, optional
        l2 regularization parameter on weights
    max_iter : int, optional
        max number of iterations
    verbose : bool, optional
        print info along the way

    Returns
    -------
    tuple
        list: models
        pd.DataFrame: eval results

    """
    from sklearn.linear_model import LogisticRegression
    metrics = []
    models = []
    index = 0
    for l2_reg in l2_reg_list:

        model = LogisticRegression(
            penalty='l2', max_iter=max_iter, C=l2_reg, class_weight='balanced')
        if verbose:
            print(model)
        t_beg = time.time()
        model.fit(features_train, targets_train)
        models.append(model)
        t_end = time.time()
        if verbose:
            print('fitting time: %1.2f sec' % (t_end - t_beg))

        if features_val is not None and targets_val is not None:
            preds = model.predict(features_val)
            results = get_precision_recall(targets_val, preds, background=None)
        else:
            print('no validation data; evaluating on train data')
            preds = model.predict(features_train)
            results = get_precision_recall(targets_train, preds, background=None)

        metrics.append(pd.DataFrame({
            'l2_reg': l2_reg,
            'precision': np.mean(results['precision']),
            'recall': np.mean(results['recall']),
            'f1': np.mean(results['f1']),
        }, index=[index]))
        index += 1

    metrics = pd.concat(metrics)
    return models, metrics


# ### fit models




# ------------------------------------
# user options
# ------------------------------------
# data split fractions
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

# number of consecutive time points for chunks of data
chunk_len = 500

# number of pre/post lags in features to use for predicting target
#n_lags = 15

# define hyperparameters to loop over
l2_reg_list = [1e-3, 1e-2, 1e-1, 1e-6, 1e-7]

# filepath_labels = '/media/mattw/behavior/results/daart/fish/multi-6/dtcn/ternary3/version_0/6.1.1L_states.npy'
filepath_labels_dir = '/Users/claireeverett/Desktop/demo_daart/GLM/data/manual/left_new/'
filepath_labels = sorted(glob(os.path.join(filepath_labels_dir,'*.csv')))

filepath_features_dir = '/Users/claireeverett/Desktop/demo_daart/GLM/data/basefeatures/right_new/'
filepath_features = sorted(glob(os.path.join(filepath_features_dir,'*.csv')))

# ------------------------------------
# load data
# ------------------------------------
results_list = []
for i in np.arange(len(filepath_labels)):
    for p in np.arange(40, 100, 1): 
        n_lags = p
        #for n in np.arange(100, )
        # store data chunks
        features_dict = {'train': [], 'val': [], 'test': []}
        targets_dict = {'train': [], 'val': [], 'test': []}

        # load discrete behavior from one fish
        targets, target_names = load_label_csv(filepath_labels[i])
        print('targets:')
        for t, target_name in enumerate(target_names):
            print('    %s: %i' % (target_name, np.sum(targets[:, t])))
        print()
        # turn one-hot vector into a dense representation
        targets = np.argmax(targets, axis=1)  

        # load features from the paired fish
        features, feature_names = load_feature_csv(filepath_features[i])
        print('feature names: {}'.format(feature_names))
        # expand features to include lags
        features_w_lags = expand_with_lags(features, n_lags=n_lags)

        # ------------------------------------
        # break data into train/val/test folds
        # ------------------------------------
        # get train/val/test indices wrt chunks
        assert (train_frac + val_frac + test_frac) == 1.0
        n_chunks = targets.shape[0] // chunk_len
        idxs_xv_chunk = split_trials(
            n_chunks, rng_seed=0, train_tr=int(train_frac*10), val_tr=int(val_frac*10), test_tr=int(test_frac*10), gap_tr=0)

        # get train/val/test indices wrt time points
        idxs_xv = {}
        for dtype, idxs_ in idxs_xv_chunk.items():
            tmp = [np.arange(c * chunk_len, (c + 1) * chunk_len) for c in idxs_]
            idxs_xv[dtype] = np.sort(np.concatenate(tmp))

            features_dict[dtype] = features_w_lags[idxs_xv[dtype]]
            targets_dict[dtype] = targets[idxs_xv[dtype]]

        #l2_reg_list = [1e-5, 1e-6, 1e-7]

        # fit models
        models, metrics_df = logistic_regression_search(
            features_dict['train'], targets_dict['train'], 
            features_dict['val'], targets_dict['val'], 
            l2_reg_list,
        )

        # get best model
        idxs_max = metrics_df.f1.idxmax()
        model_best  = models[idxs_max]

        # evaluate best model on test data
        preds = model_best.predict(features_dict['test'])
        results = get_precision_recall(targets_dict['test'], preds, background=None)
        results_list.append(['feat' + os.path.basename(filepath_features[i]).split('new')[0] + '_targ' + os.path.basename(filepath_labels[i]).split('_')[0],results])
        print(results)

sound_file = '/Users/claireeverett/Desktop/demo_daart/GLM/data/Bing Bong Sound Effects.mp3'
Audio(sound_file, autoplay=True)




results_list




results_new = pd.DataFrame(results_list)




results_new.to_csv('/Users/claireeverett/Desktop/demo_daart/GLM/lag_analysis_2state_40100.csv')





np.arange(10, 20, 1)




metrics_df




dtype = 'train'
preds = model_best.predict(features_dict[dtype])
results = get_precision_recall(targets_dict[dtype], preds, background=None)
print(results)




dtype = 'val'
preds = model_best.predict(features_dict[dtype])
results = get_precision_recall(targets_dict[dtype], preds, background=None)
print(results)




dtype = 'test'
preds = model_best.predict(features_dict[dtype])
results = get_precision_recall(targets_dict[dtype], preds, background=None)
print(results)





plt.plot(targets_dict['test'])





plt.plot(preds)




plt.plot(targets[20000:30000]*80)
plt.plot(features[:, 0][20000:30000])



#%%

# make a dataframe of the normalized weights for each feature
feature_weights = pd.DataFrame(model_best.coef_)

counter = 0
ans_list = []
delay_list = []

# graph the feature weights, and get the max weight at which delay
for i in range(0, len(feature_weights.columns), 16):
    ans = feature_weights.values[0][i:i + 16]
    ans_list.append(np.max(ans))
    delay_list.append(str(16 - int(np.where(ans == np.max(ans))[0])))
    print(np.max(ans), str(16 - int(np.where(ans == np.max(ans))[0])))
    plt.plot(ans)
    plt.title(feature_names[counter])
    plt.show()
    counter = counter + 1
    
# make the list of feature + delay, for column name    
test = pd.DataFrame(index = np.arange(len(feature_names)), columns = ['feature', 'delay', 'combo'])
test['feature'][0:len(test)] = feature_names
test['delay'][0:len(test)] = delay_list
test['combo'] = test['feature'] + test['delay']
column_names= list(test['combo'])

# make dataframe 
max_df = pd.DataFrame(index = np.arange(1), columns = column_names)
counter = 0
for i in np.arange(len(feature_names)):
    max_df[column_names[counter]].values[0] = ans_list[counter]
    counter = counter + 1
    
# make normalized dataframe
new_val_list = []
for value in max_df.values[0]:
    new_val = value * (1/max_df.values[0].sum())
    new_val_list.append(new_val)
    
norm_df = pd.DataFrame(index = np.arange(1), columns = column_names)
norm_df.values[0] = new_val_list
print(norm_df)





# plot the weight of features with optimized amnt of delay

fig, ax = plt.subplots(1,1,figsize=(10,5) )
plt.plot(norm_df.values[0])
x_label_list = list(norm_df.columns)
ax.set_xticks(np.arange(len(x_label_list)))
ax.set_xticklabels(x_label_list)
plt.ylabel('weight')
plt.xlabel('feature')
plt.xticks(rotation = 90)


