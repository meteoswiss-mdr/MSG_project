#!/usr/bin/env python
# coding: utf-8

# # U-NET model training

# This model uses the unet package cloned from https://github.com/jakeret/unet. This notebook is for illustration purposes only. Since the processing time is very large the actual training is going to be performed using python scripts so that it can be trained in background.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import unet
from unet import utils
from unet.datasets import circles
from copy import deepcopy
from sklearn.metrics import confusion_matrix

import inspect

print("TF version:", tf.__version__)
#print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# ## Some global variables

# In[ ]:


# datapath = '/data/ml_course/05_Capstone_project/dl_data/'  # zueub242
datapath = '/scratch/fvj/dl_data/'  # Payerne workstation
modelpath = './unet_ld3_fr64_udw'
callbackpath = './unet_ld3_fr64_udw_callbacks'
flist_name = 'unet_ld3_fr64_udw_flist.npz'
hist_name = 'unet_ld3_fr64_udw_history.pickle'

# Number of files used for training, validation and testing
nfiles_te = 351
nfiles_va = 400
nfiles_tr = 3600

# Model architecture
layer_depth = 3
filters_root = 64

# parameters for training and evaluation
train_batch_size = 32
pred_batch_size = 32
epochs = 100
patience = 3

# weights can be obtained from class frequency or given by the user
weight_nohail = (1/9)*10/2
weight_hail = (1/1)*10/2


# ## Auxiliary functions

# In[ ]:


def read_data(fname):
    """
    Reads features and labels stored in npz files
    
    Parameters
    ----------
    fname : str
        path of the file containing the data
        
    Returns
    -------
    X : float array 
        features matrix of size nx, ny, nchannels
    y : float array
        lables matrix of size, nx, ny, nclasses
    
    """
    with np.load(fname, allow_pickle=False) as npz_file:
        # Load the arrays
        X = npz_file['features']
        y = npz_file['targets']
    return X, y


# In[ ]:


def data_generator(file_list, stop_at_end=False):
    """
    data generator
    
    Parameters
    ----------
    file_list : list of str
        lists of files where the data is stored
    stop_at_end : bool
        Controls the behaviour when running out of files.
        If True exits the function. Otherwise reshuffles the list
        and sets the counter to 0
    
    Yield
    -------
    X : float array 
        features matrix of size nx, ny, nchannels
    y : float array
        lables matrix of size, nx, ny, nclasses
    
    """
    i = 0
    while True:
        if i >= len(file_list):
            if stop_at_end:
                break
            i = 0
            np.random.shuffle(file_list)            
        else:
            X, y = read_data(file_list[i])            
            yield X, y
            i = i + 1


# In[ ]:


def get_dataset(flist, stop_at_end=False):
    """
    Creates a tensorflow dataset from a generator
    
    Parameters
    ----------
    file_list : list of str
        lists of files where the data is stored
    stop_at_end : bool
        Controls the behaviour when running out of files.
        If True exits the function. Otherwise reshuffles the list
        and sets the counter to 0
    
    Returns
    -------
    dataset : tf.data.Dataset
        A dataset containing the features and labels
    
    """
    X, y = read_data(flist[0])
    nx = X.shape[0]
    ny = X.shape[1]
    nchannels = X.shape[2]
    nclasses = y.shape[2]
    return tf.data.Dataset.from_generator(
        data_generator, args=[flist, stop_at_end], output_types=(tf.float32, tf.float32),
        output_shapes = ((nx, ny, nchannels), (nx, ny, nclasses)))


# In[ ]:


def examine_data(flist):
    """
    Get the number of pixels corresponding to each class and the total number
    of pixels in a dataset
    
    Parameters
    ----------
    flist : list of str
        lists of files where the data is stored
    
    Returns
    -------
    total : int
        The total number of pixels in the dataset
    nel_class : array of ints
        The number of pixels for each class
    
    """
    _, y = read_data(flist[0])    
    nclasses = y.shape[2]
                 
    total = 0
    nel_class = np.zeros(nclasses, dtype=np.int)
    for i, fname in enumerate(flist):
        _, y = read_data(fname)
        total += int(y.size/nclasses)
        
        for j in range (nclasses):
            ind = np.where(y[:, :, j] == 1)[0]
            nel_class[j] += ind.size
    return total, nel_class


# In[ ]:


def weighted_binary_crossentropy(zero_weight, one_weight):
    """
    Computes the weighted binary crossentropy
    
    Parameters
    ----------
    zero_weight, one_weight : float
        The weights for each class
    
    Returns
    -------
    weigthed_bce : tensor flow array
        The weighted loss at each pixel
    
    """
    def wbce(y_true, y_pred):
        # Calculate binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        ## Apply weights
        index = tf.cast(tf.argmax(y_true, axis=-1), tf.float32) # from one hot to indices
        weight_vector = index*one_weight+(1.-index)*zero_weight
        weigthed_bce = weight_vector*bce
        
        return weigthed_bce
    return wbce


# ## Load data

# In[ ]:


flist = glob.glob(datapath+'*_data.npz')
flist.sort()
np.random.shuffle(flist)
print('Number of input files:', len(flist))


# In[ ]:


flist_te = flist[0:nfiles_te]
flist_va = flist[nfiles_te:nfiles_te+nfiles_va]
flist_tr = flist[nfiles_te+nfiles_va:nfiles_te+nfiles_va+nfiles_tr]

print('Number of test files:', len(flist_te))
print('Number of validation files:', len(flist_va))
print('Number of training files:', len(flist_tr))


# In[ ]:


# Save file names for each category:
np.savez(flist_name, flist_te=flist_te, flist_va=flist_va, flist_tr=flist_tr)


# In[ ]:


train_dataset = get_dataset(flist_tr, stop_at_end=False)
validation_dataset = get_dataset(flist_va, stop_at_end=False)
test_dataset = get_dataset(flist_te, stop_at_end=True) # Put to stop at end because there is no stopping mechanism in the u-net evaluation


# ## Check number of pixels in each class

# In[ ]:


total, nel_class = examine_data(flist_tr)
print('Total number of pixels in training dataset:', total)
print('Number of no hail pixels:', nel_class[0])
print('Number of hail pixels:', nel_class[1])
print('% of hail pixels over total:', 100*nel_class[1]/total)


# ## Compute class weights

# In[ ]:


if weight_nohail is None:
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_nohail = (1 / nel_class[0])*(total)/2.0 
    weight_hail = (1 / nel_class[1])*(total)/2.0
    
print('Weight for no hail: {:.2f}'.format(weight_nohail))
print('Weight for hail: {:.2f}'.format(weight_hail))


# ## Create model

# In[ ]:


unet_model = unet.build_model(channels=3,
                              num_classes=2,
                              layer_depth=layer_depth,
                              filters_root=filters_root)


# In[ ]:


metrics = [
    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
#    tf.keras.metrics.FalseNegatives(name='fn', thresholds=0.5),
#    tf.keras.metrics.FalsePositives(name='fp', thresholds=0.5),
#    tf.keras.metrics.TrueNegatives(name='tn', thresholds=0.5),
#    tf.keras.metrics.TruePositives(name='tp', thresholds=0.5)
]

loss = weighted_binary_crossentropy(weight_nohail, weight_hail)
# loss = tf.keras.losses.BinaryCrossentropy()

unet.finalize_model(
    unet_model, loss=loss,
    metrics=metrics,
    dice_coefficient=False,
    auc=False,
    mean_iou=False)


# ## Train model

# In[ ]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

fit_kwargs = {
    'steps_per_epoch': int(nfiles_tr/train_batch_size),
    'validation_steps': int(nfiles_va/train_batch_size)}

trainer = unet.Trainer(log_dir_path=callbackpath, checkpoint_callback=False, callbacks=[early_stopping])
history = trainer.fit(unet_model,
            train_dataset,
            validation_dataset,
            epochs=epochs,
            batch_size=train_batch_size,
            **fit_kwargs)


# ## Save trained model

# In[ ]:


# save history
with open(hist_name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


# In[ ]:


unet_model.save(modelpath)

