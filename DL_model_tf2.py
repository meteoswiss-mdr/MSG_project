#!/usr/bin/env python
# coding: utf-8

# # U-NET model

# This model uses the unet package cloned from https://github.com/jakeret/unet

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import os
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
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# ## Some global variables

# In[2]:


datapath = '/data/ml_course/05_Capstone_project/dl_data/'
modelpath = './unet_ld3_fr16'
callbackpath = './unet_ld3_fr16_callbacks'

# Number of files used for training, validation and testing
nfiles_te = 4
nfiles_va = 4
nfiles_tr = 36

# Model architecture
layer_depth = 3
filters_root = 16

# parameters for training and evaluation
train_batch_size = 4
pred_batch_size = 4
epochs = 3


# ## Auxiliary functions

# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


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

# In[8]:


flist = glob.glob(datapath+'*_data.npz')
flist.sort()
np.random.shuffle(flist)
print('Number of input files:', len(flist))


# In[9]:


flist_te = flist[0:nfiles_te]
flist_va = flist[nfiles_te:nfiles_te+nfiles_va]
flist_tr = flist[nfiles_te+nfiles_va:nfiles_te+nfiles_va+nfiles_tr]

print('Number of test files:', len(flist_te))
print('Number of validation files:', len(flist_va))
print('Number of training files:', len(flist_tr))


# In[10]:


train_dataset = get_dataset(flist_tr, stop_at_end=False)
validation_dataset = get_dataset(flist_va, stop_at_end=False)
test_dataset = get_dataset(flist_te, stop_at_end=True) # Put to stop at end because there is no stopping mechanism in the u-net evaluation


# ## Compute class weights

# In[11]:


total, nel_class = examine_data(flist_tr)
print('Total number of pixels in training dataset:', total)
print('Number of no hail pixels:', nel_class[0])
print('Number of hail pixels:', nel_class[1])
print('% o hail pixels over total:', 100*nel_class[1]/total)


# In[12]:


# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_nohail = (1 / nel_class[0])*(total)/2.0 
weight_hail = (1 / nel_class[1])*(total)/2.0

class_weight = {0: weight_nohail, 1: weight_hail}

print('Weight for no hail: {:.2f}'.format(weight_nohail))
print('Weight for hail: {:.2f}'.format(weight_hail))


# ## Create model

# In[13]:


unet_model = unet.build_model(channels=3,
                              num_classes=2,
                              layer_depth=layer_depth,
                              filters_root=filters_root)


# In[14]:


metrics = [
    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
    tf.keras.metrics.FalseNegatives(name='fn', thresholds=0.5),
    tf.keras.metrics.FalsePositives(name='fp', thresholds=0.5),
    tf.keras.metrics.TrueNegatives(name='tn', thresholds=0.5),
    tf.keras.metrics.TruePositives(name='tp', thresholds=0.5)
]

loss = weighted_binary_crossentropy(weight_nohail, weight_hail)
# loss = tf.keras.losses.BinaryCrossentropy()

unet.finalize_model(
    unet_model, loss=loss,
    metrics=metrics,
    dice_coefficient=True,
    auc=False,
    mean_iou=False)


# ## Train model

# In[15]:


fit_kwargs = {
    'steps_per_epoch': int(nfiles_tr/train_batch_size),
    'validation_steps': int(nfiles_va/train_batch_size)}

trainer = unet.Trainer(log_dir_path=callbackpath, checkpoint_callback=False)
history = trainer.fit(unet_model,
            train_dataset,
            validation_dataset,
            epochs=epochs,
            batch_size=train_batch_size,
            **fit_kwargs)


# In[16]:


history.history.keys()


# In[17]:


# Create two plots: one for the loss value, one for the accuracy
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

# Plot accuracy values
ax1.plot(history.history['loss'], label='train loss')
ax1.plot(history.history['val_loss'], label='val loss')
ax1.set_title('Val. loss {:.3f} (mean last 3)'.format(
    np.mean(history.history['val_loss'][-3:]) # last three values
))
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss value')
ax1.legend()

# Plot accuracy values
ax2.plot(history.history['binary_accuracy'], label='train acc')
ax2.plot(history.history['val_binary_accuracy'], label='val acc')
ax2.set_title('Val. accuracy {:.3f} (mean last 3)'.format(
    np.mean(history.history['val_binary_accuracy'][-3:]) # last three values
))
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.legend()

# Plot accuracy values
ax3.plot(history.history['dice_coefficient'], label='train acc')
ax3.plot(history.history['val_dice_coefficient'], label='val acc')
ax3.set_title('Val. dice coeff {:.3f} (mean last 3)'.format(
    np.mean(history.history['val_dice_coefficient'][-3:]) # last three values
))
ax3.set_xlabel('epoch')
ax3.set_ylabel('accuracy')
ax3.legend()

plt.show()


# ## Save trained model

# In[18]:


unet_model.save(modelpath)


# ## Make predictions

# In[19]:


prediction = unet_model.predict(test_dataset.batch(pred_batch_size), verbose=1, steps=int(nfiles_te/pred_batch_size))


# In[20]:


dataset = test_dataset.map(utils.crop_image_and_label_to_shape(prediction.shape[1:]))


# In[21]:


fig, ax = plt.subplots(4, 5, figsize=(20,10))
for i, (image, label) in enumerate(dataset.take(4)):
    hrv = image.numpy()[:, :, 0]
    ir = image.numpy()[:, :, 1]
    diff = image.numpy()[:, :, 2]
    
    lab = np.argmax(label, axis=-1)
    pred = np.argmax(prediction[i,...], axis=-1)
        
    ax[i][0].matshow(hrv.T[::-1, :], cmap=plt.cm.gray)
    ax[i][0].axis('off')
    ax[0][0].set_title('HRV')
    
    ax[i][1].matshow(ir.T[::-1, :], cmap=plt.cm.gray_r)
    ax[i][1].axis('off')
    ax[0][1].set_title('IR_108')
    
    ax[i][2].matshow(diff.T[::-1, :], cmap=plt.cm.gray)
    ax[i][2].axis('off')
    ax[0][2].set_title('WV_062-IR_108')
    
    ax[i][3].matshow(lab.T[::-1, :], cmap=plt.cm.gray)
    ax[i][3].axis('off')
    ax[0][3].set_title('POH90')
    
    ax[i][4].matshow(pred.T[::-1, :], cmap=plt.cm.gray)
    ax[i][4].axis('off')
    ax[0][4].set_title('Predicted Hail')
    
plt.tight_layout()


# ## Evaluate model

# In[22]:


tn = 0
fp = 0
fn = 0
tp = 0

positive = 0
negative = 0
for i, (image, label) in enumerate(dataset.take(-1)):
    lab = np.argmax(label, axis=-1)
    pred = np.argmax(prediction[i,...], axis=-1)
    
    tn_aux, fp_aux, fn_aux, tp_aux = confusion_matrix(lab.flatten(), pred.flatten()).ravel()
    
    tn += tn_aux
    fp += fp_aux
    fn += fn_aux
    tp += tp_aux
    
    positive += lab[lab == 1].size
    negative += lab[lab == 0].size
    
print('True positive: ', tp)
print('True negative: ', tn)
print('False positive: ', fp)
print('False negative: ', fn)
print('Positive pixels: ', positive)
print('Negative pixels: ', negative)


# In[23]:


pod = 100*tp/(tp+fn)
far = 100*fp/(fp+tn)
ppv = tp/(tp+fp)
print('Probability of Detection (POD):', pod)
print('False Alarm Rate (FAR):', far)
print('Positive Predictive Value (PPV):', ppv)


# In[24]:


## This is not working properly
# trainer.evaluate(unet_model, test_dataset, shape=prediction.shape[1:])


# In[25]:


## This is not working properly
# X, y = read_data(flist_te[0])
# 
# X = utils.crop_to_shape(X, prediction.shape[1:])
# y = utils.crop_to_shape(y, prediction.shape[1:])
# 
# X = X[np.newaxis, :, :, :]
# y = y[np.newaxis, :, :, :]
# 
# unet_model.evaluate(x=(X, y), batch_size=1, verbose=1, return_dict=True)


# In[ ]:




