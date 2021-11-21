#!/usr/bin/env python
# coding: utf-8

# # Minimal example with TensorFlow 2.0
# In this notebook we will recreate our machine learning algorithm using TF 2.0.

# ## Import the relevant libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ## Data generation

# In[2]:


# we specify the size of the training set and we generate some random input data:

observations = 1000

xs = np.random.uniform(low=-10, high=10, size=(observations,1))
zs = np.random.uniform(-10, 10, (observations,1))

# we put our inputs into a two column stack:
# This is the X matrix from the linear model y = x*w + b.
generated_inputs = np.column_stack((xs,zs))

# noise: random[-1,1]:
noise = np.random.uniform(-1, 1, (observations,1))

# Supervised parameters:
generated_targets = 2*xs - 3*zs + 5 + noise

# .npz files; it is NumPy’s file type. Stores n-dimensional arrays, aka. Tensors
np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)


# ## Solving with TensorFlow

# In[3]:


# It is better that we load the saved data
training_data = np.load('TF_intro.npz')


# In[4]:


# Using TF we must specify input and output size:
input_size = 2
output_size = 1

# We specify our model with "tf.keras.Sequential"
# takes the inputs provided by the model and calculates the dot product of the inputs and ...
# ... the weights and adds the bias [*also applies activation function (optional)]
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(output_size,
                                                 kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                                 bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
                                                 )
                            ])

# [optional], we can specify the optimizer (optimization algorithm)
custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)

# we must specify our Objective Function = Loss Function
# model.compile(optimizer, loss): configures the model for training
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

# Indicate to the model which data to fit
# Each iteration over the full data set in machine learning is called an Epoch.
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2)


# ## Extract the weights and bias

# In[5]:


model.layers[0].get_weights()


# In[6]:


weights = model.layers[0].get_weights()[0]
weights


# In[7]:


bias = model.layers[0].get_weights()[1]
bias


# ## Extract the outputs (make predictions)

# In[16]:


model.predict_on_batch(training_data['inputs']).round(1)


# In[10]:


training_data['targets'].round(1)


# ## Plotting the data

# In[17]:


plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()


# In[ ]:




