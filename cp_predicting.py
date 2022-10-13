#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
 
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import utils

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.model_selection import train_test_split

from keras.layers import Input, Flatten, Dense

from keras.models import Model

from tensorflow.keras.optimizers import Adam

import os

import pandas as pd

from tensorflow.keras.models import load_model

import ast


# In[2]:


# Importing cnn model
cnn_model_2 = load_model('cnn_model_2.h5')


# # Predicting Test Data:
path = os.getcwd()
# In[3]:


data_path = './extracted_pics'


# In[4]:


data_dir = os.listdir(data_path)


# In[6]:


# importing pictures and adding them into a list

img_data_list = []
image_paths = []

for img in data_dir:
    img_path = data_path + '/' +img
    try:
        img = image.load_img(img_path, target_size= (100,100))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        img_data_list.append(x)
        image_paths.append(img_path)
    except:
        continue


# In[10]:


img_data = np.array(img_data_list)


# In[11]:


np.save('./data/current_sq_data.npy', img_data)


# In[12]:


img_data = np.load('./data/current_sq_data.npy')


# In[13]:


x = img_data


# # Predicting on test Data:

# In[14]:


non_vegetarian_images = []


# In[15]:


places_images = []


# In[16]:


vegetarian_images = []


# In[17]:


for i, im in enumerate(x):
    test_img = np.expand_dims(im, axis=0)
    single_pred = cnn_model_2.predict(test_img,verbose=0)
    if np.argmax(single_pred) == 0:
        vegetarian_images.append(image_paths[i])
    elif np.argmax(single_pred)==1:
        non_vegetarian_images.append(image_paths[i])
    else:
        places_images.append(image_paths[i])


# In[18]:


places_df = pd.read_csv('./data/places_df.csv')


# In[19]:


places_df[places_df['name']=='Kana Sushi']


# In[21]:


ast.literal_eval(places_df['geometry'][0])['lat']


# In[22]:


places_df['lat'] = places_df['geometry'].apply(lambda x: ast.literal_eval(x)['lat'])


# In[23]:


places_df['lng'] = places_df['geometry'].apply(lambda x: ast.literal_eval(x)['lng'])


# In[24]:


def text_extract(txt_list):
    text = ''
    txt = ast.literal_eval(txt_list)
    for each in txt:
        text += each['text']
    return text
    


# In[25]:


places_df['text_extract'] = places_df['reviews'].apply(text_extract)


# In[26]:


# Creating vegetarian Images dataframe
images_df = pd.DataFrame({'image_path': vegetarian_images})


# In[37]:


images_df['image_path'][0].split('/')[2].split('+')[0]


# In[38]:


# Adding restaurant name to images_df
images_df['name'] = images_df['image_path'].apply(lambda x: x.split('/')[2].split('+')[0])


# In[40]:


# Merging places_df with images_df
vegetarian_places = places_df.merge(images_df, on='name')


# In[41]:


# Cleaning geometry column
vegetarian_places['geometry'] = vegetarian_places['geometry'].apply(lambda x: ast.literal_eval(x))


# In[42]:


# Cleaning reviews column
vegetarian_places['reviews'] = vegetarian_places['reviews'].apply(lambda x: ast.literal_eval(x))


# In[43]:


def review_table(name):
    
    reviews_df = pd.DataFrame({'author':[],'rating':[],'time':[] ,'text':[]})
    reviews_list = vegetarian_places[vegetarian_places['name']== name].reviews.iloc[0]
    for each in reviews_list:
        author = each['author_name']
        rating = each['rating']
        time = each['relative_time_description']
        text = each['text']
        reviews_df.loc[len(reviews_df)] = [author, rating, time, text]
    
    return reviews_df


# In[44]:


place_review = review_table('Yard House')


# In[45]:


place_review


# In[46]:


images_df

