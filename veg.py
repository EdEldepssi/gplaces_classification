import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

import os

import requests
import json
import time
from PIL import Image
import googlemaps
import numpy as np
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from keras.models import Model
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import ast




img = Image.open('./data/vege.png')

st.set_page_config(
    page_title="Ed Eldepssi",
    page_icon=img,
    layout="wide",
    initial_sidebar_state='collapsed',  # 'expanded' # 'auto'
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


st.title("Hello Plant-Based Lovers ❤️ ")

curr_location = st.text_input(
    "Enter Location:",
    value='lat,lng'
    )

#A class to take in location api key and return place and place details:
class GooglePlaces(object):
    def __init__(self, apikey):
        super(GooglePlaces, self).__init__()
        self.apikey = apikey
    
    def search_places_by_coordinate(self, location, radius, types):
        endpoint_rul = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?'
        places = []
        params = {
            'location' : location,
            'radius' : radius,
            'type' : types,
            'key' : self.apikey}
        
        res = requests.get(endpoint_rul, params = params)
        print(res)
        
        # loading to the python dictionary using json.loads function
        
        results = json.loads(res.content)
        #return results
        
        places.extend(results['results'])
        time.sleep(2)
        while 'next_page_token' in results:
            params['pagetoken'] = results['next_page_token'],
            res = requests.get(endpoint_rul, params=params)
            results = json.loads(res.content)
            places.extend(results['results'])
            time.sleep(2)
        return places
    
    def get_place_details(self, place_id, fields):
        endpoint_url = "https://maps.googleapis.com/maps/api/place/details/json?"
        params = {
            'placeid': place_id,
            'fields': ",".join(fields),
            'key': self.apikey
        }
        res = requests.get(endpoint_url, params = params)
        place_details =  json.loads(res.content)
        return place_details
    
api_key = st.secrets["api_key"]

api = GooglePlaces(api_key)
places = api.search_places_by_coordinate(curr_location,  "3000", "restaurant")
fields = ['name', 'user_ratings_total', 'formatted_address','geometry', 'international_phone_number', 'website', 'rating', 'review', 'photo']





gmaps = googlemaps.Client(key = api_key)

# for place in places:
#     details = api.get_place_details(place['place_id'], fields)
    
#     for i in range(30):
#         try:
#             photo_id = details['result']['photos'][i]['photo_reference']

#             # define dimentions
#             photo_width = 700
#             photo_height = 700

#             name = place['name']
#             raw_image_data = gmaps.places_photo(photo_reference= photo_id, max_height=photo_height,
#                                                max_width = photo_width)
#             f = open(f'./extracted_pics/{name}+{i}myimage.jpg', 'wb')
#             for each in raw_image_data:
#                 if each:
#                     f.write(each)
#             f.close()
#         except:
#             continue


# Extracting places informations:
places_df = pd.DataFrame({'name':[],'address':[],'geometry':[],'phone_number':[],
                       'website':[], 'total_user_ratings':[], 'reviews':[]})

for place in places:
    details = api.get_place_details(place['place_id'], fields)
    
    try:
        website = details['result']['website']
    except KeyError:
        website = ""
        
    try:
        name = details['result']['name']
    except KeyError:
        name = ""
    try:
        geometry = details['result']['geometry']['location']
    except:
        geometry = ""
    
    try:
        user_ratings_total = details['result']['user_ratings_total']
    except KeyError:
        user_ratings_total = ""
        
    try:
        address = details['result']['formatted_address']
    except KeyError:
        address = ""
        
    try:
        phone_number = details['result']['international_phone_number']
    except KeyError:
        phone_number = ""
        
    try:
        reviews = details['result']['reviews']
    except KeyError:
        reviews = []
        
    places_df.loc[len(places_df)] = [name, address, geometry, phone_number, website, user_ratings_total, reviews]


places_df.to_csv('./data/place9s_df.csv', index=False)


df = pd.read_csv('./data/place9s_df.csv')


st.dataframe(df)

# Importing cnn model
cnn_model_2 = load_model('resnet.h5')


# # Predicting Test Data:

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
        img = image.load_img(img_path, target_size= (224,224))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        img_data_list.append(x)
        image_paths.append(img_path)
    except:
        continue

st.write(image_paths)
# In[10]:
# img_data = np.array(img_data_list)
x = np.array(img_data_list)


# In[11]:
# np.save('./data/current_sq_data.npy', img_data)

# # In[12]:
# img_data = np.load('./data/current_sq_data.npy')


# # In[13]:
# x = img_data


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
images_df0 = pd.DataFrame({'image_path': vegetarian_images})


# In[37]:
# images_df0['image_path'][0].split('/')[2].split('+')[0]


# In[38]:
# Adding restaurant name to images_df
images_df0['name'] = images_df0['image_path'].apply(lambda x: x.split('/')[2].split('+')[0])


# In[40]:
# Merging places_df with images_df
vegetarian_places = places_df.merge(images_df0, on='name')


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

# Creating non_vegetarian Images dataframe
images_df1 = pd.DataFrame({'image_path': non_vegetarian_images})
images_df1['name'] = images_df1['image_path'].apply(lambda x: x.split('/')[2].split('+')[0])

# Creating places Images dataframe
images_df2 = pd.DataFrame({'image_path': places_images})
images_df2['name'] = images_df2['image_path'].apply(lambda x: x.split('/')[2].split('+')[0])


# In[44]:
# place_review = review_table('Yard House')


# In[45]:
# st.dataframe(place_review)
# place_review
# In[46]:

# st.dataframe(images_df0)

st.dataframe(images_df0)

st.title("Classified Plant-Based Items ❤️")

st.image(images_df0['image_path'].values.tolist(), width=200, 
	caption = images_df0['name'].values.tolist())



st.title("Classified Non-Plant-Based Items ❤️")

st.image(images_df1['image_path'].values.tolist(), width=200, 
	caption = images_df1['name'].values.tolist())


st.title("Classified Places Images ❤️")

st.image(images_df2['image_path'].values.tolist(), width=200, 
	caption = images_df2['name'].values.tolist())


# for each in images_df['image_path']:
# 	st.image(each, width=200)





