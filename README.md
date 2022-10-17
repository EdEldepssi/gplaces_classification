## Problem Statement:

The number of people following a plant-based/vegetarian diet is increasing each year all over the world. The problem that most people new to vegetarian/plant-based diet face is the limited options of restaurants that offers vegetarian meals on their menus.

My idea is to develop a model that accepts a geographical location and quickly scans the restaurant around that location, extracts photos from google places any photos posted by the restaurant or the users, run a cnn model to detect the vegetarian meals in these photos, and reports back with the photos of the vegetarian meals found around the area along with restaurant information.

### Who would be interested in this project?
people who are interested in following a plant-based diet and struggle to find places that serves more vegetarian items on their menus.


### Models used:
	- resent50 model
	- sequential model

### Metrics of Success:
	- accuracy
	- loss by categorical crossentropy
	- precision




### Table of Contents:

### cp_google-places.ipynb:
	Extracting photos of google places using google places API and putting them into a dataframe

### renset_model.ipynb
	Modeling on training data and testing on testing data using a resnet50 model.

### cp_sequential_model.ipynb:
	Modeling on training data and testing on testing data using a sequential model

### resent.h5:
	The saved model form rasnet_model.ipynb

### cp_predicting.ipynb:
	Using resent.h5 to predict on newly extracted data, putting them into a data frame along with restaurant name, location and other place information

### veg.py:
	The code to run in the entire process a streamlit app.

### data:
	All data extracted by the model, including dataframe of the places information and other .npy files for the extracted photos data.
### testing:
	Testing data

### training:
	Training data



### Findings:
Out of all models I have used, I found that resent50 model is performing much better with an accuracy score of 84% that sequential model with accuracy of 62 %

I was able to achieve 84% accuracy with my resent50 model after I decided to train my model on 7 different classes [ vegetarian, non-vegetarian, places, drinks, deserts, ads, clutter ]


