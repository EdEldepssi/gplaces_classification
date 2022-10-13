{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1b34cabf-f4c4-4296-8824-850a7a1fda55",
   "metadata": {},
   "outputs": [],
   "source": [
    "import requests\n",
    "import json\n",
    "import time\n",
    "from PIL import Image\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "344f250c-0598-405d-8843-b96b29288df7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import googlemaps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0dfee73d-74f3-42e2-9148-4261a415a114",
   "metadata": {},
   "outputs": [],
   "source": [
    "url = 'http://ipinfo.io/json'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4532823c-a40e-455e-8a65-29b2db90a20c",
   "metadata": {},
   "outputs": [],
   "source": [
    "loc_json = requests.get(url)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "dec34378-1ac6-44b7-a707-751597c02106",
   "metadata": {},
   "outputs": [],
   "source": [
    "my_loc = json.loads(loc_json.content)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c89059bd-dbe9-4700-89cb-525c2b7dfc11",
   "metadata": {},
   "outputs": [],
   "source": [
    "curr_location = my_loc['loc']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f4d327c7-5c66-4f62-a53e-02d9d18e318e",
   "metadata": {},
   "outputs": [],
   "source": [
    "class GooglePlaces(object):\n",
    "    def __init__(self, apikey):\n",
    "        super(GooglePlaces, self).__init__()\n",
    "        self.apikey = apikey\n",
    "    \n",
    "    def search_places_by_coordinate(self, location, radius, types):\n",
    "        endpoint_rul = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?'\n",
    "        places = []\n",
    "        params = {\n",
    "            'location' : location,\n",
    "            'radius' : radius,\n",
    "            'type' : types,\n",
    "            'key' : self.apikey}\n",
    "        \n",
    "        res = requests.get(endpoint_rul, params = params)\n",
    "        print(res)\n",
    "        \n",
    "        # loading to the python dictionary using json.loads function\n",
    "        \n",
    "        results = json.loads(res.content)\n",
    "        # return results\n",
    "        \n",
    "        places.extend(results['results'])\n",
    "        time.sleep(2)\n",
    "        while 'next_page_token' in results:\n",
    "            params['pagetoken'] = results['next_page_token'],\n",
    "            res = requests.get(endpoint_rul, params=params)\n",
    "            results = json.loads(res.content)\n",
    "            places.extend(results['results'])\n",
    "            time.sleep(2)\n",
    "        return places\n",
    "    \n",
    "    def get_place_details(self, place_id, fields):\n",
    "        endpoint_url = \"https://maps.googleapis.com/maps/api/place/details/json?\"\n",
    "        params = {\n",
    "            'placeid': place_id,\n",
    "            'fields': \",\".join(fields),\n",
    "            'key': self.apikey\n",
    "        }\n",
    "        res = requests.get(endpoint_url, params = params)\n",
    "        place_details =  json.loads(res.content)\n",
    "        return place_details\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "60a09279-958b-493f-ac4b-6f27f23deb0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "api = GooglePlaces('AIzaSyDLTOkFGf3Kz6FhtDWLi_jUlQY1ImVrLhc')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "c05f3d10-a01d-4369-8585-2655c07a1862",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<Response [200]>\n"
     ]
    }
   ],
   "source": [
    "places = api.search_places_by_coordinate(curr_location,  \"3000\", \"restaurant\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0532f6c9-533a-4dde-b04c-64100570b7c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "fields = ['name', 'user_ratings_total', 'formatted_address','geometry', 'international_phone_number', 'website', 'rating', 'review', 'photo']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "2cc6fbe6-14b6-4a64-b7cb-eee85c326e17",
   "metadata": {},
   "outputs": [],
   "source": [
    "api_key = 'AIzaSyDLTOkFGf3Kz6FhtDWLi_jUlQY1ImVrLhc'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "4ba1179c-3901-497f-94dd-a9a4a5a12150",
   "metadata": {},
   "outputs": [],
   "source": [
    "gmaps = googlemaps.Client(key = api_key)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "f2cf5b64-ed79-49a1-b269-0d9c010ff475",
   "metadata": {
    "tags": []
   },
   "source": [
    "for place in places:\n",
    "    details = api.get_place_details(place['place_id'], fields)\n",
    "    \n",
    "    for i in range(30):\n",
    "        try:\n",
    "            photo_id = details['result']['photos'][i]['photo_reference']\n",
    "\n",
    "            # define dimentions\n",
    "            photo_width = 700\n",
    "            photo_height = 700\n",
    "\n",
    "            name = place['name']\n",
    "            raw_image_data = gmaps.places_photo(photo_reference= photo_id, max_height=photo_height,\n",
    "                                               max_width = photo_width)\n",
    "            f = open(f'./extracted_pics/{name}+{i}myimage.jpg', 'wb')\n",
    "            for each in raw_image_data:\n",
    "                if each:\n",
    "                    f.write(each)\n",
    "            f.close()\n",
    "        except:\n",
    "            continue"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "27727694-143f-46d4-8908-d0befc893650",
   "metadata": {},
   "outputs": [],
   "source": [
    "places_df = pd.DataFrame({'name':[],'address':[],'geometry':[],'phone_number':[],\n",
    "                       'website':[], 'total_user_ratings':[], 'reviews':[]})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "196cdd04-4166-4788-b17c-d9a4e0ad549c",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "scrolled": true,
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/lib/python3.9/site-packages/pandas/core/dtypes/cast.py:883: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.\n",
      "  element = np.asarray(element)\n"
     ]
    }
   ],
   "source": [
    "for place in places:\n",
    "    details = api.get_place_details(place['place_id'], fields)\n",
    "    \n",
    "    try:\n",
    "        website = details['result']['website']\n",
    "    except KeyError:\n",
    "        website = \"\"\n",
    "        \n",
    "    try:\n",
    "        name = details['result']['name']\n",
    "    except KeyError:\n",
    "        name = \"\"\n",
    "    try:\n",
    "        geometry = details['result']['geometry']['location']\n",
    "    except:\n",
    "        geometry = \"\"\n",
    "    \n",
    "    try:\n",
    "        user_ratings_total = details['result']['user_ratings_total']\n",
    "    except KeyError:\n",
    "        user_ratings_total = \"\"\n",
    "        \n",
    "    try:\n",
    "        address = details['result']['formatted_address']\n",
    "    except KeyError:\n",
    "        address = \"\"\n",
    "        \n",
    "    try:\n",
    "        phone_number = details['result']['international_phone_number']\n",
    "    except KeyError:\n",
    "        phone_number = \"\"\n",
    "        \n",
    "    try:\n",
    "        reviews = details['result']['reviews']\n",
    "    except KeyError:\n",
    "        reviews = []\n",
    "        \n",
    "    places_df.loc[len(places_df)] = [name, address, geometry, phone_number, website, user_ratings_total, reviews]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "7752b9d7-c673-49a2-90be-60210c7e7e68",
   "metadata": {},
   "outputs": [],
   "source": [
    "places_df.to_csv('./data/places_df.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "c50ae256-cf4b-4f2e-9767-feebb307f3ed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>name</th>\n",
       "      <th>address</th>\n",
       "      <th>geometry</th>\n",
       "      <th>phone_number</th>\n",
       "      <th>website</th>\n",
       "      <th>total_user_ratings</th>\n",
       "      <th>reviews</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Silver Diner</td>\n",
       "      <td>6592 Springfield Mall, Springfield, VA 22150, USA</td>\n",
       "      <td>{'lat': 38.7775425, 'lng': -77.17359239999999}</td>\n",
       "      <td>+1 703-924-1701</td>\n",
       "      <td>http://www.silverdiner.com/</td>\n",
       "      <td>2484</td>\n",
       "      <td>[{'author_name': 'Hilda Kroll', 'author_url': ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Thai Cafe</td>\n",
       "      <td>6701 Loisdale Rd E, Springfield, VA 22150, USA</td>\n",
       "      <td>{'lat': 38.7728444, 'lng': -77.17861940000002}</td>\n",
       "      <td>+1 703-922-4942</td>\n",
       "      <td>http://www.thaicafespringfield.com/</td>\n",
       "      <td>333</td>\n",
       "      <td>[{'author_name': 'Anne M.', 'author_url': 'htt...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>PaperMoon Springfield</td>\n",
       "      <td>6315 Amherst Ave, Springfield, VA 22150, USA</td>\n",
       "      <td>{'lat': 38.78111879999999, 'lng': -77.185665}</td>\n",
       "      <td>+1 703-866-4160</td>\n",
       "      <td>http://papermoonvip.com/</td>\n",
       "      <td>229</td>\n",
       "      <td>[{'author_name': 'Charles Lundgate', 'author_u...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Chick-fil-A</td>\n",
       "      <td>6681A Backlick Rd, Springfield, VA 22150, USA</td>\n",
       "      <td>{'lat': 38.773536, 'lng': -77.1830657}</td>\n",
       "      <td>+1 703-644-0155</td>\n",
       "      <td>https://www.chick-fil-a.com/springfieldinline</td>\n",
       "      <td>2344</td>\n",
       "      <td>[{'author_name': 'Winter Bray', 'author_url': ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Bertucci's Italian Restaurant</td>\n",
       "      <td>6525 Frontier Dr, Springfield, VA 22150, USA</td>\n",
       "      <td>{'lat': 38.7750714, 'lng': -77.17107779999999}</td>\n",
       "      <td>+1 703-313-6700</td>\n",
       "      <td>https://locations.bertuccis.com/us/va/springfi...</td>\n",
       "      <td>748</td>\n",
       "      <td>[{'author_name': 'Petter Llanos', 'author_url'...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                            name  \\\n",
       "0                   Silver Diner   \n",
       "1                      Thai Cafe   \n",
       "2          PaperMoon Springfield   \n",
       "3                    Chick-fil-A   \n",
       "4  Bertucci's Italian Restaurant   \n",
       "\n",
       "                                             address  \\\n",
       "0  6592 Springfield Mall, Springfield, VA 22150, USA   \n",
       "1     6701 Loisdale Rd E, Springfield, VA 22150, USA   \n",
       "2       6315 Amherst Ave, Springfield, VA 22150, USA   \n",
       "3      6681A Backlick Rd, Springfield, VA 22150, USA   \n",
       "4       6525 Frontier Dr, Springfield, VA 22150, USA   \n",
       "\n",
       "                                         geometry     phone_number  \\\n",
       "0  {'lat': 38.7775425, 'lng': -77.17359239999999}  +1 703-924-1701   \n",
       "1  {'lat': 38.7728444, 'lng': -77.17861940000002}  +1 703-922-4942   \n",
       "2   {'lat': 38.78111879999999, 'lng': -77.185665}  +1 703-866-4160   \n",
       "3          {'lat': 38.773536, 'lng': -77.1830657}  +1 703-644-0155   \n",
       "4  {'lat': 38.7750714, 'lng': -77.17107779999999}  +1 703-313-6700   \n",
       "\n",
       "                                             website total_user_ratings  \\\n",
       "0                        http://www.silverdiner.com/               2484   \n",
       "1                http://www.thaicafespringfield.com/                333   \n",
       "2                           http://papermoonvip.com/                229   \n",
       "3      https://www.chick-fil-a.com/springfieldinline               2344   \n",
       "4  https://locations.bertuccis.com/us/va/springfi...                748   \n",
       "\n",
       "                                             reviews  \n",
       "0  [{'author_name': 'Hilda Kroll', 'author_url': ...  \n",
       "1  [{'author_name': 'Anne M.', 'author_url': 'htt...  \n",
       "2  [{'author_name': 'Charles Lundgate', 'author_u...  \n",
       "3  [{'author_name': 'Winter Bray', 'author_url': ...  \n",
       "4  [{'author_name': 'Petter Llanos', 'author_url'...  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "places_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42f806ee-0423-4116-919c-9828001be0ec",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
