from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
# load the Model
classifier = load_model('model.h5')
print('\n Model is Loaded \n\n')
print("**************************************************************** \n\n")

# Take input from user
import sys
entry = sys.argv[1]


##Prediction Part
import numpy as np
from keras.preprocessing import image

img_pred = image.load_img(entry, target_size = (64, 64))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
rslt = classifier.predict(img_pred)
result= pd.Series(rslt[0,:])

# import level dataset
data = pd.read_csv('files.csv')
dataset = pd.Series(data['Name'])

# Map level ith output
dataframe = pd.DataFrame()
dataframe['label'] = result
dataframe['species'] = dataset
output = dataframe[dataframe['label']==1].values
print("Bird Species Is :",output[:,-1])
print("\n\n**************************************************************** \n\n")


