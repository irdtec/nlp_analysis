import pandas as pd
import tensorflow as tf
from tensorflow import keras
 #for some reason, this library is needed, even if it's not used
import tensorflow_text as text

emails = pd.read_csv("spam.csv")

# Load trained model
print('\r\nLOADING SAVED MODEL')
model = keras.models.load_model('Processed_Model')

print('\r\nMODEL LOADED, EVALUATING EMAIL TEXT')
print(model.predict(emails['Message']))
# counter = 0
# while counter < len(emails):
#     temp_array = [emails.iloc[counter]['Message']]
#     print(str(model.predict(temp_array)) + " -->" + emails.iloc[counter]['Category'])    
#     counter = counter+1


    