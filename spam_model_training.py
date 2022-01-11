'''
===== Learning text classification using BERT =====
This tutorial comes from https://www.youtube.com/watch?v=D9yyt6BfgAM
code files from https://github.com/codebasics/deep-learning-keras-tf-tutorial/tree/master/47_BERT_text_classification

Pre trained BERT model comes from https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3
Pre processor comes from https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3

You can download the trained BERT model and preprocesor, UNZIP the files on folder 'TrainedModels'
'''
from os import name
from numpy import dtype
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import shape
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd


#Read spam file to train machine
dframe = pd.read_csv('spam.csv')

#Create a classifier column that has boolean value if email is SPAM
#emailCategory represents the 'Category' column
#Basically, if the email category  = spam, then assign 1, else 0 
dframe['spam'] = dframe['Category'].apply(
    lambda emailCategory:1 if emailCategory=='spam' else 0)

# print(dframe.head(5)) #read the first 5 rows of data.

# ===== Begin to train data.
from sklearn.model_selection import train_test_split

'''
test_size = 0.2, it means that 80% is TRAINING SAMPLES and 20% TEST SAMPLES
    of the passes dframe data.

stratify: this parameter makes a split so that the proportion of values in
    the sample produced will be the same as the proportion of values 
    provided to parameter stratify.

    For example, if variable y is a binary categorical variable with 
    values 0 and 1 and there are 25% of zeros and 75% of ones, 
    stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.
'''
x_train, x_test,y_train,  y_test =  train_test_split(dframe['Message'],
    dframe['spam'],test_size=0.2, stratify=dframe['spam'])

#Check for balance between train and test data results
# print(y_train.value_counts() )
# print(y_test.value_counts() )

#BERT Model location
print('loading keras processors')
main_encoder = "TrainedModels\\bert_en_uncased_L-12_H-768_A-12_3"
preprocesor = "TrainedModels\\bert_en_uncased_preprocess_3"

bert_preprocessor = hub.KerasLayer(preprocesor)
bert_encoder = hub.KerasLayer(main_encoder)

# #Helper functions for TEST
# def get_sentence_embedding(sentences):
#     preprocessed_text = bert_preprocessor(sentences)
#     #Print the vectors for the processed text
#     print(bert_encoder(preprocessed_text)['pooled_output'])


# get_sentence_embedding(['$5000 discount, hurry up for best deals',
#     'Are you up for a basketball game tomorrow?'])


#Create keras layers
text_input = tf.keras.layers.Input(shape=(),dtype=tf.string,name='text_layer')
preprocessed_text = bert_preprocessor(text_input)
outputs = bert_encoder(preprocessed_text)

tflayer = tf.keras.layers.Dropout(0.1,name='dropout')(outputs['pooled_output'])
tflayer =  tf.keras.layers.Dense(1,activation='sigmoid', name='output')(tflayer)

model = tf.keras.Model(inputs=[text_input],outputs=[tflayer])
print(model.summary())

# ===== TRAINING MODEL

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)

# ===== SAVE MODEL
model.save('Processed_Model',overwrite=True)

# ==== EVALUATING MODEL
model.evaluate(x_test,y_test)


# ===== REAL TEST!!
# Test some text and see if they are classified as spam or not spam

emails = ['07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow'
    'K fyi x has a ride early tomorrow morning but he\'s crashing at our place tonight',
    'As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a Â£1500 Bonus Prize, call 09066364589']

'''
The higher the value, means that those emals are SPAM
In 'sigmoid' evaluation, values of 0.5 + means a TRUE possibility to be 
    classified as SPAM (or whatever other classifer)
'''
print(model.predict(emails))