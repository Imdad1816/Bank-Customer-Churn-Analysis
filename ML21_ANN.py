#importing libraries
import numpy as np
import tensorflow as tf
import pandas as pd 

#importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values


####Encoding the categorical Data

#Label encoding of the gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

#Onehot encoding of the Geography column
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x = np.array(ct.fit_transform(x))


##splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

##Building the ANN
#Initializing the ANN
ann = tf.keras.models.Sequential()
#Adding input and first hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
#Adding second hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

##Training the ANN
#Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'] )
#Training the ANN
ann.fit(x_train,y_train,batch_size=32,epochs=100)

#Making the prediction and evaluating the model
print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])) > 0.5)

#predicting the test set results
y_pred = ann.predict(x_test)
y_pred = (y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#predicting the confusing matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
