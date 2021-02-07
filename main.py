import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB

st.title("Dasboard")

@st.cache
def load_data(classifier_name):
   data = pd.read_csv('LAUNDRY.csv')
   del data['No']
   data.dropna(axis=0, subset=['Age_Range'], inplace=True)
   data['Gender'].fillna('male', inplace = True)
   data['Race'].fillna('Other', inplace = True)
   data['Body_Size'].fillna('Other', inplace = True)
   data['With_Kids'].fillna('no', inplace = True)
   data['Kids_Category'].fillna('Other', inplace = True)
   data['Basket_Size'].fillna('Other', inplace = True)
   data['Basket_colour'].fillna('Other', inplace = True)
   data['Shirt_Colour'].fillna('Other', inplace = True)
   data['shirt_type'].fillna('Other', inplace = True)
   data['Pants_Colour'].fillna('Other', inplace = True)
   data['pants_type'].fillna('Other', inplace = True)
   data['Wash_Item'].fillna('Other', inplace = True)
   data['Attire'].fillna('Other', inplace = True)

  
   return data


# selecting classifier 
classifier_name = st.sidebar.selectbox("Select Classifier",("Naive Bayes","K-Nearest Neighbors","Random Forest"))

df = load_data(classifier_name)
df_fs = df.copy()
del df_fs['Time']
del df_fs['Date']
st.write(df_fs.head(5))
st.write("Shape of dataset:", df_fs.shape)

#simple visualization
fig = plt.figure()
b=sns.countplot(x='Washer_No', data = df_fs)
plt.title('Number of Washer ')
st.pyplot(fig)

fig = plt.figure()
df_fs.Age_Range.hist()
plt.title('Histogram of Age Range')
plt.xlabel('Age')
plt.ylabel('Frequency') 
st.pyplot(fig)

fig = plt.figure()
sns.distplot(df_fs["Age_Range"], bins=10)
st.write("Skewness",df_fs["Age_Range"].skew()) 
st.pyplot(fig)

fig = plt.figure()
fig, ax = plt.subplots(figsize=(11.7, 8.27))
sns.boxplot(data=df_fs, x="Washer_No", y="Age_Range", ax=ax)
plt.title('Boxplot of People Age Range Using Washer Category ')
st.pyplot(fig)

fig = plt.figure()
fig, ax = plt.subplots(figsize=(11.7, 8.27))
sns.boxplot(data=df_fs, x="Gender", y="Age_Range", ax=ax)
plt.title('Boxplot of Gender With Age Range')
st.pyplot(fig)

st.write("Statistical details:")
st.write(df_fs.describe())



categorical = ['Race','Gender','Body_Size', 'With_Kids','Kids_Category','Basket_Size','Basket_colour','Attire','Shirt_Colour','shirt_type','Pants_Colour','pants_type','Wash_Item','Spectacles']
d = defaultdict(LabelEncoder) 
df_fs[categorical] = df_fs[categorical].apply(lambda x: d[x.name].fit_transform(x.astype(str)))

class_variable = st.selectbox("Class Variables",("Gender","Spectacles"))
if(class_variable == "Gender"):
   X = df_fs.drop('Gender', axis=1) 
   y = df_fs['Gender']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=10)
elif (class_variable == "Spectacles"):
   X = df_fs.drop('Spectacles', axis=1) 
   y = df_fs['Spectacles']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=10)
else:
   #default Gender
   X = df_fs.drop('Gender', axis=1) 
   y = df_fs['Gender']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=10)

if(classifier_name == "K-Nearest Neighbors"):
   st.title(classifier_name)
   
   #Classification-KNN
   k=11
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(X_train,y_train)
   st.write('KNN Score= {:.2f}'.format(knn.score(X_test, y_test)))

   y_pred = knn.predict(X_test)
   confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)
   st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
   st.write('Recall= {:.2f}'.format(recall_score(y_test, y_pred, pos_label=0)))
   st.write('F1= {:.2f}'.format(f1_score(y_test, y_pred, pos_label=0)))
   st.write('Accuracy= {:.2f}'.format(accuracy_score(y_test, y_pred)))

   st.title("Class Variable:" + class_variable)

   #plot knn
   prob_KNN = knn.predict_proba(X_test)
   prob_KNN = prob_KNN[:, 1]
   fpr_knn, tpr_knn, thresholds_DT = roc_curve(y_test, prob_KNN) 

   fig = plt.figure()
   plt.plot(fpr_knn, tpr_knn, color='red', label='KNN') 
   plt.plot([0, 1], [0, 1], color='green', linestyle='--')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('K-Nearest Neighbors (KNN) Curve')
   plt.legend()

   st.pyplot(fig)

elif(classifier_name == "Naive Bayes"):
   st.title(classifier_name)

   #Classification-NB
   nb = GaussianNB()
   nb.fit(X_train, y_train)
   y_pred = nb.predict(X_test)

   confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)
   st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
   st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
   st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
   st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
   st.title("Class Variable:" + class_variable)
   #plot NB
   prob_NB = nb.predict_proba(X_test)
   prob_NB = prob_NB[:, 1]
   fpr_NB, tpr_NB, thresholds_DT = roc_curve(y_test, prob_NB) 

   fig = plt.figure()
   plt.plot(fpr_NB, tpr_NB, color='orange', label='NB')
   plt.plot([0, 1], [0, 1], color='green', linestyle='--')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Naive Bayes (NB) Curve')
   plt.legend()

   st.pyplot(fig)

else:
   st.title(classifier_name)

   rf = RandomForestClassifier(random_state=10)
   rf.fit(X_train, y_train)
   y_pred = rf.predict(X_test)
   confusion_majority=confusion_matrix(y_test, y_pred)

   confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)

   st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
   st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
   st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
   st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
   st.title("Class Variable:" + class_variable)
   #plot RF
   fig = plt.figure()
   prob_RF = rf.predict_proba(X_test)
   prob_RF = prob_RF[:, 1]
   fpr_rf, tpr_rf, thresholds_DT = roc_curve(y_test, prob_RF)
   plt.plot(fpr_rf, tpr_rf, color='blue', label='RF') 
   plt.plot([0, 1], [0, 1], color='green', linestyle='--')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic (ROC) Curve')
   plt.legend()

   st.pyplot(fig)
