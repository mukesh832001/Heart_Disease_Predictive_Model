import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset
df=pd.read_csv("C:/Users/Mukesh/OneDrive/Desktop/HDPM/heart.csv")
df.info()

#Feature Engineering
#Renaming Atrribute Names
df=df.rename(columns={"age":"Age","sex":"Sex","cp":"Chest Pain","trestbps":"resting bps","chol":"cholestral","fbs":"fasting bps"
                ,"restecg":"resting electrocardiographic results","thalach":"maximum heartrate","exang":"exercise induced angina",})

#Checking Correlation Features
describe = df.describe()
corr_matrix = df.corr()
c = corr_matrix.iloc[:,-1].values
c = pd.DataFrame(c)

print(df['target'].value_counts())

#Checking Null Values
df_null = df.isnull().sum()

#Visualizing BarPlot
print(df['target'].value_counts().plot(kind="bar",color=['red','blue']))

#Visualisig HeatMap
plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix,annot=True,fmt=".2f")
print(df.groupby('Age')['target'].value_counts())
df['target'].plot(kind="hist",orientation="horizontal")

#Visualising the Relation Between Age and Target Attributes
plt.xlabel("Age")
plt.ylabel("Heart attack(1/0)")
plt.figure(figsize=(30,20))

#Visualising Individual Columns
plt.subplot(3,3,1)
sns.countplot(df['Sex'])

plt.subplot(3,3,2)
sns.countplot(df['Chest Pain'])

plt.subplot(3,3,3)
sns.countplot(df['resting electrocardiographic results'])

plt.subplot(3,3,4)
sns.countplot(df['slope'])

plt.subplot(3,3,5)
sns.countplot(df['ca'])

plt.subplot(3,3,6)
sns.countplot(df['thal'])

plt.subplot(3,3,7)
sns.countplot(df['target'])

plt.show()

y=df['target']
x=df.iloc[:,:-1].values


#Splitting into Train data and Test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

#fitting ml models
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
        
#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
lo_re=LogisticRegression()
lo_re.fit(x_train,y_train)
print_score(lo_re,x_train,y_train,x_test,y_test,train=True)
print_score(lo_re,x_train,y_train,x_test,y_test,train=False)


test_score = accuracy_score(y_test, lo_re.predict(x_test)) * 100
train_score = accuracy_score(y_train, lo_re.predict(x_train)) * 100
results_df = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

#KNeighbors Classifier Model
from sklearn.neighbors import KNeighborsClassifier
kn_cl = KNeighborsClassifier()
kn_cl.fit(x_train, y_train)
print_score(lo_re, x_train, y_train, x_test, y_test, train=True)
print_score(lo_re, x_train, y_train, x_test, y_test, train=False)


test_score = accuracy_score(y_test, kn_cl.predict(x_test)) * 100
train_score = accuracy_score(y_train, kn_cl.predict(x_train)) * 100
results_df_2 = pd.DataFrame(data=[["K-nearest neighbors", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

#Support Vector Machine Model
from sklearn.svm import SVC
sc_cl = SVC(kernel="rbf")
sc_cl.fit(x_train, y_train)
print_score(lo_re, x_train, y_train, x_test, y_test, train=True)
print_score(lo_re, x_train, y_train, x_test, y_test, train=False)


test_score = accuracy_score(y_test, sc_cl.predict(x_test)) * 100
train_score = accuracy_score(y_train, sc_cl.predict(x_train)) * 100

results_df_2 = pd.DataFrame(data=[["Support Vector Machine", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

#Decision Tree Classifier Model
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=0)
tree_clf.fit(x_train, y_train)
print_score(tree_clf, x_train, y_train, x_test, y_test, train=True)
print_score(tree_clf, x_train, y_train, x_test, y_test, train=False)


test_score = accuracy_score(y_test, tree_clf.predict(x_test)) * 100
train_score = accuracy_score(y_train, tree_clf.predict(x_train)) * 100

results_df_2 = pd.DataFrame(data=[["Decision Tree Classifier", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df

#Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(x_train, y_train)
print_score(rf_clf, x_train, y_train, x_test, y_test, train=True)
print_score(rf_clf, x_train, y_train, x_test, y_test, train=False)

test_score = accuracy_score(y_test, rf_clf.predict(x_test)) * 100
train_score = accuracy_score(y_train, rf_clf.predict(x_train)) * 100

results_df_2 = pd.DataFrame(data=[["Random Forest Classifier", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


#Prediction Using Random Forest Classifier Model
pred = rf_clf.predict(x_test)
accuracy = accuracy_score(y_test, pred)*100
print(accuracy)



y_pred = rf_clf.predict(x_test)
y_pred

#Confusion Matrix between predicted values and given output values
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pred , y_test)
print(cm)
plt.subplot(3,3,7)
plt.title('Heart Disease --- Model: Random Forest Classifier')
sns.heatmap(cm, annot=True, cmap="Blues")
prediction = rf_clf.predict([[65,0,2,135,240,0,1,180,1,3,2,1,2]])

#Future Predicition
n = int(input("Enter no. of predictions to be made"))
for z in range(0,n):
   a = int(input("Enter Age"))
   b = int(input("Enter Sex male(1)or femlae(0)"))
   c = int(input("Enter Chest Pain type 0 or 1 or 2 or 3"))
   d = int(input("Enter resting bps"))
   e = int(input("Enter cholestral"))
   f = int(input("Enter fasting bps"))
   g = int(input("Enter resting electrocardiographic results 0 or 1"))
   h = int(input("Enter maximum heartrate"))
   i = int(input("Enter exercise induced angina 0 or 1"))
   j = float(input("Enter oldpeak"))
   k = int(input("Enter slope 0 or 1 or 2"))
   l = int(input("Enter ca 0 or 1 or 2"))
   m = int(input("Enter thal 0 or 1 or 2"))
   prediction = rf_clf.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m]])
   if prediction==1:
      print("Patient is Not Safe earlier precaution should be taken for Heart Disease")
   else:
      print("Patient is Safe from Heart Disease")