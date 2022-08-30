# %% [markdown]
# # Building a classification model for Cardiac Arrest Prediction 

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:47.938878Z","iopub.execute_input":"2022-06-07T14:45:47.940232Z","iopub.status.idle":"2022-06-07T14:45:49.150496Z","shell.execute_reply.started":"2022-06-07T14:45:47.940072Z","shell.execute_reply":"2022-06-07T14:45:49.149486Z"}}
import pandas as  pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2022-06-07T14:45:49.152351Z","iopub.execute_input":"2022-06-07T14:45:49.152819Z","iopub.status.idle":"2022-06-07T14:45:49.303042Z","shell.execute_reply.started":"2022-06-07T14:45:49.152771Z","shell.execute_reply":"2022-06-07T14:45:49.301737Z"}}
data=pd.read_csv('../input/cardio-train/cardio_train.csv')
data

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:49.305492Z","iopub.execute_input":"2022-06-07T14:45:49.305956Z","iopub.status.idle":"2022-06-07T14:45:49.355376Z","shell.execute_reply.started":"2022-06-07T14:45:49.305912Z","shell.execute_reply":"2022-06-07T14:45:49.354227Z"}}
data=data.drop_duplicates()
data

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:49.357061Z","iopub.execute_input":"2022-06-07T14:45:49.357411Z","iopub.status.idle":"2022-06-07T14:45:49.366176Z","shell.execute_reply.started":"2022-06-07T14:45:49.357379Z","shell.execute_reply":"2022-06-07T14:45:49.365411Z"}}
data=data.drop(['id','active'],axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:49.367338Z","iopub.execute_input":"2022-06-07T14:45:49.367681Z","iopub.status.idle":"2022-06-07T14:45:49.381188Z","shell.execute_reply.started":"2022-06-07T14:45:49.367651Z","shell.execute_reply":"2022-06-07T14:45:49.380405Z"}}
data.age=round(data.age/365)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:49.382393Z","iopub.execute_input":"2022-06-07T14:45:49.382883Z","iopub.status.idle":"2022-06-07T14:45:49.406132Z","shell.execute_reply.started":"2022-06-07T14:45:49.382851Z","shell.execute_reply":"2022-06-07T14:45:49.404905Z"}}
data

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:49.407768Z","iopub.execute_input":"2022-06-07T14:45:49.408266Z","iopub.status.idle":"2022-06-07T14:45:49.413084Z","shell.execute_reply.started":"2022-06-07T14:45:49.408201Z","shell.execute_reply":"2022-06-07T14:45:49.411815Z"}}
# here we can get one more feature i.e BMI
# to calculate BMI we need height in meter squared but given is cm.

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:49.414837Z","iopub.execute_input":"2022-06-07T14:45:49.415354Z","iopub.status.idle":"2022-06-07T14:45:49.443365Z","shell.execute_reply.started":"2022-06-07T14:45:49.415306Z","shell.execute_reply":"2022-06-07T14:45:49.442058Z"}}
data['BMI']=data['weight']/((data.height/100)**2)
data

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:49.445102Z","iopub.execute_input":"2022-06-07T14:45:49.44586Z","iopub.status.idle":"2022-06-07T14:45:50.380493Z","shell.execute_reply.started":"2022-06-07T14:45:49.445809Z","shell.execute_reply":"2022-06-07T14:45:50.379279Z"}}
cm=data.corr()
plt.figure(figsize=(12,9))
sns.heatmap(cm,annot=True,linewidths=2)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:50.384587Z","iopub.execute_input":"2022-06-07T14:45:50.385611Z","iopub.status.idle":"2022-06-07T14:45:50.471291Z","shell.execute_reply.started":"2022-06-07T14:45:50.385556Z","shell.execute_reply":"2022-06-07T14:45:50.470091Z"}}
data.describe()

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:50.472927Z","iopub.execute_input":"2022-06-07T14:45:50.473579Z","iopub.status.idle":"2022-06-07T14:45:50.478623Z","shell.execute_reply.started":"2022-06-07T14:45:50.473533Z","shell.execute_reply":"2022-06-07T14:45:50.477528Z"}}
# there are some unrealastic values from ap_lo and ap_hi
# neglecting those values

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:50.480095Z","iopub.execute_input":"2022-06-07T14:45:50.480686Z","iopub.status.idle":"2022-06-07T14:45:50.518117Z","shell.execute_reply.started":"2022-06-07T14:45:50.480643Z","shell.execute_reply":"2022-06-07T14:45:50.517192Z"}}
data=data[data['ap_lo']>0]
data=data[data['ap_hi']<250]
data=data[data['ap_hi']>60]
data=data[data['ap_lo']<150]
data=data[data['ap_lo']>50]

# %% [markdown]
# ## Univariate analyasis

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:50.519196Z","iopub.execute_input":"2022-06-07T14:45:50.519658Z","iopub.status.idle":"2022-06-07T14:45:50.661331Z","shell.execute_reply.started":"2022-06-07T14:45:50.519628Z","shell.execute_reply":"2022-06-07T14:45:50.660291Z"}}
sns.countplot(data.cardio)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:50.663005Z","iopub.execute_input":"2022-06-07T14:45:50.66366Z","iopub.status.idle":"2022-06-07T14:45:50.865168Z","shell.execute_reply.started":"2022-06-07T14:45:50.663612Z","shell.execute_reply":"2022-06-07T14:45:50.864466Z"}}
sns.countplot(data.cholesterol,hue=data.cardio)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:50.866194Z","iopub.execute_input":"2022-06-07T14:45:50.866597Z","iopub.status.idle":"2022-06-07T14:45:51.047066Z","shell.execute_reply.started":"2022-06-07T14:45:50.866569Z","shell.execute_reply":"2022-06-07T14:45:51.046243Z"}}
sns.countplot(data.smoke,hue=data.cardio)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:51.047993Z","iopub.execute_input":"2022-06-07T14:45:51.048899Z","iopub.status.idle":"2022-06-07T14:45:51.233259Z","shell.execute_reply.started":"2022-06-07T14:45:51.048859Z","shell.execute_reply":"2022-06-07T14:45:51.23254Z"}}
sns.countplot(data.alco,hue=data.cardio)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:51.234202Z","iopub.execute_input":"2022-06-07T14:45:51.235032Z","iopub.status.idle":"2022-06-07T14:45:51.427764Z","shell.execute_reply.started":"2022-06-07T14:45:51.234996Z","shell.execute_reply":"2022-06-07T14:45:51.426644Z"}}
sns.countplot(data.cholesterol,hue=data.gender)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:51.429035Z","iopub.execute_input":"2022-06-07T14:45:51.429376Z","iopub.status.idle":"2022-06-07T14:45:52.550275Z","shell.execute_reply.started":"2022-06-07T14:45:51.429347Z","shell.execute_reply":"2022-06-07T14:45:52.549126Z"}}
sns.barplot(data.cholesterol,data.cardio,hue=data.gender)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:52.551917Z","iopub.execute_input":"2022-06-07T14:45:52.552298Z","iopub.status.idle":"2022-06-07T14:45:52.71431Z","shell.execute_reply.started":"2022-06-07T14:45:52.552266Z","shell.execute_reply":"2022-06-07T14:45:52.712407Z"}}
sns.countplot(data.BMI<25,hue=data.cardio)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:45:52.715609Z","iopub.execute_input":"2022-06-07T14:45:52.716133Z","iopub.status.idle":"2022-06-07T14:46:04.332621Z","shell.execute_reply.started":"2022-06-07T14:45:52.7161Z","shell.execute_reply":"2022-06-07T14:46:04.331297Z"}}
sns.regplot(data.ap_hi,data.ap_lo)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:04.334Z","iopub.execute_input":"2022-06-07T14:46:04.334397Z","iopub.status.idle":"2022-06-07T14:46:04.672562Z","shell.execute_reply.started":"2022-06-07T14:46:04.334362Z","shell.execute_reply":"2022-06-07T14:46:04.671231Z"}}
sns.scatterplot(data.ap_hi,data.age)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:04.674367Z","iopub.execute_input":"2022-06-07T14:46:04.675327Z","iopub.status.idle":"2022-06-07T14:46:04.970009Z","shell.execute_reply.started":"2022-06-07T14:46:04.675275Z","shell.execute_reply":"2022-06-07T14:46:04.968878Z"}}
plt.scatter(data.ap_hi,data.ap_lo)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:04.97136Z","iopub.execute_input":"2022-06-07T14:46:04.971987Z","iopub.status.idle":"2022-06-07T14:46:05.925136Z","shell.execute_reply.started":"2022-06-07T14:46:04.971954Z","shell.execute_reply":"2022-06-07T14:46:05.92417Z"}}
col=data.columns
plt.figure(figsize=(15,10))
for i in range(len(col)-6):
    plt.subplot(2,3,i+1)
    plt.hist(data[col[i]])
    plt.xlabel(col[i])
    

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2022-06-07T14:46:05.926556Z","iopub.execute_input":"2022-06-07T14:46:05.927711Z","iopub.status.idle":"2022-06-07T14:46:06.489327Z","shell.execute_reply.started":"2022-06-07T14:46:05.927662Z","shell.execute_reply":"2022-06-07T14:46:06.488587Z"}}
col=data.columns
plt.figure(figsize=(15,10))
for i in range(len(col)-6):
    plt.subplot(2,3,i+1)
    plt.boxplot(data[col[i]])
    plt.xlabel(col[i])

# %% [markdown]
# ## Result of Univariate Analysis
# * By observing the histogram and box plot there are more no of outliers are present in Height, Weight, ap_hi, ap_lo.
# * To deal with this here i am using 2nd and 3rd standard deviation

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:06.490242Z","iopub.execute_input":"2022-06-07T14:46:06.491116Z","iopub.status.idle":"2022-06-07T14:46:06.501072Z","shell.execute_reply.started":"2022-06-07T14:46:06.491082Z","shell.execute_reply":"2022-06-07T14:46:06.500129Z"}}
upper_limit= data.weight.mean() + 2*data.weight.std()
print(upper_limit)
lower_limit= data.weight.mean() - 2*data.weight.std()
lower_limit

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:06.502083Z","iopub.execute_input":"2022-06-07T14:46:06.503165Z","iopub.status.idle":"2022-06-07T14:46:06.516916Z","shell.execute_reply.started":"2022-06-07T14:46:06.503008Z","shell.execute_reply":"2022-06-07T14:46:06.515915Z"}}
data=data[data['weight']<upper_limit] 

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:06.520756Z","iopub.execute_input":"2022-06-07T14:46:06.52116Z","iopub.status.idle":"2022-06-07T14:46:06.529777Z","shell.execute_reply.started":"2022-06-07T14:46:06.521129Z","shell.execute_reply":"2022-06-07T14:46:06.528906Z"}}
data=data[data['weight']>lower_limit]

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:06.535028Z","iopub.execute_input":"2022-06-07T14:46:06.536177Z","iopub.status.idle":"2022-06-07T14:46:06.61181Z","shell.execute_reply.started":"2022-06-07T14:46:06.536136Z","shell.execute_reply":"2022-06-07T14:46:06.610968Z"}}
data.describe()

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:06.613295Z","iopub.execute_input":"2022-06-07T14:46:06.614085Z","iopub.status.idle":"2022-06-07T14:46:06.630768Z","shell.execute_reply.started":"2022-06-07T14:46:06.614008Z","shell.execute_reply":"2022-06-07T14:46:06.629908Z"}}
upper_limit= data.height.mean() + 2*data.height.std()
print('upper limit: ',upper_limit)
lower_limit= data.height.mean() - 2*data.height.std()
print('Lower limit: ',lower_limit)

data=data[data['height']<upper_limit]
data=data[data['height']>lower_limit]


# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:06.632154Z","iopub.execute_input":"2022-06-07T14:46:06.633192Z","iopub.status.idle":"2022-06-07T14:46:06.650384Z","shell.execute_reply.started":"2022-06-07T14:46:06.633153Z","shell.execute_reply":"2022-06-07T14:46:06.649282Z"}}
upper_limit= data.ap_hi.mean() + 3*data.ap_hi.std()
print('upper limit: ',upper_limit)
lower_limit= data.ap_hi.mean() - 3*data.ap_hi.std()
print('Lower limit: ',lower_limit)

data=data[data['ap_hi']<upper_limit]
data=data[data['ap_hi']>lower_limit]


# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:06.651857Z","iopub.execute_input":"2022-06-07T14:46:06.652644Z","iopub.status.idle":"2022-06-07T14:46:06.67108Z","shell.execute_reply.started":"2022-06-07T14:46:06.652601Z","shell.execute_reply":"2022-06-07T14:46:06.669867Z"}}
upper_limit= data.ap_lo.mean() + 3*data.ap_lo.std()
print('upper limit: ',upper_limit)
lower_limit= data.ap_lo.mean() - 3*data.ap_lo.std()
print('Lower limit: ',lower_limit)

data=data[data['ap_lo']<150]
data=data[data['ap_lo']>50]

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:06.673342Z","iopub.execute_input":"2022-06-07T14:46:06.674636Z","iopub.status.idle":"2022-06-07T14:46:07.24168Z","shell.execute_reply.started":"2022-06-07T14:46:06.674589Z","shell.execute_reply":"2022-06-07T14:46:07.240469Z"}}
col=data.columns
plt.figure(figsize=(15,10))
for i in range(len(col)-6):
    plt.subplot(2,3,i+1)
    plt.boxplot(data[col[i]])
    plt.xlabel(col[i])

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:07.243047Z","iopub.execute_input":"2022-06-07T14:46:07.243435Z","iopub.status.idle":"2022-06-07T14:46:08.004032Z","shell.execute_reply.started":"2022-06-07T14:46:07.243402Z","shell.execute_reply":"2022-06-07T14:46:08.002842Z"}}
col=data.columns
plt.figure(figsize=(15,10))
for i in range(len(col)-6):
    plt.subplot(2,3,i+1)
    plt.hist(data[col[i]])
    plt.xlabel(col[i])

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:08.005389Z","iopub.execute_input":"2022-06-07T14:46:08.005732Z","iopub.status.idle":"2022-06-07T14:46:08.083056Z","shell.execute_reply.started":"2022-06-07T14:46:08.005701Z","shell.execute_reply":"2022-06-07T14:46:08.081898Z"}}
data.describe()

# %% [markdown]
# # Building the model

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:08.084568Z","iopub.execute_input":"2022-06-07T14:46:08.08496Z","iopub.status.idle":"2022-06-07T14:46:08.090696Z","shell.execute_reply.started":"2022-06-07T14:46:08.08493Z","shell.execute_reply":"2022-06-07T14:46:08.089379Z"}}
y=data.cardio

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:08.092186Z","iopub.execute_input":"2022-06-07T14:46:08.092639Z","iopub.status.idle":"2022-06-07T14:46:08.107Z","shell.execute_reply.started":"2022-06-07T14:46:08.092604Z","shell.execute_reply":"2022-06-07T14:46:08.106028Z"}}
x=data[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco']]

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:08.108161Z","iopub.execute_input":"2022-06-07T14:46:08.108598Z","iopub.status.idle":"2022-06-07T14:46:08.131949Z","shell.execute_reply.started":"2022-06-07T14:46:08.108562Z","shell.execute_reply":"2022-06-07T14:46:08.131032Z"}}
x

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:08.132839Z","iopub.execute_input":"2022-06-07T14:46:08.133146Z","iopub.status.idle":"2022-06-07T14:46:08.511051Z","shell.execute_reply.started":"2022-06-07T14:46:08.133119Z","shell.execute_reply":"2022-06-07T14:46:08.509937Z"}}
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:08.514159Z","iopub.execute_input":"2022-06-07T14:46:08.514934Z","iopub.status.idle":"2022-06-07T14:46:08.543302Z","shell.execute_reply.started":"2022-06-07T14:46:08.514885Z","shell.execute_reply":"2022-06-07T14:46:08.542328Z"}}
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=True)

# %% [markdown]
# ### Logistic Regression

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:08.544573Z","iopub.execute_input":"2022-06-07T14:46:08.545888Z","iopub.status.idle":"2022-06-07T14:46:09.457551Z","shell.execute_reply.started":"2022-06-07T14:46:08.545837Z","shell.execute_reply":"2022-06-07T14:46:09.456396Z"}}
lgr=LogisticRegression(solver='newton-cg').fit(x_train,y_train)
print('Accuracy of Logistic Regression:',accuracy_score(y_test,lgr.predict(x_test)))

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:09.459072Z","iopub.execute_input":"2022-06-07T14:46:09.466127Z","iopub.status.idle":"2022-06-07T14:46:09.561637Z","shell.execute_reply.started":"2022-06-07T14:46:09.466035Z","shell.execute_reply":"2022-06-07T14:46:09.560398Z"}}
print('Classification Report:\n',classification_report(y_test,lgr.predict(x_test)))

# %% [markdown]
# ### Gradient Boosting Classifier

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:09.568547Z","iopub.execute_input":"2022-06-07T14:46:09.572386Z","iopub.status.idle":"2022-06-07T14:46:12.652686Z","shell.execute_reply.started":"2022-06-07T14:46:09.5723Z","shell.execute_reply":"2022-06-07T14:46:12.651658Z"}}
model=GradientBoostingClassifier().fit(x_train,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:12.654158Z","iopub.execute_input":"2022-06-07T14:46:12.654647Z","iopub.status.idle":"2022-06-07T14:46:12.697372Z","shell.execute_reply.started":"2022-06-07T14:46:12.654605Z","shell.execute_reply":"2022-06-07T14:46:12.696186Z"}}
pred=model.predict(x_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:12.698546Z","iopub.execute_input":"2022-06-07T14:46:12.70018Z","iopub.status.idle":"2022-06-07T14:46:12.708989Z","shell.execute_reply.started":"2022-06-07T14:46:12.700138Z","shell.execute_reply":"2022-06-07T14:46:12.70791Z"}}
print('Accuracy of GradientBoostingClassifier:',accuracy_score(y_test,pred))

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:12.710668Z","iopub.execute_input":"2022-06-07T14:46:12.711157Z","iopub.status.idle":"2022-06-07T14:46:12.766146Z","shell.execute_reply.started":"2022-06-07T14:46:12.711112Z","shell.execute_reply":"2022-06-07T14:46:12.764879Z"}}
print('Classification Report:\n',classification_report(y_test,pred))

# %% [markdown]
# ### Random Forest Classifier

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:12.767708Z","iopub.execute_input":"2022-06-07T14:46:12.768053Z","iopub.status.idle":"2022-06-07T14:46:17.605161Z","shell.execute_reply.started":"2022-06-07T14:46:12.768024Z","shell.execute_reply":"2022-06-07T14:46:17.604323Z"}}
rfc = RandomForestClassifier(random_state=True)
rfc.fit(x_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:17.606578Z","iopub.execute_input":"2022-06-07T14:46:17.607139Z"}}
print('Accuracy of random forrest classifier:',accuracy_score(y_test,rfc.predict(x_test)))

# %% [code] {"execution":{"iopub.status.idle":"2022-06-07T14:46:19.209086Z","shell.execute_reply.started":"2022-06-07T14:46:18.413769Z","shell.execute_reply":"2022-06-07T14:46:19.207827Z"}}
print('Classification Report:\n',classification_report(y_test,rfc.predict(x_test)))

# %% [markdown]
# ## Evalution of above models

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:19.210236Z","iopub.execute_input":"2022-06-07T14:46:19.210549Z","iopub.status.idle":"2022-06-07T14:46:19.215244Z","shell.execute_reply.started":"2022-06-07T14:46:19.210522Z","shell.execute_reply":"2022-06-07T14:46:19.214291Z"}}
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve

# %% [markdown]
# ### Based on confusion matrix

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:19.216572Z","iopub.execute_input":"2022-06-07T14:46:19.216909Z","iopub.status.idle":"2022-06-07T14:46:19.471362Z","shell.execute_reply.started":"2022-06-07T14:46:19.21688Z","shell.execute_reply":"2022-06-07T14:46:19.470473Z"}}
print('Confusion matrix of Logistic Regresssion model:')
lgr_cf=plot_confusion_matrix(lgr,x_test,y_test,cmap='Blues_r')

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:19.472602Z","iopub.execute_input":"2022-06-07T14:46:19.474616Z","iopub.status.idle":"2022-06-07T14:46:20.487926Z","shell.execute_reply.started":"2022-06-07T14:46:19.474568Z","shell.execute_reply":"2022-06-07T14:46:20.486752Z"}}
print('Confusion matrix of Random Forest Classifier model:')
lgr_cf=plot_confusion_matrix(rfc,x_test,y_test,cmap='Blues_r')

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:20.489179Z","iopub.execute_input":"2022-06-07T14:46:20.489512Z","iopub.status.idle":"2022-06-07T14:46:20.7224Z","shell.execute_reply.started":"2022-06-07T14:46:20.489484Z","shell.execute_reply":"2022-06-07T14:46:20.721705Z"}}
print('Confusion matrix of Gradient Boosting classifier model:')
lgr_cf=plot_confusion_matrix(model,x_test,y_test,cmap='Blues_r')

# %% [markdown]
# ### ROC and Precision curve

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:20.723291Z","iopub.execute_input":"2022-06-07T14:46:20.724071Z","iopub.status.idle":"2022-06-07T14:46:20.961993Z","shell.execute_reply.started":"2022-06-07T14:46:20.724036Z","shell.execute_reply":"2022-06-07T14:46:20.960804Z"}}
gbc_disp=plot_roc_curve(model,x_test,y_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:20.963638Z","iopub.execute_input":"2022-06-07T14:46:20.96407Z","iopub.status.idle":"2022-06-07T14:46:22.026274Z","shell.execute_reply.started":"2022-06-07T14:46:20.964037Z","shell.execute_reply":"2022-06-07T14:46:22.025047Z"}}
lgr_disp=plot_roc_curve(lgr,x_test,y_test)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, x_test, y_test, ax=ax)
gbc_disp.plot(ax=ax)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:22.027626Z","iopub.execute_input":"2022-06-07T14:46:22.028031Z","iopub.status.idle":"2022-06-07T14:46:22.306537Z","shell.execute_reply.started":"2022-06-07T14:46:22.027997Z","shell.execute_reply":"2022-06-07T14:46:22.305799Z"}}
gbc_prc=plot_precision_recall_curve(model,x_test,y_test)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:22.307572Z","iopub.execute_input":"2022-06-07T14:46:22.308482Z","iopub.status.idle":"2022-06-07T14:46:23.445335Z","shell.execute_reply.started":"2022-06-07T14:46:22.308435Z","shell.execute_reply":"2022-06-07T14:46:23.444552Z"}}
lgr_prc=plot_precision_recall_curve(lgr,x_test,y_test)
ax = plt.gca()
rfc_prc = plot_precision_recall_curve(rfc, x_test, y_test, ax=ax)
gbc_prc.plot(ax=ax)
plt.show()

# %% [markdown]
# # Result of evalution
# * Considering all the obervation and metrics result, logistic regression and Gradient Boosting are performing similar.
# * But based on ROC and Precision curve Gradient Boosting is at top of all the three models.
# * Hence here choosing Gradient boosting as final model. 

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:23.44637Z","iopub.execute_input":"2022-06-07T14:46:23.447312Z","iopub.status.idle":"2022-06-07T14:46:23.458591Z","shell.execute_reply.started":"2022-06-07T14:46:23.447242Z","shell.execute_reply":"2022-06-07T14:46:23.457523Z"}}
pickle.dump(model,open('GradBClass.pkl','wb'))

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:23.460109Z","iopub.execute_input":"2022-06-07T14:46:23.460515Z","iopub.status.idle":"2022-06-07T14:46:23.478355Z","shell.execute_reply.started":"2022-06-07T14:46:23.460483Z","shell.execute_reply":"2022-06-07T14:46:23.477312Z"}}
pic=pickle.load(open('GradBClass.pkl','rb'))

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:23.479798Z","iopub.execute_input":"2022-06-07T14:46:23.481159Z","iopub.status.idle":"2022-06-07T14:46:23.494918Z","shell.execute_reply.started":"2022-06-07T14:46:23.481108Z","shell.execute_reply":"2022-06-07T14:46:23.493603Z"}}
pic.predict([[52.0,1,165,64.0,130,70,3,1,0,0]])

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:23.497911Z","iopub.execute_input":"2022-06-07T14:46:23.499776Z","iopub.status.idle":"2022-06-07T14:46:23.508679Z","shell.execute_reply.started":"2022-06-07T14:46:23.499654Z","shell.execute_reply":"2022-06-07T14:46:23.507654Z"}}
import sklearn

# %% [code] {"execution":{"iopub.status.busy":"2022-06-07T14:46:23.510265Z","iopub.execute_input":"2022-06-07T14:46:23.510985Z","iopub.status.idle":"2022-06-07T14:46:23.522363Z","shell.execute_reply.started":"2022-06-07T14:46:23.510942Z","shell.execute_reply":"2022-06-07T14:46:23.52129Z"}}
print('sklearn: {}'.format(sklearn.__version__))