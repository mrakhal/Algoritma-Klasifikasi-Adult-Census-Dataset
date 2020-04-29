import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

adult_df = pd.read_csv(r'Z:\Pemrograman\Tubes Damin\adult.csv')
print("Number of Observations in adult dataset:", adult_df.shape)

adult_df.head()
adult_df.info()
adult_df.describe()

# Eksplorasi Data dan Visualisasi Melihat persebaran data berdasarkan presentase
cat_col = adult_df.dtypes[adult_df.dtypes == 'object']
num_col = adult_df.dtypes[adult_df.dtypes != 'object']

edit_cols = ['workclass','occupation','native.country']

##########MISSING VALUE##########################
#melakukan replace ? Private karena data paling banyak sebesar 19982
edit_cols1 = ['workclass']
#melakukan replace ? dengan unknown
for col in edit_cols1:
    adult_df.loc[adult_df[col] == '?', col] = 'Private'

edit_cols2 = ['occupation']
#melakukan replace ? Craft-repair karena data paling banyak sebesar 3593
for col in edit_cols2:
    adult_df.loc[adult_df[col] == '?', col] = 'Prof-specialty'
    
edit_cols3 = ['native.country']
#melakukan replace ? dengan United-States karena data paling banyak sebesar 25320
for col in edit_cols3:
    adult_df.loc[adult_df[col] == '?', col] = 'United-States'

# Periksa ? jumlahkan
for col in edit_cols:
    print(f"? in {col}: {adult_df[(adult_df[col] == '?')].any().sum()}")



hs_grad = ['HS-grad','11th','10th','9th','12th']
elementary = ['1st-4th','5th-6th','7th-8th']

# replace elemen dalam list.
adult_df['education'].replace(to_replace = hs_grad,value = 'HS-grad',inplace = True)
adult_df['education'].replace(to_replace = elementary,value = 'elementary_school',inplace = True)

adult_df['education'].value_counts()
# In[10]:

married= ['Married-spouse-absent','Married-civ-spouse','Married-AF-spouse']
separated = ['Separated','Divorced']

#replace elemen dalam list.
adult_df['marital.status'].replace(to_replace = married ,value = 'Married',inplace = True)
adult_df['marital.status'].replace(to_replace = separated,value = 'Separated',inplace = True)
adult_df['marital.status'].value_counts()

self_employed = ['Self-emp-not-inc','Self-emp-inc']
govt_employees = ['Local-gov','State-gov','Federal-gov']

#replace elemen dalam list.
adult_df['workclass'].replace(to_replace = self_employed ,value = 'Self_employed',inplace = True)
adult_df['workclass'].replace(to_replace = govt_employees,value = 'Govt_employees',inplace = True)
adult_df['workclass'].value_counts()

#Grafik Batang Pendapatan
plt.figure(figsize =(12,6));
sns.countplot(x = 'income', data = adult_df);
plt.xlabel("Income",fontsize = 12);
plt.ylabel("Frequency",fontsize = 12);
plt.show()


adult_df[list(num_col.index)].hist(figsize = (12,12));
plt.show()

capital_loss_df = adult_df[adult_df['capital.loss']>0]
capital_gain_df = adult_df[adult_df['capital.gain']>0]

print(f"Jumlah pengamatan yang memiliki kerugian modal di atas nilai median: {capital_loss_df.shape}\nJumlah pengamatan dalam setoran capital gain di atas nilai median: {capital_gain_df.shape}")
print(f"Persentase orang yang memiliki capital gain lebih besar dari nilai median: {(adult_df.loc[adult_df['capital.gain'] > 0,:].shape[0] / adult_df.shape[0])*100:.4f}%")
print(f"Persentase orang yang mengalami kerugian modal lebih besar dari nilai median: {(adult_df.loc[adult_df['capital.loss'] > 0,:].shape[0] / adult_df.shape[0])*100:.4f}%")
print(f"Jumlah pengamatan yang memiliki capital gain dan capital loss nol: {adult_df[(adult_df['capital.loss'] == 0) & (adult_df['capital.gain'] == 0)].shape}")

for col in cat_col.index:
    print(f"================================{col}=================================")
    print(adult_df[(adult_df['fnlwgt'] > 0) & (adult_df['fnlwgt'] > 0)][col].value_counts())
    

adult_df.loc[adult_df['capital.gain'] > 0,:].describe()
adult_df.loc[adult_df['capital.loss'] > 0,:].describe()

table_occu = pd.crosstab(adult_df['occupation'], adult_df['income'])
table_workclass = pd.crosstab(adult_df['workclass'], adult_df['income'])
table_edu = pd.crosstab(adult_df['education'], adult_df['income'])
table_marital = pd.crosstab(adult_df['marital.status'], adult_df['income'])
table_race = pd.crosstab(adult_df['race'], adult_df['income'])
table_sex = pd.crosstab(adult_df['sex'], adult_df['income'])
table_country = pd.crosstab(adult_df['native.country'], adult_df['income'])

fig = plt.figure(figsize = (17,6))

ax = fig.add_subplot(1,2,1)

(table_occu.div(table_occu.sum(axis= 1),axis = 0)*100).sort_values(by= '<=50K').plot(kind = 'bar',ax=ax);
plt.xlabel("Occupation",fontsize = 14);
plt.ylabel('Proportion of People',fontsize = 14);


ax = fig.add_subplot(1,2,2)
(table_workclass.div(table_workclass.sum(axis = 1),axis = 0)*100).sort_values(by = '<=50K').plot(kind = 'bar',ax=ax);
plt.xlabel("Workclass",fontsize = 14);

fig = plt.figure(figsize = (17,6))
ax = fig.add_subplot(1,2,1)
(table_edu.div(table_edu.sum(axis = 1),axis = 0)*100).sort_values(by = '<=50K').plot(kind = 'bar',ax =ax);
plt.xlabel('Education',fontsize = 14);
plt.ylabel('Proportion of People',fontsize = 14);

ax = fig.add_subplot(1,2,2)
(table_marital.div(table_marital.sum(axis = 1),axis = 0)*100).sort_values(by = '<=50K').plot(kind = 'bar',ax = ax);
plt.xlabel('Marital Status',fontsize = 14);
plt.ylabel('Proportion of People',fontsize = 14);

fig = plt.figure(figsize = (17,6))
ax = fig.add_subplot(1,2,1)
(table_race.div(table_race.sum(axis = 1),axis = 0)*100).sort_values(by = '<=50K').plot(kind = 'bar',ax =ax);
plt.xlabel('Race',fontsize = 14);
plt.ylabel('Proportion of People',fontsize = 14);

ax = fig.add_subplot(1,2,2)
(table_sex.div(table_sex.sum(axis = 1),axis = 0)*100).sort_values(by = '<=50K').plot(kind = 'bar',ax =ax);
plt.xlabel('Sex',fontsize = 14);
plt.ylabel('Proportion of People',fontsize = 14);

table_country = pd.crosstab(adult_df['native.country'], adult_df['income'])
(table_country.div(table_country.sum(axis = 1),axis = 0)*100).sort_values(by = '<=50K').plot(kind = 'bar',stacked = True,figsize = (17,6));
plt.xlabel('Native Country',fontsize = 14);
plt.ylabel('Proportion of People',fontsize = 14);

fig = plt.figure(figsize = (12,6))

sns.heatmap(adult_df[list(num_col.index)].corr(),annot = True,square = True);

fig = plt.figure(figsize = (17,10))
ax = fig.add_subplot(2,1,1)
sns.stripplot('age', 'capital.gain', data = adult_df,jitter = 0.2,ax = ax);
plt.xlabel('Age',fontsize = 12);
plt.ylabel('Capital Gain',fontsize = 12);

ax = fig.add_subplot(2,1,2)
sns.stripplot('age', 'capital.gain', data = adult_df,jitter = 0.2);
plt.xlabel('Age',fontsize = 12);
plt.ylabel('Capital Gain',fontsize = 12);
plt.ylim(0,40000);

adult_df[adult_df['age'] == 90].hist(figsize = (17,8));

fig = plt.figure(figsize = (17,10))
ax = fig.add_subplot(2,1,1)
sns.stripplot('hours.per.week', 'capital.gain', data = adult_df,jitter = 0.2,ax = ax);
plt.xlabel('Hours per week',fontsize = 12);
plt.ylabel('Capital Gain',fontsize = 12);

ax = fig.add_subplot(2,1,2)
sns.stripplot('hours.per.week', 'capital.gain', data = adult_df,jitter = 0.2,ax = ax);
plt.xlabel('Hours per week',fontsize = 12);
plt.ylabel('Capital Gain',fontsize = 12);
plt.ylim(0,40000);

fig = plt.figure(figsize = (17,6))

sns.stripplot('age','hours.per.week', data = adult_df,jitter = 0.2);
plt.xlabel('Age',fontsize = 12);
plt.ylabel('Hours per week',fontsize = 12);

print(f"Jumlah kolom sebelum didelete: {adult_df.shape[1]}")
del_cols = ['education.num']
adult_df.drop(labels = del_cols,axis = 1,inplace = True)
print(f"Jumlah kolom setelah didelete: {adult_df.shape[1]}")

num_col_new = ['age','capital.gain', 'capital.loss',
       'hours.per.week','fnlwgt']
cat_col_new = ['workclass', 'education', 'marital.status', 'occupation',
               'race', 'sex', 'native.country', 'income','relationship']

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
pd.DataFrame(scaler.fit_transform(adult_df[num_col_new]),columns = num_col_new).head(3)

class DataFrameSelector(TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names

    def fit(self,X,y = None):
        return self

    def transform(self,X):
        return X[self.attribute_names]


class num_trans(TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        df = pd.DataFrame(X)
        df.columns = num_col_new
        return df

pipeline = Pipeline([('selector',DataFrameSelector(num_col_new)), ('scaler',MinMaxScaler()),('transform',num_trans())])

num_df = pipeline.fit_transform(adult_df)
num_df.shape


cols = ['workclass_Govt_employess','education_Some-college','marital.status_Never-married','occupation_Other-service','race_Black','sex_Male','income_>50K']

class dummies(TransformerMixin):
    def __init__(self,cols):
        self.cols = cols

    def fit(self,X,y = None):
        return self

    def transform(self,X):
        df = pd.get_dummies(X)
        df_new = df[df.columns.difference(cols)]
        return df_new

pipeline_cat=Pipeline([('selector',DataFrameSelector(cat_col_new)),('dummies',dummies(cols))])
cat_df = pipeline_cat.fit_transform(adult_df)
cat_df.shape

cat_df['id'] = pd.Series(range(cat_df.shape[0]))
num_df['id'] = pd.Series(range(num_df.shape[0]))
final_df = pd.merge(cat_df,num_df,how = 'inner', on = 'id')
print(f"Jumlah observations final dataset: {final_df.shape}")

final_df.to_csv(r'Z:\Pemrograman\Tubes Damin\final_df.csv', index = False)

y = final_df['income_<=50K']
final_df.drop(labels = ['id','income_<=50K'],axis = 1,inplace = True)
X = final_df

#############TRAINING###########
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.25,random_state = 42)

clf_logreg = LogisticRegression(C=0.5,solver='liblinear')
clf_svc = SVC(kernel = 'rbf', probability = True)
clf_forest = RandomForestClassifier(n_estimators = 200)

classifiers = ['LogisticRegression', 'SVC', 'RandomForest']

models = {clf_logreg:'LogisticRegression',
          clf_svc: 'SVC',
          clf_forest: 'RandomForest'}

# train function fits the model and returns accuracy score

def train(algo, name, X_train, y_train, X_test, y_test):
    algo.fit(X_train, y_train)
    y_pred = algo.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"--------------------------------------------{name}---------------------------------------------------")
    return y_test, y_pred, score

def acc_res(y_test, y_pred):
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    TN = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    TP = matrix[1, 1]
    accuracy_score = (TN + TP) / float(TP + TN + FP + FN)
    recall_score = (TP) / float(TP + FN)
    precision_score = TP / float(TP + FP)
    f_measure = 2 * (float(precision_score*recall_score)/float(precision_score+recall_score))
    print(f"Accuracy Score: {accuracy_score * 100:.4f}%")
    print(f"Recall Score: {recall_score * 100:.4f}%")
    print(f"Precision Score: {precision_score * 100:.4f}%")
    print(f"F-measure: {f_measure:.4f}")


def main(models):
    accuracy_scores = []
    for algo, name in models.items():
        y_test_train, y_pred, acc_score = train(algo, name, X_train, y_train, X_test, y_test)
        acc_res(y_test_train, y_pred)
        accuracy_scores.append(acc_score)
    return accuracy_scores

accuracy_scores = main(models)

pd.DataFrame(accuracy_scores,columns = ['Accuracy Scores'],index = classifiers).sort_values(by = 'Accuracy Scores', ascending = False)
plt.show()


