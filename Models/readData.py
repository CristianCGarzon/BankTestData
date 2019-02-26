import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer as dictV
from imblearn.over_sampling import SMOTE


# Load the csv which contains the client information
clientsDF = pd.read_csv('Data/clientesLimpios.csv', skipinitialspace=True)
# print(clientsDF.head())

# Load the csv which contains the client's movements information
movsDF = pd.read_csv('Data/movsLimpios.csv', skipinitialspace=True)
# print(movsDF.head())

# Unify the two data frames before created in one data frame joined by the client identification
crossedDF = pd.merge(clientsDF, movsDF, left_on='CLIENTE_CC', right_on='CLIENTE_CC', how='inner')

# Verify if which values are present in the fuga field
# print(type(crossedDF['fuga'].unique()[1]))

# Replace some values in null in our data set
crossedDF['fuga'] = np.where(crossedDF['fuga'].isnull(), 0, crossedDF['fuga'])
print(crossedDF['fuga'].unique())
crossedDF['fuga'] = crossedDF['fuga'].astype(int)
crossedDF['MES_DE_FUGA'] = crossedDF['MES_DE_FUGA'].replace(np.nan, 0).astype(int)
crossedDF['ESTADO_CIVIL'] = crossedDF['ESTADO_CIVIL'].replace(np.nan, 'N/A')

crossedDF.columns = crossedDF.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').\
                    str.replace(')', '')

# Check if are some rows with null value in the data frame
# print("SOME NULLS: \n", crossedDF.isnull())
#sns.heatmap(crossedDF.isnull())
#plt.show()

#sns.countplot(x='fuga', data=crossedDF)
#plt.show()

#sns.countplot(x='fuga', hue='SEXO', data=crossedDF)
#plt.show()

#sns.countplot(x='fuga', hue='ESTADO_CIVIL', data=crossedDF)
#plt.show()

categorical_data = crossedDF[['sexo', 'situacion_laboral', 'estado_civil']]

categorical_data = pd.DataFrame(categorical_data)

#print(crossedDF.info())
crossedDF = crossedDF.drop(['fecha_alta', 'fecha_nacimiento', 'fecha_informacion', 'cliente_cc', 'situacion_laboral',
                            'estado_civil', 'sexo'], axis=1)
#print(type(categorical_data))

#print("CLASEEEEE:", crossedDF.head())

x_categorical_data = categorical_data.to_dict(orient='records')

vectorizer = dictV(sparse=False)
vec_x_categorical_data = vectorizer.fit_transform(x_categorical_data)

X_train = np.hstack((crossedDF, vec_x_categorical_data))

X_train, X_test, y_train, y_test = train_test_split(crossedDF.drop('fuga', axis=1), crossedDF['fuga'],
                                                    test_size=0.30, random_state=101)

print("X_TRAIN: \n", len(X_train))
print("X_test: \n", len(X_test))
print("y_train: \n", len(y_train))
print("y_test: \n", len(y_test))

logModel = LogisticRegression()
logModel.fit(X_train, y_train)
predictions = logModel.predict(X_test)
print(classification_report(y_test, predictions))

os = SMOTE(random_state=0)
columns = X_train.columns

os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ", len(os_data_X))
print("Number of no subscription in oversampled data", len(os_data_y[os_data_y['y'] == 0]))
print("Number of subscription", len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ", len(os_data_y[os_data_y['y'] == 0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ", len(os_data_y[os_data_y['y'] == 1])/len(os_data_X))

logModel = LogisticRegression()
logModel.fit(os_data_X, os_data_y.values.ravel())
predictions = logModel.predict(X_test)
print(classification_report(y_test, predictions))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logModel.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logModel.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

