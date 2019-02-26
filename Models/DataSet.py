import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer as dictV
import seaborn as sns
import matplotlib.pyplot as plt


class DataSet:

    def getParticionedDataSet():

        # Load the csv which contains the client information
        clientsDF = pd.read_csv('../Data/clientesLimpios.csv', skipinitialspace=True)
        # print(clientsDF.head())

        # Load the csv which contains the client's movements information
        movsDF = pd.read_csv('../Data/movsLimpios.csv', skipinitialspace=True)
        # print(movsDF.head())

        # Unify the two data frames before created in one data frame joined by the client identification
        crossedDF = pd.merge(clientsDF, movsDF, left_on='CLIENTE_CC', right_on='CLIENTE_CC', how='inner')

        # Verify if which values are present in the fuga field
        # print(type(crossedDF['fuga'].unique()[1]))

        # Check if are some rows with null value in the data frame
        # print("SOME NULLS: \n", crossedDF.isnull())
        # sns.heatmap(crossedDF.isnull())
        # plt.savefig('../Cant_Nulos')
        # plt.show()

        # Replace some values in null in our data set
        crossedDF['fuga'] = np.where(crossedDF['fuga'].isnull(), 0, crossedDF['fuga'])

        # sns.countplot(x='fuga', data=crossedDF)
        # plt.savefig('../Cant_Nulos_Fuga')
        # plt.show()

        # sns.countplot(x='fuga', hue='SEXO', data=crossedDF)
        # plt.savefig('../Cant_Nulos_Fuga_Sexo')
        # plt.show()

        # sns.countplot(x='fuga', hue='ESTADO_CIVIL', data=crossedDF)
        # plt.savefig('../Cant_Nulos_EstadoCivil')
        # plt.show()

        # Casting de data information
        crossedDF['fuga'] = crossedDF['fuga'].astype(int)
        crossedDF['MES_DE_FUGA'] = crossedDF['MES_DE_FUGA'].replace(np.nan, 0).astype(int)
        crossedDF['ESTADO_CIVIL'] = crossedDF['ESTADO_CIVIL'].replace(np.nan, 'N/A')

        crossedDF.columns = crossedDF.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', ''). \
            str.replace(')', '')

        categorical_data = crossedDF[['sexo', 'situacion_laboral', 'estado_civil']]
        categorical_data = pd.DataFrame(categorical_data)

        # print(crossedDF.info())
        crossedDF = crossedDF.drop(
            ['fecha_alta', 'fecha_nacimiento', 'fecha_informacion', 'cliente_cc', 'situacion_laboral',
             'estado_civil', 'sexo'], axis=1)
        # print(type(categorical_data))

        # print("CLASEEEEE:", crossedDF.head())

        x_categorical_data = categorical_data.to_dict(orient='records')
        vectorizer = dictV(sparse=False)
        vec_x_categorical_data = vectorizer.fit_transform(x_categorical_data)
        X_train = np.hstack((crossedDF, vec_x_categorical_data))

        X_train, X_test, y_train, y_test = train_test_split(crossedDF.drop('fuga', axis=1), crossedDF['fuga'],
                                                            test_size=0.30, random_state=101)

        # Check the length of the data sets train and test
        # print("X_TRAIN: \n", len(X_train))
        # print("X_test: \n", len(X_test))
        # print("y_train: \n", len(y_train))
        # print("y_test: \n", len(y_test))

        return X_train, X_test, y_train, y_test

