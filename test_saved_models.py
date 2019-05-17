import pandas as pd
import numpy as np
#from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import decomposition

def get_data(file_name):
    data = pd.read_csv(file_name)
    data = data.drop(0,axis=0)
    data = np.array(data)
    labels = data[:,-1]
    data = data[:,1:-1]
    #print (data.shape)
    #rint (labels.shape)
    return (data,labels)



def pca(X_train,X_test,components)
    pca = decomposition.PCA(n_components=components)
    pca.fit(X_train)
    decomposed_X_train = pca.transform(X_train)
    decomposed_X_test = pca.transform(X_test)
    return (decomposed_X_train,decomposed_X_test)

X_test,Y_test = get_data('/media/shashank/Study1/6140/project/dataset2-master/testing_data.csv')
X_train, Y_train = get_data('/media/shashank/Study1/6140/project/dataset2-master/training_data.csv')

decomposed_X_train, decomposed_X_test = pca(X_train,X_test,200)
loaded_model = pickle.load('/media/shashank/Study1/6140/project/code/models/logistic_regression_with_pca_200.sav')
result = loaded_model.score(decomposed_X_test, Y_test)
pred = loaded_model.predict(decomposed_X_test)
print (result)
print (pred)
print (Y_test)
