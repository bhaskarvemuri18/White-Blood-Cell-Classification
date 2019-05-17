import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def get_data(file_name):
    data = pd.read_csv(file_name)
    data = data.drop(0,axis=0)
    data = np.array(data)
    labels = data[:,-1]
    data = data[:,1:-1]
    #print (data.shape)
    #rint (labels.shape)
    return (data,labels)
X_train, Y_train = get_data('/media/shashank/Study1/6140/project/dataset2-master/training_data.csv')


def pca(X_train,components):
    pca = decomposition.PCA(n_components=components)
    pca.fit(X_train)
    return pca.transform(X_train)


def train_model(model, data, labels):
    data, labels = shuffle(data, labels)

    print ("Starting Training")
    model.fit(data, labels)
    result = model.score(data, labels)
    print (result)

def save_model(model,file_name)
    pickle.dump(model,open(file_name,'wb'))

decomposed_X_train = pca(X_train,4000)
decomposed_X_train, Y_train = shuffle(decomposed_X_train,Y_train)
logreg = LogisticRegression(C=1e5,solver='lbfgs',multi_class='multinomial',max_iter=20000)
tree = DecisionTreeClassifier(random_state=0)
nb = GaussianNB(priors=None, var_smoothing=1e-09)
model = logreg
train_model(model,decomposed_X_train,Y_train)
save_model(model,'/media/shashank/Study1/6140/project/code/models/naive_bayes_with_pca_4000.sav')

