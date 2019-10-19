
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy 
print('Scipy: {}'.format(scipy.__version__))
import numpy 
import matplotlib
import sklearn 
import pandas 
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('pandas: {}'.format(pandas.__version__))

# let’s import all of the modules, functions and objects we are going to use in this tutorial.
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt 
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Dataset 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv('./iris.csv', names=names)
print(dataset.shape)
# head 
print(dataset.head(20))

# print statistical summary 
print(dataset.describe())

# visualize data
dataset.hist()
plt.show()

# Evaluate some algorithms on data
# 1. Separate out validation dataset 
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Build Models 
models = []
models.append(('LR', LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsRegressor()))
models.append(('DTC',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVC',SVC(gamma='auto')))
# Evaluate each model in turn 
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=7)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)".format(name, cv_results.mean(), cv_results.std())
    print(msg)





