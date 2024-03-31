from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold

# Set shuffle=True to enable shuffling of data before splitting
#kfold = KFold(n_splits=15, shuffle=True, random_state=None)
# Set shuffle=True to enable shuffling of data before splitting

# load the data
dataset = pd.read_csv('dataset9000.data', header = None)
print(dataset.head())
X=np.array(dataset.iloc[:, 0:17]) 
print(X)
Y = np.array(dataset.iloc[:, 17])
print(Y)
dataset.columns= ["Database Fundamentals","Computer Architecture","Distributed Computing Systems",
"Cyber-Security","Networking","Development","Programming Skills","Project Management",
"Computer Forensics Fundamentals","Technical Communication","AI ML","Software Engineering","Business Analysis",
"Communication skills","Data Science","Troubleshooting-skills","Graphics Designing","Roles"]
dataset.dropna(how ='all', inplace = True)

  
seed =5 
kfold = KFold(n_splits=15, shuffle=True, random_state=None)
#kfold = model_selection.KFold(n_splits = 15, random_state = seed)
  
# initialize the base classifier
base_cls = DecisionTreeClassifier()
  
# no. of base classifier
num_trees = 50
  
# bagging classifier
model = BaggingClassifier(
                          n_estimators = num_trees,
                          random_state = seed)
  
results = model_selection.cross_val_score(model, X, Y, cv = kfold)
print("accuracy :",results.mean()*100)