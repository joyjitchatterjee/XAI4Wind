# -*- coding: utf-8 -*-
"""KG-ResourcePaper_New.py
Script for integration of KG with XAI model
"""

!pip install py2neo

from matplotlib import rc
from IPython.display import clear_output

from py2neo import Graph
graph = Graph("bolt://localhost:11005", auth=("neo4j", "reset123")) #Can look up port address from inside Neo4j (11005 at present)

#Sample query
graph.run("MATCH (a:Feature {feature_no:0})-[:RELATESTO]-(p) RETURN a,p").to_table()

import pandas as pd
df = pd.read_csv("/Users/joyjitchatterjee/Desktop/df_ldt_per_engie_finalcsvfile.csv")
df.head()

df["FunctionalGroup"] = df["FunctionalGroup"].map({
    "NoFault": 0,
    "Partial Performance-Degraded": 1,
    "Pitch System Interface Alarms": 2,
    "Gearbox": 3,
    "Pitch System EFC Monitoring": 4,
    "PCS": 5,
    "MVTR": 6,
    "Yaw Brake":7,
    "Hydraulic System": 8,
    "Yaw" : 9,
    "Wind Condition Alarms": 10,
    "Pitch" : 11,
    "IPR" : 12,
    "Test" : 13
}).astype(int)

X= df.iloc[:,1:-1]
y =df.iloc[:,-1]

df

#Training and testing of XGBoost classifier model
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
import matplotlib.pylab as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


# create a train/test split
import joblib

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.30)

xgbc = xgb.XGBClassifier(n_estimators = 100,learning_rate = 0.1,early_stopping_rounds = 10,objective='multi:softprob',\
                    num_class=14,\
                    random_state=123)

# mcl = xgbc.fit(X_train, y_train, eval_metric='mlogloss')

# X_train.to_csv('X_train_KG.csv')
# y_train.to_csv('y_train_KG.csv')
# X_test.to_csv('X_test_KG.csv')
# y_test.to_csv('y_test_KG.csv')
# X_train= pd.read_csv('X_train_KG.csv')
# y_train = pd.read_csv('y_train_KG.csv')
# X_test = pd.read_csv('X_test_KG.csv')
# y_test = pd.read_csv('y_test_KG.csv')

# joblib.dump(mcl, 'XGB_KGResourcePaper.dat') 
# mcl.save_model('XGB_KGResourcePaper.model')

#Use this script to load the model later (when needed)
mcl = joblib.load("XGB_KGResourcePaper.dat")
mcl

#Predict faults transparency using SHAP

pred = mcl.predict(X_test)
proba = mcl.predict_proba(X_test)

# Print model report
print("Classification report (Test): \n")
print(metrics.classification_report(y_test, pred))
# print("Confusion matrix (Test): \n")
# print(metrics.confusion_matrix(y_test, pred)/len(y_test))

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
print(accuracy_score(y_test, pred))
# Creates a confusion matrix
cm = confusion_matrix(y_test, pred) 

# Transform to df for easier plotting
# cm_df = pd.DataFrame(cm,
#                      index = ['NoFault','Partial Performance-Degraded','Pitch System Interface Alarms','Gearbox','Pitch System EFC Monitoring','PCS','MVTR','Yaw Brake','Hydraulic System','Yaw','Wind Conditon Alarms','Pitch','IPR','Test'], 
#                      columns = ['NoFault','Partial Performance-Degraded','Pitch System Interface Alarms','Gearbox','Pitch System EFC Monitoring','PCS','MVTR','Yaw Brake','Hydraulic System','Yaw','Wind Conditon Alarms','Pitch','IPR','Test'])

# sns.heatmap(cm_df)
# plt.title('Predicted faults in the simulated offshore wind farm\nAccuracy:{0:.3f}'.format(accuracy_score(y_test, pred)))
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.savefig('ConfusionMatrix_WindFarm.jpg')
# plt.show()

#This is simply to check at what indices a particular class (fault type occurs), so that we can
#appropriately specify the current_sample_totest
faultclass_totest = 2
result = np.where(y_test == faultclass_totest)
print(result)
result = np.array(result)
result = result.flatten()

import random #to select a random fault sample
# use SHAP to explain test set predictions
current_sample_totest = int(random.choice(result)) #define a current sample to test (nth observation from the test set)

import shap
explainer = shap.TreeExplainer(mcl)
shap_values = explainer.shap_values(X_test.iloc[current_sample_totest:current_sample_totest+1,:]) #could try a cllection of test set values using e.g. x1:z2,:

y_test.iloc[current_sample_totest]

print(current_sample_totest) #Initially sample 2612 was our test case for gearbox shap experiment

from pylab import rcParams
import seaborn as sns
rcParams['figure.figsize'] = 14, 8

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

#Multiclass- SHAP classification code based on blog article by Evgeny Pogorelov (https://evgenypogorelov.com/multiclass-xgb-shap.html)

shap.initjs()
# plot the SHAP values for the output of the specific fault class instance
shap.force_plot(explainer.expected_value[faultclass_totest], shap_values[faultclass_totest][0,:], X_test.iloc[current_sample_totest,:], link="logit")
plt.savefig('Pitch_LDT_Case2.eps',bbox_inches='tight',dpi=1000)

# shap.initjs()

# # plot the SHAP values for the i-th output class of all instances
# shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit")

# shap.dependence_plot(0, shap_values[0], X) #Feature no. specified here

# shap_values = explainer.shap_values(X)
# shap.summary_plot(shap_values, X)

import numpy as np
vals= np.abs(shap_values).mean(0)

feature_importance = pd.DataFrame(list(zip(X_test.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
feature_importance.head()

feature_importance.to_csv("PitchKGExample2.csv")

feature_importance.reset_index(drop=True, inplace=True)

for i in range(10):
  imp_try = feature_importance['col_name'][i]
  imp_try = str(imp_try)
  print(imp_try)
  
  # query = "MATCH(n {name:$name})-[:RELATESTO]->(p) RETURN n,p"

  # query2 = "MATCH(n {name:$name})-[:RELATESTO]->(p)-[:FOR]-(q) RETURN n,p,q"
  query2 = "MATCH(n:Corrective)-[:ACTION]->(p)-[:FOR]->(q)-[:RELATESTO]-(r:Feature{name:$name}) RETURN p,q,r"
  query3 = "MATCH(n:Corrective)-[:ACTION]->(p)-[:FOR]->(q)-[:AFFECTS]-(r{fno:$fno}) RETURN p,q,r"

  # print(graph.run(query, parameters= {"name": imp_try}).data())
  print("General Corrective Maintenance Action based on identified feature importances (not specific to the component):\n")
  print(graph.run(query2, parameters= {"name": imp_try}).data())
  print("Corrective Maintenance Action based on identified feature importances specific to currently predicted fault:\n")
  print(graph.run(query3, parameters= {"fno": int(y_test.iloc[current_sample_totest])}).data())

#Preventive actions for current component failure
query4 = "MATCH(n:Preventive)-[:ACTION]->(p)-[:FOR]->(q{fno:$fno}) RETURN p,q"
query5 = "MATCH(n:Preventive)-[:ACTION]->(p)-[:FOR]->(q)-[:CONSISTSOF]-(r {fno:$fno}) RETURN p,q,r"

# print(graph.run(query, parameters= {"name": imp_try}).data())
print("Preventive Maintenance Actions:")
print(graph.run(query4, parameters= {"fno": int(y_test.iloc[current_sample_totest])}).data())
print(graph.run(query5, parameters= {"fno": int(y_test.iloc[current_sample_totest])}).data())
