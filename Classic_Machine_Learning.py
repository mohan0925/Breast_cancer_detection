#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


# example of grid searching key hyperparameters for RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


print('\nPART I – Application: Load and overview data related to your theme')
print('===========================================================================')
#PART I – Application: Load and overview data related to your theme

#Importing the data
df = pd.read_csv('data.csv',header = 0)

#dropping all the null rows from the data
df = df.dropna(axis=1)

#Displaying the data
print('\n\tLoading the first 5 rows of data:\n',df.head())

#Identifing key aspects of data
print("\n\t The data frame has {0[0]} rows and {0[1]} columns.".format(df.shape))
print("\tNumber of Dimensions:",df.shape)

print("\n\t The data has {} diagnosis, {} benign and {} malignant.".format(df.shape[0],df['diagnosis'].value_counts()[0],df['diagnosis'].value_counts()[1]))
print("\tNumber of Classes:\n",df['diagnosis'].value_counts())

df['diagnosis'].value_counts().plot(kind="bar")
plt.show()

print("\n\tDatatype of variables:\n",df.dtypes)

#dropping the id column from the data
df = df.drop('id',axis=1)

#Mapping diagnosis column M:1 and B:0
diag_map = {'M':1, 'B':0}
df['diagnosis'] = df['diagnosis'].map(diag_map)

All_columns_data_values = df.iloc[:, 1:31].values
Diagnosis_column_values = df.iloc[:, 0].values

# Plot histograms for each variable to check for continuous data
df.hist(figsize = (18, 18))
plt.show()

print('===========================================================================')

print('\nPART II – Application: Clustering')
print('===========================================================================')
#PART II – Application: Clustering

#Creating a 2D visualization to visualize the clusters
from sklearn.manifold import TSNE
tsne = TSNE(verbose=1, perplexity=40, n_iter= 4000)
Array_x_y_values = tsne.fit_transform(All_columns_data_values)

# a)	You should use at least two clustering methods to partition the dataset into two clusters.

print('\n\t a)Two clustering methods to partition the dataset into two clusters')
#Cluster using k-means
from sklearn.cluster import KMeans
kmns = KMeans(n_clusters=2, init='k-means++', n_init=1, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
kmns.fit(All_columns_data_values)
kmeans_values_0_1 = kmns.predict(All_columns_data_values)
label1=kmeans_values_0_1

# Cluster using AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
aggClustering = AgglomerativeClustering(n_clusters=2,affinity='euclidean', linkage='ward')
agglomerative_values_0_1=aggClustering.fit_predict(All_columns_data_values)
label2=agglomerative_values_0_1

print('\nBlue points indicate=Benign')
print('Red points indicate=Malignent')

# Creates two subplots and unpacks the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(Array_x_y_values[:,0],Array_x_y_values[:,1],c=kmeans_values_0_1, cmap = "jet", edgecolor = "None", alpha=0.30)
ax1.set_title('k-means clustering')

ax2.scatter(Array_x_y_values[:,0],Array_x_y_values[:,1],c = Diagnosis_column_values, cmap = "jet", edgecolor = "None", alpha=0.30)
ax2.set_title('Actual cluster')
    

# Creates two subplots and unpacks the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(Array_x_y_values[:,0],Array_x_y_values[:,1],c=agglomerative_values_0_1, cmap = "jet", edgecolor = "None", alpha=0.30)
ax1.set_title('Agglomerative clustering')

ax2.scatter(Array_x_y_values[:,0],Array_x_y_values[:,1],c = Diagnosis_column_values, cmap = "jet", edgecolor = "None", alpha=0.30)
ax2.set_title('Actual cluster')
plt.show()
#######################################################################################################################################
#b)	Evaluate the clustering methods using appropriate metrics such as the Adjusted Rand index, Homogeneity, Completeness and V-Measure, using the ground truth.

from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import adjusted_rand_score

rand_index=adjusted_rand_score(Diagnosis_column_values, label1)
homogenity=homogeneity_score(Diagnosis_column_values, label1)
v_scores= v_measure_score(Diagnosis_column_values, label1)
completeness=completeness_score(Diagnosis_column_values, label1)

print('\n\t b)Metrics for K-Means Clustering Method')
print('\nrandom index= ',rand_index)
print('homogenity= ',homogenity)
print('V-measure score= ',v_scores)
print('completeness= ',completeness)
###############################################################################################
rand_index=adjusted_rand_score(Diagnosis_column_values, label2)
homogenity=homogeneity_score(Diagnosis_column_values, label2)
v_scores= v_measure_score(Diagnosis_column_values, label2)
completeness=completeness_score(Diagnosis_column_values, label2)

print('\n\t b)Metrics for Agglomerative Clustering Method')
print('\nrandom index= ',rand_index)
print('homogenity= ',homogenity)
print('V-measure score= ',v_scores)
print('completeness= ',completeness)
################################################################################################
print('\n\t c)Overloading the parameters of K-Means clustering method\n')

# List of V-Measure Scores for different models 
v_scores = [] 
score=0
V_score_high=0
V_score_high_cluster=0

v_scores1 = [] 
score1=0
V_score_high1=0
V_score_high1_cluster=0
 
# List of different types of covariance parameters 
N_Clusters = [2,4,6,8,10] 

for x in N_Clusters:
    # Building the clustering model 
    kmeans = KMeans(n_clusters = x, init='k-means++', n_init=1, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')   
    # Training the clustering model 
    kmeans.fit(All_columns_data_values)   
    # Storing the predicted Clustering labels 
    labels = kmeans.predict(All_columns_data_values)   
    # Evaluating the performance 
    v_scores.append(v_measure_score(Diagnosis_column_values, labels)) 
    
    score=v_measure_score(Diagnosis_column_values, labels)
    if score > V_score_high:
        V_score_high = score
        V_score_high_cluster=x
    print('Number of clusters = {}, V-measure score = {}'.format(x,v_measure_score(Diagnosis_column_values, labels)))

print('\nBest V-measure Score:{} for Number of clusters: {}'.format(V_score_high,V_score_high_cluster))

# Plotting a Bar Graph to compare the models 
plt.bar(N_Clusters, v_scores) 
plt.xlabel('Number of Clusters') 
plt.ylabel('V-Measure Score') 
plt.title('K-Means V score') 
plt.show()


print('\n\tc)Overloading the parameters of Agglomerative clustering method\n')
for x in N_Clusters:
    aggClustering = AgglomerativeClustering(n_clusters=x,affinity='euclidean', linkage='ward')
    kY = aggClustering.fit_predict(All_columns_data_values)
    v_scores1.append(v_measure_score(Diagnosis_column_values, kY)) 
    score1=v_measure_score(Diagnosis_column_values, kY)
    if score1 > V_score_high1:
        V_score_high1 = score1
        V_score_high1_cluster=x
    print('Number of clusters = {}, V-measure score = {}'.format(x,v_measure_score(Diagnosis_column_values, kY)))

print('\nBest V-measure Score:{} for Number of clusters: {}'.format(V_score_high1,V_score_high1_cluster))
    
#Visualizing the results and comparing the performances

# Plotting a Bar Graph to compare the models 
plt.bar(N_Clusters, v_scores1) 
plt.xlabel('Number of Clusters') 
plt.ylabel('V-Measure Score') 
plt.title('AgglomerativeClustering V score') 
plt.show()

print('===========================================================================')

print('\nPART III – Application: Classification: Training and Testing')
print('===========================================================================')
#PART III – Application: Classification: Training and Testing

#a)	You should use at least two classification methods to distinguish between the classes. Both the following training/testing protocols should be used:
#	Split the data into training (70%) and testing (30%).
#	K-fold cross-validation for K=10.

#Split the data into training (70%) and testing (30%).
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(All_columns_data_values, Diagnosis_column_values, test_size = 0.30, random_state = 0)

from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

#Using Random Forest Classifier Algorithm to the Training Set

#Using RandomForestClassifier 
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, Y_train)  

#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier 
decision_tree_Classifier = DecisionTreeClassifier(criterion = 'entropy',splitter='best', random_state = 0)
decision_tree_Classifier.fit(X_train, Y_train)

#print model accuracy on the training data.
print('\n\tTraining Data Accuracy for different classifiers:\n')
print('\nRandom Forest Classifier Training Accuracy:', rf.score(X_train, Y_train))
print('Decision Tree Classifier Training Accuracy:', decision_tree_Classifier.score(X_train, Y_train))
print()

print('\n\tTraining Data K-Fold Score for different classifiers:')
from sklearn.model_selection import cross_val_score
K_fold_cross_validation=cross_val_score(RandomForestClassifier(n_estimators=40),All_columns_data_values, Diagnosis_column_values,cv=10)

for score in K_fold_cross_validation:
    print('Random Forest Classifier k-fold score:{}'.format(np.mean(score)))
print()

decision_tree_Classifier_cross_val_score=cross_val_score(DecisionTreeClassifier(criterion = 'entropy',splitter='best', random_state = 0), All_columns_data_values, Diagnosis_column_values,cv=10)

for score in decision_tree_Classifier_cross_val_score:
    print('decision tree Classifier k-fold score:{}'.format(np.mean(score)))


# b) evaluate the classification approaches using appropriate metrics such as the Balanced Accuracy, F1-Score, ROC AUC, and drawing ROC curves and a confusion matrices.
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

#Classification method (Random Forest Classifier) Metrics  
predictions = rf.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
cm = confusion_matrix(Y_test, rf.predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
Precision= TP/(TP+FP)
Recall=TP/(TP+FN)
f1_score=(2*((Precision*Recall)/(Precision+Recall)))
probs = rf.predict_proba(X_test)
probs=probs[:,1]
auc_score = roc_auc_score(Y_test, probs)

print('\n\tRandom Forest Classifier Metrics:')
print('Random Forest Classifier Testing Accuracy = ',accuracy)
print('Random Forest Classifier Testing F1-score = ',f1_score)
print('Random Forest Classifier Testing ROC AUC = ',auc_score)
print('Random Forest Classifier Testing Confusion Matrix:\n',cm)
print()

#Plotting Random Forest Classifier ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, probs)
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('Random Forest Classifier: ROC curves ')
#show the legend
pyplot.legend()
pyplot.show()

#Classification method (DecisionTreeClassifier) Metrics  
predictions = decision_tree_Classifier.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
con_matrix = confusion_matrix(Y_test, predictions)
TN = con_matrix[0][0]
TP = con_matrix[1][1]
FN = con_matrix[1][0]
FP = con_matrix[0][1]
Precision= TP/(TP+FP)
Recall=TP/(TP+FN)
f1_score=(2*((Precision*Recall)/(Precision+Recall)))
decision_probs = decision_tree_Classifier.predict_proba(X_test)
decision_probs=decision_probs[:,1]
auc_score = roc_auc_score(Y_test, decision_probs)

print('\n\tDecision Tree Classifier Metrics:')
print('DecisionTreeClassifier Testing Accuracy = ',accuracy)
print('DecisionTreeClassifier Testing F1-score = ',f1_score)
print('DecisionTreeClassifier Testing ROC AUC = ',auc_score)
print('DecisionTreeClassifier Testing Confusion Matrix:\n',con_matrix)
print()

#Plotting nnDecisionTreeClassifier ROC curve
false_positive_rate, true_positive_rate, threshold = roc_curve(Y_test, decision_probs)
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
pyplot.plot(false_positive_rate, true_positive_rate, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('Decision Tree Classifier: ROC curves')
#show the legend
pyplot.legend()
pyplot.show()

#c)	Consider and implement any configuration of the parameters of your classification methods that could further improve the results.

# define models and parameters
model = RandomForestClassifier()
n_estimators = [20, 30, 40, 45, 50]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(All_columns_data_values, Diagnosis_column_values)

# summarize results
print("\n Overloading the parameters of RandomForestClassifier classification for best results\n\n \t Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
# define models and parameters
model = DecisionTreeClassifier()
max_features = ['sqrt', 'log2']
criterion=['gini','entropy']
splitter=['best','random']

# define grid search
grid = dict(max_features=max_features,criterion=criterion,splitter=splitter)
cv = RepeatedStratifiedKFold(n_splits=10)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(All_columns_data_values, Diagnosis_column_values)
# summarize results
print("\n Overloading the parameters of DecisionTreeClassifier classification for best results\n\n \t Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
print('===========================================================================')