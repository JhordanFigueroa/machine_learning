#BEST WAY TO MEASURE ACCURACY OF MODEL 

#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.construct import rand

#IMPORT DATASET
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#SPLITTING TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#FEATURE SCALING 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#TRAIN KERNEL SVM MODEL ON TRAINING SET
from sklearn.svm import SVC
model_svc = SVC(kernel='rbf', random_state=0)
model_svc.fit(X_train, y_train)

#CONFUSION MATRIX 
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = model_svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
#APPLY K-FOLD CROSS VALIDATION 
#Evaluates model on K tests - gives a more accuracy estimate of models bc tested k times
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=model_svc, X = X_train, y = y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

#VISUALIZE TRAINING SET RESULTS 
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model_svc.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#VISUALIZE TEST SET RESULTS 