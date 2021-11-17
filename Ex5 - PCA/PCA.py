import pandas as pd 
import numpy as np  
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
#-----importing the data-----
breast_cancer = load_breast_cancer()
#print(breast_cancer.DESCR)

#-----getting the shape of data-----
print("\nShape before PCA : ",breast_cancer.data.shape)
print("\nShape of Target : ",breast_cancer.target.shape)

#-----scalling the data-----
raw_data = breast_cancer.data
df= StandardScaler().fit_transform(raw_data)

#-----applying PCA for data-----
pca = PCA(n_components= 2)
pca_data = pca.fit_transform(df)
#print(pca_data)

print("\nShape after PCA : ",pca_data.shape)

#-----covarience matrix with eigen values-----
features = pca_data.T
covarience_matrix = np.cov(features)
print("\n",covarience_matrix)
eig_vals, eig_vecs = np.linalg.eig(covarience_matrix)
print('\nEigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#-----splitting the data into training And testing data-----
y=breast_cancer.target
X_train,X_test,y_train,y_test=train_test_split(pca_data,y,test_size=0.3,random_state=10)

#-----applying the Logistic Regression model-----
lr=LogisticRegression()
lr.fit(X_train,y_train)
prediction = lr.predict(X_test)
print("\nThe Mean Squared error is :",mean_squared_error(y_test,prediction))

#-----ploting the data ater PCA dimentionality reduction-----
import seaborn as sns 
import matplotlib.pyplot as plt

pca_df = pd.DataFrame(np.vstack((pca_data.T, breast_cancer.target)).T,
                      columns = ['1st_Prin', '2nd_Prin', 'label'])

pca_df['label'].replace(0.0, 'Malignant',inplace=True)
pca_df['label'].replace(1.0, 'Benign',inplace=True)

print(pca_df.label.value_counts())

pal = dict(Malignant="red", Benign="green")

ax = sns.FacetGrid(pca_df, hue='label', height=6, palette=pal,
                   hue_order=["Malignant", "Benign"]).\
                   map(plt.scatter, '1st_Prin', '2nd_Prin').\
                   add_legend()

plt.show()