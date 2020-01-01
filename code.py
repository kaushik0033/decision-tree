# --------------
# Importing Necessary libraries
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
#from sklearn.compose import ColumnTransformer
# Load the train data stored in path variable
train_df=pd.read_csv(path)
# Load the test data stored in path1 variable
test_df=pd.read_csv(path1)
train_df.columns=(c.lower() for c in train_df.columns)
test_df.columns=(c.lower() for c in test_df.columns)
print(train_df.head(5))
print(test_df.head(5))
#test_df.target.value_counts().isnull().sum()
# necessary to remove rows with incorrect labels in test dataset
test_df=test_df[test_df.target.notnull()]
#test_df.target.value_counts(dropna=False)
# encode target variable as integer
train_df=train_df.assign(enc_tar=np.where(train_df.target=='<=50k',0,1))
train_df.drop('target',inplace=True,axis=1)
test_df=test_df.assign(enc_tar=np.where(test_df.target=='<=50k',0,1))
test_df.drop('target',inplace=True,axis=1)


# Plot the distribution of each feature
sns.pairplot(train_df)

# convert the data type of Age column in the test data to int type
test_df.age=test_df.age.astype(int)
test_df.info()
test_df[['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
       'hours_per_week', 'enc_tar']]=test_df.select_dtypes(exclude='object').astype('int')
# cast all float features to int type to keep types consistent between our train and test data
numcols=train_df.select_dtypes(exclude='object').columns
catcols=train_df.select_dtypes(include='object').columns
# choose categorical and continuous features from data and print them
X_train,X_test,y_train,y_test=train_df.iloc[:,:-1],test_df.iloc[:,:-1],train_df.iloc[:,-1],test_df.iloc[:,-1]
numcols=X_train.select_dtypes(exclude='object').columns
catcols=X_train.select_dtypes(include='object').columns
#fit_missingtrans=ColumnTransformer(
#    [('num_fit',Imputer(strategy='mean'),numcols),
#    ('cat_fit',Imputer(strategy='most_frequent'),catcols)])
#fit_missingtrans
from sklearn.base import TransformerMixin

class CustomImputer(TransformerMixin):
    def __init__(self, cols=None, strategy='mean'):
               self.cols = cols
               self.strategy = strategy

    def transform(self, df):
            X = df.copy()
            impute = Imputer(strategy=self.strategy)
            if self.cols == None:
                self.cols = list(X.columns)
            for col in self.cols:
                if X[col].dtype == np.dtype('O'):
                    X[col].fillna(X[col].value_counts().index[0], inplace=True)
                else : X[col] = impute.fit_transform(X[[col]])

            return X

    def fit(self, *_):
               return self

impcat=CustomImputer()
X_train_trans=impcat.fit_transform(X_train)
X_test_trans=impcat.fit_transform(X_test)
coltrans=list(numcols)+list(catcols)
#X_train_trans=pd.DataFrame(X_train_trans+X_train_trans_cat,columns=coltrans)
#X_test_trans=pd.DataFrame(X_test_trans+X_test_trans_cat,columns=coltrans)
X_train_trans=pd.concat([X_train_trans[numcols],pd.get_dummies(X_train_trans[catcols])],axis=1)
X_test_trans=pd.concat([X_test_trans[numcols],pd.get_dummies(X_test_trans[catcols])],axis=1)
dt1=DecisionTreeClassifier(max_depth=3,random_state=17)
X_train_trans,X_test_trans = X_train_trans.align(X_test_trans, join='outer', axis=1, fill_value=0)
dt1.fit(X_train_trans,y_train)
y_pred=dt1.predict(X_test_trans)
#X_test_trans.head(1)
accuracy_score(y_test,y_pred)
# fill missing data for catgorical columns
tree_params={'max_depth':[2,3,4,5,6,7,8,9,10,11]}
dt2=DecisionTreeClassifier(max_depth=tree_params,random_state=17)
gridCV=GridSearchCV(dt2,tree_params,cv=5)
gridCV.fit(X_train_trans,y_train)
print(gridCV.best_params_)
print(gridCV.best_score_)
dt3=DecisionTreeClassifier(max_depth=2,random_state=17)
dt3.fit(X_train_trans,y_train)
y_pred=dt3.predict(X_test_trans)
#X_test_trans.head(1)
accuracy_score(y_test,y_pred,normalize=True)
# fill missing data for numerical columns   


# Dummy code Categoricol features


# Check for Column which is not present in test data


# New Zero valued feature in test data for Holand


# Split train and test data into X_train ,y_train,X_test and y_test data


# train a decision tree model then predict our test data and compute the accuracy


# Decision tree with parameter tuning


# Print out optimal maximum depth(i.e. best_params_ attribute of GridSearchCV) and best_score_


#train a decision tree model with best parameter then predict our test data and compute the accuracy




