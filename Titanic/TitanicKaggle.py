#Titanic DataSet accuracy 0.79186
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import cross_val_score

#Reading the Data
train=pd.read_csv('train.csv', index_col='PassengerId')
test=pd.read_csv('test.csv',index_col='PassengerId')
y=train['Survived']
arr=[]
for i in y:
    arr.append(i)
X=train.append(test)

#Filling Empty Values
X['Embarked']=X['Embarked'].fillna('S')
def rem_fare(r):
    if r.Fare==0:
        r.Fare=np.NaN
    return r    
X=X.apply(rem_fare,axis=1)        



X['Fare']=X['Fare'].fillna(X['Fare'].median())
X['Age']=X['Age'].fillna(X.groupby(['Sex','Pclass'])['Age'].transform('median'))

X['Deck'] = X['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
X['Deck']=X['Deck'].replace(['T'],'A')
X['Deck'] = X['Deck'].replace(['A', 'B', 'C'], 'ABC')
X['Deck'] = X['Deck'].replace(['D', 'E'], 'DE')
X['Deck'] = X['Deck'].replace(['F', 'G'], 'FG')

#Transforming the Data
X['Fare']=pd.qcut(X['Fare'],13)
X['Age']=pd.cut(X['Age'],10)

def rel(r):
    return r.SibSp+r.Parch+1
X['Relation']=X.apply(rel,axis=1)
X['Relation']=pd.cut(X['Relation'],[0,1,4,7,11])

X['Ticket'].replace(['LINE'],'0',inplace=True)
X['Ticket_Freq'] = X.groupby('Ticket')['Ticket'].transform('count')
#X['Ticket_Lett']=X.Ticket.apply(lambda x: x[:2])
X['Title']=X['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
X['Title'].replace(['Miss', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Ms', inplace=True)
X['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Other', inplace=True)



non_numeric_features = ['Embarked', 'Sex','Relation','Age','Fare','Title','Deck']

for feature in non_numeric_features:        
    X[feature] = LabelEncoder().fit_transform(X[feature])

encoded_features = []
cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Relation']
for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(X[feature].values.reshape(-1, 1)).toarray()
        n = X[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = X.index
        encoded_features.append(encoded_df)
    

X=pd.concat([X,*encoded_features[:6]], axis=1)
X=X.drop(['SibSp','Parch','Name','Cabin','Ticket','Title','Embarked','Pclass','Relation','Sex','Deck'],axis=1)   



X_train=X.drop(['Survived'],axis=1)
X_train=X_train[:891]
X_test=X.drop(['Survived'],axis=1)
X_test=X_test[891:]
print(X)
rfs= RandomForestClassifier(n_estimators=1750,max_depth=7,min_samples_split=6,min_samples_leaf=6,
                            max_features='auto', oob_score=True, n_jobs=-1, verbose=1,random_state=42)
rfs.fit(X_train,y)
result=rfs.predict(X_test)
rfs.feature_importances_

a=cross_val_score(rfs,X_train,y,cv=6)
print(a)
res=pd.read_csv('gender_submission.csv',index_col=None)
res['Survived']=result
res.to_csv('SubFinal4.csv',index=False)