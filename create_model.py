from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

X,y= datasets.load_iris(return_X_y=True)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)

model= RandomForestClassifier(random_state=101)

model.fit(X_train,y_train)

print("Score on the training set is: {:2}"
      .format(model.score(X_train, y_train)))

print("Score on the test set is: {:.2}"
      .format(model.score(X_test, y_test)))

model_filename='iris-rf-v1.pkl'
print("Saving model to {}....".format(model_filename))
joblib.dump(model,model_filename)