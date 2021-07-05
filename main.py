import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("data/heart.csv")

predictors = data[["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa", "thall"]]
targets = data.output
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size = .4, random_state = 10)

print(pred_train.shape)
print(pred_test.shape)
print(tar_train.shape)
print(tar_test.shape)

classifier = RandomForestClassifier(n_estimators = 300)
classifier = classifier.fit(pred_train, tar_train)

predictions = classifier.predict(pred_test)
print(predictions[20])

print(sklearn.metrics.confusion_matrix(tar_test, predictions))

accuracy = sklearn.metrics.accuracy_score(tar_test, predictions)
print(accuracy)

model = ExtraTreesClassifier()
model.fit(pred_train, tar_train)

print(model.feature_importances_)
