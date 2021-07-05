import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("data/fetal_health.csv")

predictors = data[["baseline value", "accelerations", "fetal_movement", "uterine_contractions", "light_decelerations", "prolongued_decelerations", "histogram_width"]]
targets = data.fetal_health
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size = .4, random_state = 10)

classifier = RandomForestClassifier(n_estimators = 300)
classifier = classifier.fit(pred_train, tar_train)

predictions = classifier.predict(pred_test)
print(predictions[20])

accuracy = sklearn.metrics.accuracy_score(tar_test, predictions)
print(accuracy)

model = ExtraTreesClassifier()
model.fit(pred_train, tar_train)

print(model.feature_importances_)
