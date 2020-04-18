from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd


y = df["blabla"].values
X = df.drop("blabla",axis=1).values

knn = KNeighborsClassifier(n_neighbors=6)


knn.fit(X,y)
y_pred = knn.predict(X)

new_prediction = X_new
new_prediction=knn.predict(X_new)
print("Prediction: {}".format(new_prediction))
