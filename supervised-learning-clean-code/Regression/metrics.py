from sklearn.metrics import confusion_matrix,classification_report
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

knn = KNeighborsClassifier(6)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

# Confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))