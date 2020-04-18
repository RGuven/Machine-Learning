from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



digits = datasets.load_digits()
print(digits.keys())
print(digits.DESCR)


print(digits.images.shape)
print(digits.data.shape)

plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()



X = digits.data
y = digits.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42,stratify=y)


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)


print(knn.score(X_test, y_test))
