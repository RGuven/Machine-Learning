from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

reg = LinearRegression()


cv_scores = cross_val_score(reg,X,y,cv=5) #cv=5 changeable

print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
