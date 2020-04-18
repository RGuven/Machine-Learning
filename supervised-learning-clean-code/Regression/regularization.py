# Import Lasso (L1)
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.4,normalize=True)


lasso.fit(X,y)

lasso_coef = lasso.coef_
print(lasso_coef)

#----------------------------------------------------
##(L2)
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []


ridge = Ridge(normalize=True)

for alpha in alpha_space:
    ridge.alpha = alpha
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))

display_plot(ridge_scores, ridge_scores_std)