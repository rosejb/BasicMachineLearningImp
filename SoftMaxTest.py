import SoftMax
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import svm


data, target = load_digits(return_X_y=True)

X_train_cv, X_test, y_train_cv, y_test = train_test_split(
    data, target, test_size=0.2, random_state=22)

X_train, X_cv, y_train, y_cv = train_test_split(
    X_train_cv, y_train_cv, test_size=.25, random_state=42)

sm = SoftMax.SoftMaxRegression(X_train, y_train, 10, add_bias_term=True)
sm.fit_lambda(X_cv, y_cv)

test_predictions = sm.predict(X_test)
test_acc = (test_predictions.flatten() == y_test.flatten()).mean()
print('Test set accuracy using SoftMax Regression is ' + str(test_acc))


opt_c = .001
opt_coef = .001
opt_score = 0

for coef in [.001 * 2 ** i for i in range(10)]:
    for C in [.001 * 2 ** j for j in range(10)]:
        poly_svc = svm.SVC(C=C, kernel='poly', degree=3, coef0=coef, tol=1e-3)
        poly_svc.fit(X_train, y_train)
        cur_score = poly_svc.score(X_cv, y_cv)
        if cur_score > opt_score:
             opt_c, opt_coef, opt_score = C, coef, cur_score


poly_svc = svm.SVC(C=opt_c, kernel='poly', degree=2, coef0=opt_coef, tol=1e-3)
poly_svc.fit(X_train, y_train)
poly_svm_acc = poly_svc.score(X_test, y_test)
print('Test set accuracy using 3rd degree polynomial SVM is ' + str(poly_svm_acc))

# Unsuprisingly, sklearn's polynomial svm classifier outperforms classification using softmax
