import SoftMax
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


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
