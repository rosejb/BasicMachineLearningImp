import numpy as np
from scipy.optimize import minimize, check_grad


class SoftMaxRegression(object):
    def __init__(self, x_mat, y_vals, k, lamb, needs_bias_term=False):

        self.y_vals = y_vals
        self.k = k
        self.lamb = lamb
        self.theta = np.zeros((x_mat.shape[1], k))

        if needs_bias_term:
            self.x_mat = np.concatenate((np.zeros((self.x_mat.shape[0], 1)), x_mat), axis=1)
        else:
            self.x_mat = x_mat

        self.train(self.theta)

    def softmax_cost(self, theta):
        theta = theta.reshape((self.x_mat.shape[1], self.k), order='F')
        exp_mat = np.exp(self.x_mat@theta)
        p_mat = np.divide(exp_mat, np.sum(exp_mat, axis=1)[:, None])
        lp_mat = np.log(p_mat)

        j = -1 / self.x_mat.shape[0] * np.sum(lp_mat[range(self.x_mat.shape[0]), self.y_vals])
        j += self.lamb / 2 * np.sum(theta[1:]**2)

        return j

    def softmax_grad(self, theta):
        theta = theta.reshape((self.x_mat.shape[1], self.k), order='F')

        exp_mat = np.exp(self.x_mat@theta)
        p_mat = np.divide(exp_mat, np.sum(exp_mat, axis=1)[:, None])

        t_mat = np.zeros((self.x_mat.shape[0], self.k), dtype=int)
        t_mat[range(t_mat.shape[0]), self.y_vals] = 1

        adjust_mat = t_mat - p_mat
        jac = np.zeros(theta.shape)

        for i in range(self.x_mat.shape[0]):
            jac += np.multiply(adjust_mat[i], self.x_mat[i][:, None])

        jac = (-1 / self.x_mat.shape[0]) * jac
        jac += self.lamb * np.concatenate((np.zeros(1, theta.shape[1]), theta), axis=0)

        return jac.flatten('F')

    def train(self):
        theta = np.zeros(self.x_mat.shape[1], self.k)

        result = minimize(self.softmax_cost, theta, method='L-BFGS-B', jac=self.softmax_grad,
                          options={'gtol':1e-5})

        self.theta = result.x.reshape((self.x_mat.shape[1], self.k), order='F')

    def predict(self, x):
        return np.argmax(x@self.theta)

    def fit_lambda(self, cv_feature_mat, cv_labels):
        max_cv_acc = 0
        max_lamb = .01

        for l in [.01 * 2**i for i in range(8)]:
            self.lamb = l
            self.train()

            cv_predict = np.argmax(cv_feature_mat@self.theta, axis=1)
            cv_accuracy = np.equals(cv_predict.flatten(), cv_labels.flatten()).mean()
            if cv_accuracy > max_cv_acc:
                max_lamb = l
                max_cv_acc = cv_accuracy

        self.lamb = max_lamb
        return max_lamb, max_cv_acc

