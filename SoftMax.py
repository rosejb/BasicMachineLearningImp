import numpy as np
from scipy.optimize import minimize


class SoftMaxRegression(object):
    def __init__(self, x_mat, y_vals, k, lamb=.08, add_bias_term=False):

        self.y_vals = y_vals
        self.k = k
        self.lamb = lamb
        self.opt_theta = None
        self.bias_term_added = add_bias_term

        if add_bias_term:
            self.x_mat = np.concatenate((np.zeros((x_mat.shape[0], 1)), x_mat), axis=1)
        else:
            self.x_mat = x_mat

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
        jac += self.lamb * np.concatenate((np.zeros((1, theta.shape[1])), theta[1:]), axis=0)

        return jac.flatten('F')

    def train(self):
        theta = np.zeros((self.x_mat.shape[1], self.k))

        result = minimize(self.softmax_cost, theta, method='L-BFGS-B', jac=self.softmax_grad,
                          options={'gtol':1e-5})

        self.opt_theta = result.x.reshape((self.x_mat.shape[1], self.k), order='F')

    def predict(self, new_x_mat, add_bias_term=None):
        needs_bias_term = add_bias_term if add_bias_term is not None else self.bias_term_added

        if needs_bias_term:
            new_x_mat = np.concatenate([np.ones((new_x_mat.shape[0], 1)), new_x_mat], axis=1)

        return np.argmax(new_x_mat@self.opt_theta, axis=1)

    def fit_lambda(self, cv_feature_mat, cv_labels):
        max_cv_acc = 0
        max_lamb = .01

        for l in [.01 * 2**i for i in range(8)]:
            self.lamb = l
            self.train()

            cv_predict = self.predict(cv_feature_mat)
            cv_accuracy = (cv_predict.flatten() == cv_labels.flatten()).mean()
            if cv_accuracy > max_cv_acc:
                max_lamb = l
                max_cv_acc = cv_accuracy

        self.lamb = max_lamb
        return max_lamb, max_cv_acc

