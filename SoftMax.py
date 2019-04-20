import numpy as np
from scipy.optimize import minimize, check_grad


class SoftMaxRegression(object):
    def __init__(self, x_mat, y_vals, k, lamb, needs_bias_term=False):

        self.y_vals= y_vals
        self.k = k
        self.lamb = lamb

        if needs_bias_term:
            self.x_mat = np.concatenate((np.zeros((self.x_mat.shape[0], 1)), x_mat), axis=1)
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
        jac += self.lamb * np.concatenate((np.zeros(1, theta.shape[1]), theta), axis=0)

        return jac.flatten('F')



# def train_softmax(X, y, k, lamb):
#     theta = np.zeros(X.shape[1] * k)
#
#     opt_sc = lambda theta: softmax_cost(theta, X, y, k, lamb)
#     opt_jac = lambda theta: softmax_grad(theta, X, y, k, lamb)
#
#
#     result = minimize(opt_sc, theta, method='L-BFGS-B', jac=opt_jac,
#                       options={'gtol':1e-5})
#
#     return result.x.reshape((X.shape[1], k), order='F')
