import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.spatial import distance_matrix
from scipy.optimize import lsq_linear
from scipy.special import psi, polygamma, betainc, gamma

class DirichletGOF:
    def __init__(self, significance_level, dim, sample_size, num_iter, energy_dist_pow=1):
        self.significance_level = significance_level
        self.sample_size = sample_size
        self.num_iter = num_iter
        self.dim = dim
        self.energy_dist_pow = energy_dist_pow

        self.dir_dist = self.avg_dirichlet_distance()
        self.crit_value = self.critical_value()

    def mle_alpha(self, x, max_iter=200):
        '''Given a set of points x, compute minimum likelihood estimation of Dirichlet concentration parameters '''
        n = x.shape[0]
        alpha = initial_params(x)
        prev = alpha

        for idx in range(max_iter):
        grad = n * (psi(np.sum(alpha)) - psi(alpha)) + np.sum(np.log(x), axis=0)
        q = -n * trigamma(alpha)
        c = n * trigamma(np.sum(alpha))
        b = np.sum(grad/q) / ((1/c) + np.sum(1/q))
        alpha = alpha - (grad - b)/q

        if np.any(alpha<0):
          return prev
        else:
          prev = alpha

        return alpha

    def dirichlet_transform(self, x, alpha):
        '''Transform Dirichlet variable X with parameters alpha into standard Dirichlet distribution'''
        y = np.zeros((x.shape[0], x.shape[1]))
        x_ = np.zeros((x.shape[0], x.shape[1]))

        num_samples = x.shape[0]
        n = x.shape[1]-1

        for i in range(num_samples):
        up_lim = x[i][0]
        y[i][0] = betainc(alpha[0], np.sum(alpha[1:]), up_lim)
        x_[i][0] = 1-np.power(1-y[i][0], 1/n)

        for j in range(1, n):
          remainder = 1 - np.sum(x[i][:j])
          if remainder == 0:
            up_lim = np.float64(1)
          else:
            up_lim = x[i][j] / remainder

          y[i][j] = betainc(alpha[j], np.sum(alpha[j+1:]), up_lim.clip(min=0, max=1))
          x_[i][j] = np.prod([np.power(1-y[i][k], 1/(n-k)) for k in range(j)]) * (1-np.power(1-y[i][j], 1/(n-j)))

        x_[:, -1] = 1 - np.sum(x_[:, :-1], axis=1)

        return x_

    def avg_dirichlet_distance(self, num_samples=1e6):
        '''Simulates E|Z_1 - Z_2| where Z_1 and Z_2 are standard Dirichlet distributed variables'''
        z1 = np.random.dirichlet(alpha=[1]*self.dim, size=num_samples)
        z2 = np.random.dirichlet(alpha=[1]*self.dim, size=num_samples)
        dirichlet_dist = np.mean(np.linalg.norm(z1-z2, axis=1))
        return dirichlet_dist

    def critical_value(self):
        '''Compute critical value of test depending on significance level'''
        x = np.random.dirichlet(alpha=[1]*self.dim, size=self.sample_size)
        alpha_mle = self.mle_alpha(x)
        x = self.dirichlet_transform(x, alpha_mle)

        tests = []
        for i in range(20000):
            z = np.random.dirichlet(alpha=[1]*self.dim, size=self.num_samples)
            tests.append(np.sum(np.mean(np.linalg.norm(x[:, None, :] - z[None, :, :], axis=-1), axis=1)))

        sample_dirichlet_dist = np.quantile(tests, q=self.significance_level)
        sample_dists = np.sum(np.power(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1), self.energy_dist_pow))
        statistic = sample_size * ( ((2/sample_size) * sample_dirichlet_dist) - self.dir_dist - ((1/sample_size**2) * sample_dists))

        return statistic

    def energy_statistic(self, x):
        '''Compute test statistic of a given sample x'''
        sample_size = x.shape[0]
        n = x.shape[1]

        tests = []
        for i in range(20000):
        z = np.random.dirichlet(alpha=[1]*self.dim, size=n)
        tests.append(np.sum(np.mean(np.linalg.norm(x[:, None, :] - z[None, :, :], axis=-1), axis=1)))

        sample_dirichlet_dist = np.mean(tests)
        sample_dists = np.sum(np.power(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1), d))
        statistic = sample_size * ( ((2/sample_size) * sample_dirichlet_dist) - self.dir_dist - ((1/sample_size**2) * sample_dists))

        return statistic

    def compute_power(self, x, log=False):
        '''Compute power of Dirichlet GOF test'''

        if log:
            print('Using critical value at %.2f level of significance = %.4f' % (self.significance_level, self.crit_value))

        test_stats_list = []
        for i in range(n_iter):
            x_sample = x[np.random.choice(np.linspace(0, x.shape[0]-1, x.shape[0]).astype(np.int32), size=sample_size, replace=False)]
            alpha = self.mle_alpha(x_sample)
            x_sample_transform = self.dirichlet_transform(x_sample, alpha)

            test_stat = self.energy_statistic(x_sample_transform, self.dir_dist)
            test_stats_list.append(test_stat)
            if print_log:
              print('Finished iteration', str(i))
              print('Test statistic:', test_stat)
              print()

        return test_stats_list, sum(i > self.crit_value for i in stats) / n_iter
