import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.spatial import distance_matrix
from scipy.optimize import lsq_linear
from scipy.special import psi, polygamma, betainc, gamma
import threading

class DirichletGOF:
    def __init__(self, significance_level, dim, sample_size, crit_val_arr=None):
        self.significance_level = significance_level
        self.sample_size = sample_size
        self.dim = dim
  
        self.dir_dist = self.dirichlet_dist(self.dim)

        self.Dir = self.symm_dirichlet(1)

        '''
        if crit_val_arr is None:
            self.crit_value = self.critical_value()
        else:
            self.crit_value = np.quantile(crit_val_arr, self.significance_level)
        '''
        self.crit_value = 0

    class symm_dirichlet:
        def __init__(self, alpha, resolution=2**16):
            self.alpha = alpha
            self.resolution = resolution
            self.range, delta = np.linspace(0, 1, resolution,
                                            endpoint=False, retstep=True)
            self.range += delta / 2
            self.table = special.gammaincinv(self.alpha, self.range)

        def draw(self, n_sampl, n_comp, interp='nearest'):
            if interp != 'nearest':
                raise NotImplementedError
            gamma = self.table[np.random.randint(0, self.resolution, (n_sampl, n_comp))]
            return gamma / gamma.sum(axis=1, keepdims=True)
        
    def initial_params(self, x):
        x1 = np.mean(x, axis=0)
        x2 = np.mean(x[:, 0]**2)

        alpha_init = x1 * (x1[0] - x2) / (x2 - x1[0]**2)

        return alpha_init

    def trigamma(self, x):
        return polygamma(1, x)

    def mle_alpha(self, x, max_iter=150, eps=1e-7):
        n = x.shape[0]
        alpha = self.initial_params(x)
        prev = alpha

        for idx in range(max_iter):
            grad = n * (psi(np.sum(alpha)) - psi(alpha)) + np.sum(np.log(x), axis=0)

            q = -n * self.trigamma(alpha)
            c = n * self.trigamma(np.sum(alpha))
            b = np.sum(grad/q) / ((1/c) + np.sum(1/q))

            alpha = alpha - (grad - b)/q

            displacement = alpha - prev
            dist = np.sqrt(np.einsum('i,i', displacement, displacement))

            if np.any(alpha<0):
                return prev

            if dist < eps:
                break
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

    def dirichlet_dist(self, alpha_dim, num_samples=1000000):
        '''Simulates E|Z_1 - Z_2| where Z_1 and Z_2 are standard Dirichlet distributed variables'''
        z1 = np.random.dirichlet(alpha=[1.]*alpha_dim, size=num_samples)
        z2 = np.random.dirichlet(alpha=[1.]*alpha_dim, size=num_samples)
        dirichlet_dist = np.mean(np.sqrt(np.einsum('ij,ij->i', z1-z2, z1-z2)))

        return dirichlet_dist


    def critical_value(self, num_samples=20000, num_threads=4):
        x_samples = np.random.dirichlet([1]*self.dim, (num_samples, self.sample_size))

        threads = list()
        subdiv = int(num_samples / num_threads)

        statistics_arr = np.zeros((num_samples,))

        def compute_statistics(samples, start):
            for i in range(subdiv):
                z = self.Dir.draw(20000, self.dim)
                alpha_mle = self.mle_alpha(samples[i])
                samples[i] = self.dirichlet_transform(samples[i], alpha_mle)

                x = samples[i]
                x_ = x[:, None, :]
                sample_dist = np.sum(np.sqrt(np.einsum('ijk, ijk->ij', x-x_, x-x_)))
                sample_dirichlet_dist = np.sum(np.mean(np.sqrt(np.einsum('ijk->ij',(x_ - z)**2)), axis=1))
                statistic = (2 * sample_dirichlet_dist) - (self.sample_size * self.dir_dist) - ((1/self.sample_size) * sample_dist)

                statistics_arr[start+i] = statistic
     
        for index in range(num_threads):
            x = threading.Thread(target=compute_statistics, args=(x_samples[index * subdiv: (index+1)*subdiv], index * subdiv,))
            threads.append(x)
            x.start()

        for thread in threads:
            thread.join()

        statistics_arr = statistics_arr[~np.isnan(statistics_arr)]
        crit_val = np.quantile(statistics_arr, self.significance_level)

        return crit_val, statistics_arr

    def energy_statistic(self, x):
        input_size = x.shape[0]
        n = x.shape[1]
      
        x_ = x[:, None, :]

        input_dist = np.sum(np.sqrt(np.einsum('ijk, ijk->ij', x-x_, x-x_)))

        input_dirichlet_dist = np.zeros((input_size, ))
        for i in range(self.sample_size):
            z = self.Dir.draw(20000, n)
            sub = x[i] - z
            input_dirichlet_dist[i] = np.mean(np.sqrt(np.einsum('ij,ij->i', sub, sub)))

        input_dirichlet_dist = np.sum(input_dirichlet_dist)

        statistic = (2 * input_dirichlet_dist) - (input_size * self.dir_dist) - ((1/input_size) * input_dist)

        return statistic

    def energy_distance(self, x, dir_dist, d=1):
        return energy_statistic(x, dir_dist, d) / x.shape[0]

    def test_statistic(self, x, n_iter=1000, print_log=False):
        
        random_indices = np.random.rand(n_iter, x.shape[0]).argpartition(self.sample_size, axis=1)[:,:self.sample_size]
        stats = []
        for i in range(n_iter):
            x_sample = x[random_indices[i]] # optimized
            alpha = self.mle_alpha(x_sample) # optimized
            x_sample_transform = self.dirichlet_transform(x_sample, alpha) # optimized

            test_stat = self.energy_statistic(x_sample_transform)
            stats.append(test_stat)
            if print_log:
                print('Finished iteration', str(i))
                print('Test statistic:', test_stat)
                #print('Test distance:', test_stat / x_sample_transform.shape[0])
                print()

        return {'Sample Test Statistics': stats, 'Power': sum(i > self.crit_value for i in stats) / n_iter}

