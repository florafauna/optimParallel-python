## developed with Python 3.7.4
import concurrent.futures
import time
import itertools
import numpy as np
from scipy.optimize import minimize

class EvalParallel:
    def __init__(self, fun, jac=None, eps=1e-8, forward=True, verbose=False):
        self.fun_in = fun
        self.jac_in = jac
        self.eps = eps
        self.forward = forward
        self.verbose = verbose
        self.x_val = None
        self.fun_val = None
        self.jac_val = None

    @staticmethod
    def _eval_approx(eps_at, fun, x, eps):
        ## helper function for parallel execution with map()
        if eps_at == 0:
            x_ = x
        elif eps_at <= len(x):
            x_ = x.copy()
            x_[eps_at-1] += eps
        else:
            x_ = x.copy()
            x_[eps_at-1-len(x)] -= eps
        return fun(x_)
    
    @staticmethod
    def _eval_fun_jac(which, fun, jac, x):
        ## helper function for parallel execution with map()
        if which == 0:
            return fun(x)
        return np.array(jac(x))

    def eval_parallel(self, x):
        ## function to evaluate the function fun and jac in parallel
        ## - if jac is None, the gradient is computed numerically
        ## - if forward is True, the numerical gradient used
        ##                      the forward difference method,
        ##   otherwise, the central difference method is used
        x = np.array(x)
        if (self.x_val is not None and max(abs(self.x_val - x)) < 1e-10):
            if self.verbose:
                print('re-use')
            return None
        self.x_val = x.copy()
        if self.jac_in is None:
            if self.forward:
                eps_at = range(len(x)+1)
            else:
                eps_at = range(2*len(x)+1)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                ret = executor.map(self._eval_approx, eps_at,
                                   itertools.repeat(self.fun_in),
                                   itertools.repeat(x),
                                   itertools.repeat(self.eps))
            ret = np.array(list(ret))
            self.fun_val = ret[0]
            if self.forward:
                self.jac_val = (ret[1:(len(x)+1)] - self.fun_val ) / self.eps
            else:
                self.jac_val = (ret[1:(len(x)+1)]
                                - ret[(len(x)+1):2*len(x)+1]) / (2*self.eps)
            return None
                    
        with concurrent.futures.ProcessPoolExecutor() as executor:
            ret = executor.map(self._eval_fun_jac, [0,1],
                               itertools.repeat(self.fun_in),
                               itertools.repeat(self.jac_in),
                               itertools.repeat(x))
            ret = list(ret)
            self.fun_val = ret[0]
            self.jac_val = ret[1]
        return None
                    
    def fun(self, x):
        self.eval_parallel(x=x)
        if self.verbose:
            print('fun(' + str(x) + ') = ' + str(self.fun_val))
        return self.fun_val
    
    def jac(self, x):
        self.eval_parallel(x=x)
        if self.verbose:
            print('jac(' + str(x)+ ') = ' + str(self.jac_val))
        return self.jac_val


def minimize_parallel(fun, x0,
                      # args,
                      jac=None,
                      # bounds,
                      # tol,
                      options={#'disp': None,
                          #'maxcor': 10,
                          #'ftol': 2.220446049250313e-09,
                          #'gtol': 1e-05,
                          'eps': 1e-08,
                          #'maxfun': 15000,
                          #'maxiter': 15000,
                          #'iprint': -1,
                          #'maxls': 20
                      },
                      # callback,
                      parallel={'forward':True, 'verbose':False}):
    
    funJac = EvalParallel(fun=fun,
                          jac=jac,
                          eps=options.get('eps'),
                          forward=parallel.get('forward'),
                          verbose=parallel.get('verbose'))

    ret = minimize(fun=funJac.fun,
                   x0=x0,
                   jac=funJac.jac,
                   method='L-BFGS-B')
    return ret



if __name__ == '__main__':
    ## a simple example
    def f(x):
        print('.', end='')
        time.sleep(1)
        out = sum((x-3)**2)
        return out
    
    print(f(np.array([1,2])))
    
    o1 = minimize(fun=f, x0=np.array([10,20]), method='L-BFGS-B')
    print(o1)
    
    o2 = minimize_parallel(fun=f, x0=np.array([10,20]))
    print(o2)
