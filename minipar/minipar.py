"""
A parallel version of the L-BFGS-B optimizer of scipy.optimize.minimize.

Function
--------
- minimize_parallel : minimization of a function of several variables
                      unsing the L-BFGS-B algorithm. All caluclatons
                      for one step (evaluation of 'fun' and 'jac') are
                      executed parallel.

Developed with Python 3.7.4
"""


import warnings
import concurrent.futures
import functools
import time
import itertools
import numpy as np
from scipy.optimize import minimize

__all__ = ['minimize_parallel']

class EvalParallel:
    def __init__(self, fun, jac=None, args=(),
                 eps=1e-8, forward=True, verbose=False, n=1):
        self.fun_in = fun
        self.jac_in = jac
        self.eps = eps
        self.forward = forward
        self.verbose = verbose
        self.x_val = None
        self.fun_val = None
        self.jac_val = None
        if not (isinstance(args, list) or isinstance(args, tuple)):
            self.args = (args,)
        else:
            self.args = tuple(args)
        self.n = n
            
    @staticmethod
    def _eval_approx_args(args, eps_at, fun, x, eps):
        ## helper function for parallel execution with map()
        ## for the case where 'fun' has additionals 'args' 
        if eps_at == 0:
            x_ = x
        elif eps_at <= len(x):
            x_ = x.copy()
            x_[eps_at-1] += eps
        else:
            x_ = x.copy()
            x_[eps_at-1-len(x)] -= eps
        return fun(x_, *args)
    
    @staticmethod
    def _eval_approx(eps_at, fun, x, eps):
        ## helper function for parallel execution with map()
        ## for the case where 'fun' has no additionals 'args' 
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
    def _eval_fun_jac_args(args, which, fun, jac, x):
        ## helper function for parallel execution with map()
        ## for the case where 'fun' has additionals 'args' 
        if which == 0:
            return fun(x, *args)
        return np.array(jac(x, *args))

    @staticmethod
    def _eval_fun_jac(which, fun, jac, x):
        ## helper function for parallel execution with map()
        ## for the case where 'fun' has no additionals 'args' 
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
                
            ## package 'self.args' into function because it cannot be
            ## serialized by 'concurrent.futures.ProcessPoolExecutor()'
            if len(self.args) > 0:
                ftmp = functools.partial(self._eval_approx_args, self.args)
            else:
                ftmp = self._eval_approx

            with concurrent.futures.ProcessPoolExecutor() as executor:
                ret = executor.map(ftmp, eps_at,
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
            self.jac_val = self.jac_val.reshape((self.n,))
            return None

        if len(self.args) > 0:
            ftmp = functools.partial(self._eval_fun_jac_args, self.args)
        else:
            ftmp = self._eval_fun_jac
            
        with concurrent.futures.ProcessPoolExecutor() as executor:
            ret = executor.map(ftmp, [0,1],
                               itertools.repeat(self.fun_in),
                               itertools.repeat(self.jac_in),
                               itertools.repeat(x))
            ret = list(ret)
        self.fun_val = ret[0]
        self.jac_val = ret[1]
        self.jac_val = self.jac_val.reshape((self.n,))
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
                      args=(),
                      jac=None,
                      bounds=None,
                      tol=None,
                      options=None,
                      callback=None,
                      parallel={'forward':True, 'verbose':False}):

    """
    A parallel version of the L-BFGS-B optimizer of `func:scipy.optimize.minimize()`.
    
    Parameters
    ----------

    All the same as in `func:scipy.optimize.minimize()` except for `method`
    which is set to `'L-BFGS-B'` and thet additional argument
    parallel: ... 
    """
    
    ## get length of x0
    try:
        n = len(x0)
    except:
        n = 1
        
    ## update default options with specified options
    options_used = {'disp': None, 'maxcor': 10,
                    'ftol': 2.220446049250313e-09, 'gtol': 1e-05,
                    'eps': 1e-08, 'maxfun': 15000,
                    'maxiter': 15000, 'iprint': -1, 'maxls': 20}
    if not options is None: 
        assert isinstance(options, dict), "argument 'options' must be of type 'dict'"
        options_used.update(options)
    if not tol is None:
        if not options is None and 'gtol' in options:
            warnings.warn("'tol' is ignored and 'gtol' in 'opitons' is used insetad.",
                          RuntimeWarning)
        else:
            options_used['gtol'] = tol

    parallel_used = {'forward': True, 'verbose': False}
    if not parallel is None: 
        assert isinstance(parallel, dict), "argument 'parallel' must be of type 'dict'"
        parallel_used.update(parallel)


            
    funJac = EvalParallel(fun=fun,
                          jac=jac,
                          args=args,
                          eps=options_used.get('eps'),
                          forward=parallel_used.get('forward'),
                          verbose=parallel_used.get('verbose'),
                          n=n)
    out = minimize(fun=funJac.fun,
                   x0=x0,
                   jac=funJac.jac,
                   method='L-BFGS-B',
                   bounds=bounds,
                   callback=callback,
                   options=options_used)
    return out

if __name__ == '__main__':
    ## a simple example
    def f(x, a, b):
        print('fn')
        time.sleep(.2)
        return sum((x-a)**2)
    
    def g(x, a, b):
        print('gr')
        return 2*(x-a)

    print(f(np.array([1,2]), a=1, b=2))
    
    o1 = minimize(fun=f, x0=np.array([10,20]), jac=g, args=(77,44),
                  method='L-BFGS-B',
                  options={'disp':False, 'maxls':2000})
    print('\n', o1)
    
    o2 = minimize_parallel(fun=f, x0=np.array([10,20]), jac=g, args=(77,44),
                           options={'disp':False, 'maxls':2000})
    print('\n', o2)

    
    all(np.isclose(o1.jac, o2.jac, atol=1e-5))

    def f(x, a, b):
        assert any(x >= b), f'x >= {b:.3f} does not hold.'
        print('fn')
        time.sleep(.2)
        return sum((x-a)**2)
    
    o1 = minimize(fun=f, x0=np.array([10,20]), args=(0, 1),
                  method='L-BFGS-B',
                  bounds=Bounds(lb=np.array([1,1]), ub=[np.inf,np.inf]))

    o2 = minimize_parallel(fun=f, x0=np.array([10,20]), args=(0,1),
                           bounds=Bounds(lb=np.array([1,1]), ub=[np.inf,np.inf]))
    print(o1)


    
    def fun_2args0(x, a, b):
        return sum((x-a)**2) + b
    def jac_2args0(x, a, b):
        return 2*(x-a)

    # args = (1, 2)
    # x0 = np.array([1, 2])
    # eps = 0.01
    # forward = False
    # jac_flag = False
    # fn_id = 0

    def test_minimize_parallel_2args(args, x0, eps, forward, jac_flag, fn_id):

        ## concurrent.futures.ProcessPoolExecutor() requires fun and jac to be globals
        global fun, jac
        
        ## load parameters of scenario
        fun=globals()['fun_2args' + str(fn_id)]
        if not jac_flag:
            jac=None
        else:
            jac=globals()['jac_2args' + str(fn_id)]
        options={'eps': eps}
        parallel={'forward': forward, 'verbose': False}
        
        
        mp=minimize_parallel(fun=fun, x0=x0,
                             args = args,
                             jac=jac,
                             # bounds,
                             # tol,
                             options=options,
                        # callback,
                             parallel=parallel)
        m=minimize(fun=fun, x0=x0, method="L-BFGS-B",
                   args=args,
                   jac=jac,
                   # bounds,
                   # tol,
                   options=options
                   # callback
        )
        
        
        assert np.isclose(mp.fun, m.fun)
        assert all(np.isclose(mp.jac, m.jac, atol=1e-5))
        assert mp.success == m.success
        assert all(np.isclose(mp.x, m.x))
        return None
    
            
    test_minimize_parallel_2args(args = (1,2),
                                 x0 = np.array([1,2]),
                                 eps = 0.01,
                                 forward = True,
                                 jac_flag = False,
                                 fn_id=0)

 # --> why same numerical results and different 'success' value????
 # >>> mp
 #      fun: 2.0000247514030973
 # hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
 #      jac: array([4.98743851e-05, 1.00247514e-02])
 #  message: b'ABNORMAL_TERMINATION_IN_LNSRCH'
 #     nfev: 42
 #      nit: 1
 #   status: 2
 #  success: False
 #        x: array([0.99502494, 1.00001238])
 # >>> m
 #      fun: 2.000024751403096
 # hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
 #      jac: array([4.98743851e-05, 1.00247514e-02])
 #  message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
 #     nfev: 63
 #      nit: 2
 #   status: 0
 #  success: True
 #        x: array([0.99502494, 1.00001238])




