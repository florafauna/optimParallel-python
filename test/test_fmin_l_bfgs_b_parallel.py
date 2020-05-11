## test test_fmin_l_bfgs_b_parallel

import pytest
import numpy as np
from scipy.optimize import Bounds, fmin_l_bfgs_b
from minipar.minipar import *

## test functions without additional arguments ---------------------------
def fun0(x):
    return sum((x-3)**2)
def jac0(x):
    return 2*(x-3)

@pytest.mark.parametrize("fn_id", [0])    
@pytest.mark.parametrize("jac_flag", [True, False])
@pytest.mark.parametrize("forward", [True])
@pytest.mark.parametrize("x0", [np.array([1]), np.array([1,2])])    
@pytest.mark.parametrize("disp", [None,True])
@pytest.mark.parametrize("maxcor", [2, 10])
@pytest.mark.parametrize("ftol", [2.220446049250313e-09, 2e-2])
@pytest.mark.parametrize("gtol", [1e-5, 1e-2])
@pytest.mark.parametrize("eps", [1e-8, 1e-2])
## it does not make sense to test for 'maxfun'
@pytest.mark.parametrize("maxiter", [15000, 2])
@pytest.mark.parametrize("iprint", [-1,100])
@pytest.mark.parametrize("maxls", [20,1])
def test_fmin_l_bfgs_b_parallel(fn_id, jac_flag, forward, x0, disp, maxcor,
                                ftol, gtol, eps, maxiter, iprint, maxls):

    ## concurrent.futures.ProcessPoolExecutor() requires fun and jac to be globals
    global fun, jac

    ## load parameters of scenario
    fun = globals()['fun' + str(fn_id)]
    jac = globals()['jac' + str(fn_id)] if jac_flag else None
    parallel = {'forward': forward, 'verbose': False}

    
    mp = fmin_l_bfgs_b_parallel(func=fun, x0=x0, fprime=jac, args=(), approx_grad=not jac_flag,
                                bounds=None, m=maxcor, factr=ftol, pgtol=gtol,
                                epsilon=eps, iprint=iprint, 
                                maxiter=maxiter, disp=disp, callback=None, maxls=maxls,
                                parallel=parallel)
    m = fmin_l_bfgs_b(func=fun, x0=x0, fprime=jac, args=(), approx_grad=not jac_flag,
                      bounds=None, m=maxcor, factr=ftol, pgtol=gtol,
                      epsilon=eps, iprint=iprint, 
                      maxiter=maxiter, disp=disp, callback=None, maxls=maxls)
    
    ## test that fmin_l_bfgs_b() close to fmin_l_bfgs_b_parallel()
    assert all(np.isclose(mp[0], m[0], atol=1e-5))
    assert np.isclose(mp[1], m[1], atol=1e-5)
    return None
