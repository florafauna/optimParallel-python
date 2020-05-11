## tests for minimize_parallel()

import pytest
import numpy as np
from scipy.optimize import Bounds, minimize
from minipar.minipar import *

## test functions without additional arguments ---------------------------
def fun0(x):
    return sum((x-3)**2)
def jac0(x):
    return 2*(x-3)

def fun1(x):
    return sum((x-4)**2)
def jac1(x):
    return 2*(x-4)

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
def test_minimize_parallel(fn_id, jac_flag, forward, x0, disp, maxcor, ftol, gtol,
                           eps, maxiter, iprint, maxls):

    ## concurrent.futures.ProcessPoolExecutor() requires fun and jac to be globals
    global fun, jac

    ## load parameters of scenario
    fun = globals()['fun' + str(fn_id)]
    jac = globals()['jac' + str(fn_id)] if jac_flag else None
    options = {'disp': disp, 'maxcor': maxcor, 'eps': eps, 'ftol': ftol,
               'gtol': gtol, 'maxiter': maxiter, 'iprint': iprint,
               'maxls': maxls}
    parallel = {'forward': forward, 'verbose': False}

    
    mp = minimize_parallel(fun=fun, x0=x0,
                           # args,
                           jac=jac,
                           # bounds,
                           # tol,
                           options=options,
                           # callback,
                           parallel=parallel)

    m = minimize(fun=fun, x0=x0, method="L-BFGS-B",
                 # args,
                 jac=jac,
                 # bounds,
                 # tol,
                 options=options
                 # callback
    ) 

    ## test that minimimze_parallel() close to minimze()
    assert np.isclose(mp.fun, m.fun, atol=1e-5)
    assert all(np.isclose(mp.jac, m.jac, atol=1e-5))
    # assert mp.success == m.success
    assert all(np.isclose(mp.x, m.x))
   
    return None


## test 'parallel' argument: 'max_workers', 'forward', 'verbose'  
@pytest.mark.parametrize("fn_id", [0])    
@pytest.mark.parametrize("jac_flag", [True, False])
@pytest.mark.parametrize("x0", [np.array([1]), np.array([1,2])])    
@pytest.mark.parametrize("max_workers", [None, 1, 2])
@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
def test_minimize_parallel_parallel_arg(fn_id, jac_flag, x0, max_workers, forward, verbose):

    ## concurrent.futures.ProcessPoolExecutor() requires fun and jac to be globals
    global fun, jac

    ## load parameters of scenario
    fun = globals()['fun' + str(fn_id)]
    jac = globals()['jac' + str(fn_id)] if jac_flag else None
    parallel = {'max_workers': max_workers, 'forward': forward, 'verbose': verbose}
    
    mp = minimize_parallel(fun=fun, x0=x0,
                           jac=jac,
                           parallel=parallel)
    m = minimize(fun=fun, x0=x0, method="L-BFGS-B",
                 jac=jac) 
    
    assert np.isclose(mp.fun, m.fun, atol=1e-5)
    assert all(np.isclose(mp.jac, m.jac, atol=1e-5))
    #    assert mp.success == m.success
    assert all(np.isclose(mp.x, m.x))
    return None


## test options=None and 'tol'
@pytest.mark.parametrize("fn_id", [0])    
@pytest.mark.parametrize("jac_flag", [True, False])
@pytest.mark.parametrize("forward", [True])
@pytest.mark.parametrize("x0", [np.array([1]), np.array([1,2])])    
@pytest.mark.parametrize("tol", [None, 1e-5, 1e-2])
@pytest.mark.parametrize("options", [None, {'gtol': 1e-5}, {'gtol': 1e-2}])
@pytest.mark.filterwarnings("ignore:'tol' is ignored and 'gtol' in 'opitons' is used insetad.")
def test_minimize_parallel_tol(fn_id, jac_flag, forward, x0, tol, options):

    ## concurrent.futures.ProcessPoolExecutor() requires fun and jac to be globals
    global fun, jac

    ## load parameters of scenario
    fun = globals()['fun' + str(fn_id)]
    jac = globals()['jac' + str(fn_id)] if jac_flag else None
    parallel = {'forward': forward, 'verbose': False}
    
    mp = minimize_parallel(fun=fun, x0=x0,
                           # args,
                           jac=jac,
                           # bounds,
                           tol=tol,
                           options=options,
                           # callback,
                           parallel=parallel)
    m = minimize(fun=fun, x0=x0, method="L-BFGS-B",
                 # args,
                 jac=jac,
                 # bounds,
                 # tol,
                 options=options
                 # callback
    )
    
    assert np.isclose(mp.fun, m.fun, atol=1e-5)
    assert all(np.isclose(mp.jac, m.jac, atol=1e-5))
#    assert mp.success == m.success
    assert all(np.isclose(mp.x, m.x))
    return None



## test functions with one additional argument ---------------------------
def fun_1args0(x, a):
    return sum((x-a)**2)
def jac_1args0(x, a):
    return 2*(x-a)

@pytest.mark.parametrize("args", [2, (3,), [3], np.array([5])])    
@pytest.mark.parametrize("x0", [np.array([1]), np.array([1,2]), np.array([1,2,3])])    
@pytest.mark.parametrize("eps", [1e-8, 1e-2])    
@pytest.mark.parametrize("forward", [True])    
@pytest.mark.parametrize("jac_flag", [True, False])    
@pytest.mark.parametrize("fn_id", [0])    
def test_minimize_parallel_1args(args, x0, eps, forward, jac_flag, fn_id):

    ## concurrent.futures.ProcessPoolExecutor() requires fun and jac to be globals
    global fun, jac

    ## load parameters of scenario
    fun = globals()['fun_1args' + str(fn_id)]
    jac = globals()['jac_1args' + str(fn_id)] if jac_flag else None
    options = {'eps': eps}
    parallel = {'forward': forward, 'verbose': False}

    mp = minimize_parallel(fun=fun, x0=x0, args=args, jac=jac,
                           options=options,
                           parallel=parallel)
    m = minimize(fun=fun, x0=x0, method="L-BFGS-B", args=args, jac=jac,
                 options=options)

    
    assert np.isclose(mp.fun, m.fun)
    assert all(np.isclose(mp.jac, m.jac, atol=1e-5))
    assert mp.success == m.success
    assert all(np.isclose(mp.x, m.x))

    return None



## test functions with two additional argument ---------------------------
def fun_2args0(x, a, b):
    return sum((x-a)**2) + b
def jac_2args0(x, a, b):
    return 2*(x-a)

@pytest.mark.parametrize("args", [(1,2), (3,4), (np.array([5]), np.array([6]))])    
@pytest.mark.parametrize("x0", [np.array([1]), np.array([1,2]), np.array([2,3])])    
@pytest.mark.parametrize("eps", [1e-8, 1e-2])    
@pytest.mark.parametrize("forward", [True])    
@pytest.mark.parametrize("jac_flag", [True, False])    
@pytest.mark.parametrize("fn_id", [0])    
def test_minimize_parallel_2args(args, x0, eps, forward, jac_flag, fn_id):

    ## concurrent.futures.ProcessPoolExecutor() requires fun and jac to be globals
    global fun, jac

    ## load parameters of scenario
    fun = globals()['fun_2args' + str(fn_id)]
    jac = globals()['jac_2args' + str(fn_id)] if jac_flag else None
    options={'eps': eps}
    parallel={'forward': forward, 'verbose': False}
    
    mp = minimize_parallel(fun=fun, x0=x0,
                           args = args,
                           jac=jac,
                           # bounds,
                           # tol,
                           options=options,
                           # callback,
                           parallel=parallel)
    m = minimize(fun=fun, x0=x0, method="L-BFGS-B",
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


## test lower boundaries ----------------------------------------
def fun_lower0(x, lb):
    assert any(x >= lb), "x has to bigger than lower bound" 
    return sum((x-1)**2)
   
def jac_lower0(x, lb):
    assert any(x >= lb), "x has to bigger than lower bound" 
    return 2*(x-1)

@pytest.mark.parametrize("fn_id", [0])    
@pytest.mark.parametrize("jac_flag", [True, False])
@pytest.mark.parametrize("forward", [True])
@pytest.mark.parametrize("x0", [np.array([10]), np.array([10,10])])    
@pytest.mark.parametrize("lb", [5,1,0])    
def test_minimize_parallel_bound_lower(fn_id, jac_flag, forward, x0, lb):

    ## concurrent.futures.ProcessPoolExecutor() requires fun and jac to be globals
    global fun, jac

    ## load parameters of scenario
    fun = globals()['fun_lower' + str(fn_id)]
    jac = globals()['jac_lower' + str(fn_id)] if jac_flag else None
    parallel = {'forward': forward, 'verbose': False}
    bounds = Bounds(lb=np.repeat(lb, len(x0)), ub=np.repeat(np.inf, len(x0)))
    
    mp = minimize_parallel(fun=fun, x0=x0, args = lb,
                           jac=jac, 
                           bounds=bounds,
                           parallel=parallel)
    m = minimize(fun=fun, x0=x0, args = lb,
                 method="L-BFGS-B", 
                 jac=jac,
                 bounds=bounds)
   
    assert np.isclose(mp.fun, m.fun, atol=1e-5)
    assert all(np.isclose(mp.jac, m.jac, atol=1e-5))
#    assert mp.success == m.success
    assert all(np.isclose(mp.x, m.x))
    return None


## test upper boundaries -------------------
def fun_upper0(x, ub):
    assert any(x <= ub), "x has to be smaller than upper bound" 
    return sum((x-1)**2)
   
def jac_upper0(x, ub):
    assert any(x <= ub), "x has to be smaller than upper bound" 
    return 2*(x-1)

@pytest.mark.parametrize("fn_id", [0])    
@pytest.mark.parametrize("jac_flag", [True, False])
@pytest.mark.parametrize("forward", [True])
@pytest.mark.parametrize("x0", [np.array([-10]), np.array([-10,-10])])    
@pytest.mark.parametrize("ub", [5,1,0])    
def test_minimize_parallel_bound_upper(fn_id, jac_flag, forward, x0, ub):

    ## concurrent.futures.ProcessPoolExecutor() requires fun and jac to be globals
    global fun, jac

    ## load parameters of scenario
    fun = globals()['fun_upper' + str(fn_id)]
    jac = globals()['jac_upper' + str(fn_id)] if jac_flag else None
    parallel = {'forward': forward, 'verbose': False}
    bounds = Bounds(lb=np.repeat(np.inf, len(x0)), ub=np.repeat(ub, len(x0)))
    
    mp = minimize_parallel(fun=fun, x0=x0, args = ub,
                           jac=jac, 
                           bounds=bounds,
                           parallel=parallel)
    m = minimize(fun=fun, x0=x0, args = ub,
                 method="L-BFGS-B", 
                 jac=jac,
                 bounds=bounds)
   
    assert np.isclose(mp.fun, m.fun, atol=1e-5)
    assert all(np.isclose(mp.jac, m.jac, atol=1e-5))
#    assert mp.success == m.success
    assert all(np.isclose(mp.x, m.x))
    return None




