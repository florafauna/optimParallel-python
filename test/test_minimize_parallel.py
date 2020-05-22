## test minimize_parallel()
import pytest
import itertools
import numpy as np
from scipy.optimize import minimize
from src.optimparallel import minimize_parallel

## test objective functions ---------------------------
## minimize_parallel() expects 'fun' and 'jac' to be global.
fun, jac = None, None

## we reserve the names 'fun' and 'jac' for them.
def fun0(x):
    return sum((x-3)**2)

def jac0(x):
    return 2*(x-3)

def fun_arg1(x, a):
    return sum((x-a)**2)

def jac_arg1(x, a):
    return 2*(x-a)

def fun_arg2(x, a, b):
    return sum((x-a)**2)

def jac_arg2(x, a, b):
    return 2*(x-a)

def fun_upper0(x, ub):
    if not any(x <= ub):
        raise ValueError("x has to be smaller than upper bound")
    return sum((x-1)**2)

def jac_upper0(x, ub):
    if not any(x <= ub):
        raise ValueError("x has to be smaller than upper bound")
    return 2*(x-1)

def fun_lower0(x, ub):
    if not any(x >= ub):
        raise ValueError("x has to be larger than lower bound")
    return sum((x-1)**2)

def jac_lower0(x, ub):
    if not any(x >= ub):
        raise("x has to be larger than lower bound")
    return 2*(x-1)

def compare_minimize(x0, args=(), bounds=None, tol=None,
                     options=None, parallel=None,
                     max_workers=None, forward=True, verbose=False,
                     CHECK_X=True, CHECK_FUN=True, CHECK_JAC=True,
                     CHECK_STATUS=True, TRACEBACKHIDE=True,
                     ATOL=1e-5):
    """Helper function to test minimize_parallel() against minimize()."""
    __tracebackhide__ = TRACEBACKHIDE

    ml = minimize(fun=fun, x0=x0, method="L-BFGS-B", args=args, jac=jac,
                  bounds=bounds, tol=tol, options=options)

    mp = minimize_parallel(fun=fun, x0=x0, args=args, jac=jac, bounds=bounds,
                           tol=tol, options=options, parallel=parallel)

    if CHECK_X and not all(np.isclose(ml.x, mp.x, atol=ATOL)):
        pytest.fail("x different: ml = {}, mp = {}".format(ml.x, mp.x))

    if CHECK_FUN and not np.isclose(ml.fun, mp.fun, atol=ATOL):
        pytest.fail("fun different: ml = {}, mp = {}".format(ml.fun, mp.fun))

    if CHECK_JAC and not all(np.isclose(ml.jac, mp.jac, atol=ATOL)):
        pytest.fail("jac different: ml = {}, mp = {}".format(ml.jac, mp.jac))

    if CHECK_STATUS and not ml.success == mp.success:
        pytest.fail("success different: ml = {}, mp = {}".format(ml.success, mp.success))


def check_minimize(fun_id, x0,
                    args=(),
                    approx_grad=0,
                    bounds=None,
                    disp=None, maxcor=10, ftol=2.220446049250313e-09, gtol=1e-5,
                    eps=1e-8, maxiter=15000, iprint=-1, maxls=20,
                    max_workers=None, forward=True, verbose=False,
                    CHECK_X=True, CHECK_FUN=True, CHECK_JAC=True,
                    CHECK_STATUS=True, TRACEBACKHIDE=True,
                    ATOL=1e-5):
    """Helper function to minimize_parallel() against minimize()."""
    __tracebackhide__ = TRACEBACKHIDE

    ## load parameters of scenario
    global fun, jac
    fun = globals()['fun' + str(fun_id)]
    jac = None if approx_grad else globals()['jac' + str(fun_id)]

    parallel = {'max_workers': max_workers, 'forward': forward, 'verbose': verbose}
    options = {'disp': disp, 'maxcor': maxcor, 'eps': eps, 'ftol': ftol,
               'gtol': gtol, 'maxiter': maxiter, 'iprint': iprint,
               'maxls': maxls}

    compare_minimize(x0=x0, args=args, bounds=bounds,
                     options=options, parallel=parallel,
                     CHECK_X=CHECK_X, CHECK_FUN=CHECK_FUN, CHECK_JAC=CHECK_JAC,
                     CHECK_STATUS=CHECK_STATUS, TRACEBACKHIDE=TRACEBACKHIDE,
                     ATOL=ATOL)



## test options ----------------------------
@pytest.mark.parametrize("fun_id", [0])
@pytest.mark.parametrize("x0", [np.array([1]), np.array([1, 2])])
@pytest.mark.parametrize("approx_grad", [True, False])
@pytest.mark.parametrize("maxcor", [2, 10])
@pytest.mark.parametrize("ftol", [2.220446049250313e-09, 1e-2])
@pytest.mark.parametrize("gtol", [1e-5, 1e-2])
@pytest.mark.parametrize("eps", [1e-8, 1e-2])
@pytest.mark.parametrize("iprint", [-1])
@pytest.mark.parametrize("maxiter", [1, 1500])
@pytest.mark.parametrize("disp", [None])
@pytest.mark.parametrize("maxls", [20,1])
def test_minimize_args0(fun_id, x0, approx_grad, maxcor, ftol, gtol,
                        eps, iprint, maxiter, disp, maxls):
    check_minimize(fun_id=fun_id, x0=x0, approx_grad=approx_grad, maxcor=maxcor,
                   ftol=ftol, gtol=gtol, eps=eps, iprint=iprint,
                   maxiter=maxiter, disp=disp, maxls=maxls)


## test if loginfo=True returns somthing --------------------
@pytest.mark.parametrize("x0", [np.array([-1]), np.array([13, 221])])
def test_minimize_loginfo(x0):
    o = minimize_parallel(fun0, x0=x0, parallel={'loginfo': True})
    assert hasattr(o, 'loginfo')
    assert isinstance(o.loginfo, dict)
    assert isinstance(o.loginfo['x'], np.ndarray)
    assert isinstance(o.loginfo['fun'], np.ndarray)
    assert isinstance(o.loginfo['jac'], np.ndarray)
    nsteps = o.loginfo['x'].shape[0]
    assert o.loginfo['x'].shape == (nsteps, len(x0))
    assert o.loginfo['fun'].shape == (nsteps, 1)
    assert o.loginfo['jac'].shape == (nsteps, len(x0))

## test if time=True returns somthing --------------------
@pytest.mark.parametrize("x0", [np.array([-1]), np.array([13, 221])])
def test_minimize_time(x0):
    o = minimize_parallel(fun0, x0=x0, parallel={'time': True})
    assert hasattr(o, 'time')
    assert isinstance(o.time, dict)
    assert isinstance(o.time['elapsed'], float)
    assert isinstance(o.time['step'], float)

## test functions with 1 extra arg and parallel options --------------------
@pytest.mark.parametrize("fun_id", ["_arg1"])
@pytest.mark.parametrize("x0", [np.array([-1]), np.array([13, 221])])
@pytest.mark.parametrize("args", [(3,), (-35,)])
@pytest.mark.parametrize("approx_grad", [True, False])
@pytest.mark.parametrize("max_workers", [2, None])
@pytest.mark.parametrize("forward", [True, False])
def test_minimize_args1(fun_id, x0, args, approx_grad, max_workers, forward):
    check_minimize(fun_id=fun_id, x0=x0, args=args, approx_grad=approx_grad,
               max_workers=max_workers, forward=forward)


## test functions with 2 extra args and parallel options -----------------
@pytest.mark.parametrize("fun_id", ["_arg2"])
@pytest.mark.parametrize("x0", [np.array([-1]), np.array([13, 221])])
@pytest.mark.parametrize("args", [(773, 66), (-3, 0)])
@pytest.mark.parametrize("approx_grad", [True, False])
@pytest.mark.parametrize("max_workers", [2, None])
@pytest.mark.parametrize("forward", [True, False])
def test_minimize_args2(fun_id, x0, args, approx_grad, max_workers, forward):
    check_minimize(fun_id=fun_id, x0=x0, args=args, approx_grad=approx_grad,
               max_workers=max_workers, forward=forward)


## test bounds upper -------------------------------
@pytest.mark.parametrize("fun_id", ["_upper0"])
@pytest.mark.parametrize("x0", [np.array([-9]), np.array([-9, -99])])
@pytest.mark.parametrize("approx_grad", [True, False])
@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("bu", [np.inf, 5, 1, 0])
def test_minimize_upper(fun_id, x0, approx_grad, forward, bu):
    check_minimize(fun_id=fun_id, x0=x0, args=(bu,), approx_grad=approx_grad,
                   bounds=list(zip(itertools.repeat(-np.inf, len(x0)),
                                   itertools.repeat(bu-1e-8, len(x0)))),
                   forward=forward)


## test bounds lower -------------------------------
@pytest.mark.parametrize("fun_id", ["_lower0"])
@pytest.mark.parametrize("x0", [np.array([9]), np.array([9, 9])])
@pytest.mark.parametrize("approx_grad", [True, False])
@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("bl", [5, 1, 0, -np.inf])
def test_minimize_lower(fun_id, x0, approx_grad, forward, bl):
    check_minimize(fun_id=fun_id, x0=x0, args=(bl,), approx_grad=approx_grad,
                   bounds=list(zip(itertools.repeat(bl+1e-8, len(x0)),
                                   itertools.repeat(np.inf, len(x0)))),
                   forward=forward)


## test options=None and 'tol' ----------------------------------
@pytest.mark.parametrize("fun_id", [0])
@pytest.mark.parametrize("approx_grad", [True, False])
@pytest.mark.parametrize("x0", [np.array([1]), np.array([1,2])])
@pytest.mark.parametrize("tol", [None, 1e-5, 1e-2])
@pytest.mark.parametrize("options", [None, {'gtol': 1e-5}, {'gtol': 1e-2}])
@pytest.mark.filterwarnings("ignore:'tol' is ignored and 'gtol' in 'opitons' is used insetad.")
def test_minimize_tol(fun_id, approx_grad, x0, tol, options):
    ## load parameters of scenario
    global fun, jac
    fun = globals()['fun' + str(fun_id)]
    jac = None if approx_grad else globals()['jac' + str(fun_id)]
    compare_minimize(x0=x0, tol=tol, options=options)
