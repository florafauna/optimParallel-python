## test minimize_parallel()
import pytest
import itertools
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from src.optimparallel import minimize_parallel

## test objective functions ---------------------------
## minimize_parallel() expects 'fun' and 'jac' to be global.
fun, jac = None, None

## we reserve the names 'fun' and 'jac' for them.
def fun0(x):
    return sum((x-3)**2)

def jac0(x):
    return 2*(x-3)

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

    with ProcessPoolExecutor() as p:
        mp = minimize_parallel(p, fun=fun, x0=x0, args=args, jac=jac, bounds=bounds,
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
@pytest.mark.parametrize("x0", [np.array([1])])
@pytest.mark.parametrize("approx_grad", [False])
@pytest.mark.parametrize("maxcor", [2])
@pytest.mark.parametrize("ftol", [2.220446049250313e-09])
@pytest.mark.parametrize("gtol", [1e-5])
@pytest.mark.parametrize("eps", [1e-8])
@pytest.mark.parametrize("iprint", [-1])
@pytest.mark.parametrize("maxiter", [1500])
@pytest.mark.parametrize("disp", [None])
@pytest.mark.parametrize("maxls", [20])
def test_minimize_args0(fun_id, x0, approx_grad, maxcor, ftol, gtol,
                        eps, iprint, maxiter, disp, maxls):
    check_minimize(fun_id=fun_id, x0=x0, approx_grad=approx_grad, maxcor=maxcor,
                   ftol=ftol, gtol=gtol, eps=eps, iprint=iprint,
                   maxiter=maxiter, disp=disp, maxls=maxls)

