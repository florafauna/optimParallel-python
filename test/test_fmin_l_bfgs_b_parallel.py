## test fmin_l_bfgs_b_parallel()
import pytest
import itertools
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.optimparallel import fmin_l_bfgs_b_parallel

## test objective functions ---------------------------
## 'func' and 'fprime' cannot be used as function names
func, fprime = None, None

def func0(x):
    return sum((x-3)**2)

def fprime0(x):
    return 2*(x-3)

def func_arg1(x, a):
    return sum((x-a)**2)

def fprime_arg1(x, a):
    return 2*(x-a)

def func_arg2(x, a, b):
    return sum((x-a)**2)

def fprime_arg2(x, a, b):
    return 2*(x-a)

def func_upper0(x, ub):
    if not any(x <= ub):
        raise ValueError("x has to be smaller than upper bound")
    return sum((x-1)**2)

def fprime_upper0(x, ub):
    if not any(x <= ub):
        raise ValueError("x has to be smaller than upper bound")
    return 2*(x-1)

def func_lower0(x, ub):
    if not any(x >= ub):
        raise ValueError("x has to be larger than lower bound")
    return sum((x-1)**2)

def fprime_lower0(x, ub):
    if not any(x >= ub):
        raise ValueError("x has to be larger than lower bound")
    return 2*(x-1)



def check_fmin(func_id, x0,
               args=(),
               approx_grad=0,
               bounds=None,
               m=10, factr=1e7, pgtol=1e-5, epsilon=1e-8, iprint=-1, maxiter=15000,
               disp=None, maxls=20,
               max_workers=None, forward=True, verbose=False,
               CHECK_X=True, CHECK_FUN=True, CHECK_JAC=True,
               CHECK_STATUS=True, TRACEBACKHIDE=True,
               ATOL=1e-5):
    """Helper function to test fmin_l_bfgs_b_parallel() against fmin_l_bfgs_b()."""
    __tracebackhide__ = TRACEBACKHIDE

    ## load parameters of scenario
    global func, fprime
    func = globals()['func' + str(func_id)]
    fprime = None if approx_grad else globals()['fprime' + str(func_id)]
    parallel = {'max_workers': max_workers, 'forward': forward, 'verbose': verbose}

    ml = fmin_l_bfgs_b(func=func, x0=x0, fprime=fprime, args=args,
                       approx_grad=approx_grad, bounds=bounds, m=m, factr=factr,
                       pgtol=pgtol, epsilon=epsilon, iprint=iprint,
                       maxiter=maxiter, disp=disp, callback=None, maxls=maxls)

    mp = fmin_l_bfgs_b_parallel(func=func, x0=x0, fprime=fprime, args=args,
                                approx_grad=approx_grad, bounds=bounds, m=m, factr=factr,
                                pgtol=pgtol, epsilon=epsilon, iprint=iprint,
                                maxiter=maxiter, disp=disp, callback=None, maxls=maxls,
                                parallel=parallel)

    if CHECK_X and not all(np.isclose(ml[0], mp[0], atol=ATOL)):
        pytest.fail("x different: ml = {}, mp = {}".format(ml[0], mp[0]))

    if CHECK_FUN and not np.isclose(ml[1], mp[1], atol=ATOL):
        pytest.fail("fun different: ml = {}, mp = {}".format(ml[1], mp[1]))

    if CHECK_JAC and not all(np.isclose(ml[2].get('grad'), mp[2].get('grad'), atol=ATOL)):
        pytest.fail("jac different: ml = {}, mp = {}".format(ml[2].get('grad'),
                                                             mp[2].get('grad')))

    if CHECK_STATUS and not ml[2].get('warnflag') == mp[2].get('warnflag'):
        pytest.fail("warnflag different: ml = {}, mp = {}".format(ml[2].get('warnflag'),
                                                             mp[2].get('warnflag')))


## test options ----------------------------
@pytest.mark.parametrize("func_id", [0])
@pytest.mark.parametrize("x0", [np.array([1]), np.array([1, 2])])
@pytest.mark.parametrize("approx_grad", [True, False])
@pytest.mark.parametrize("m", [2, 10])
@pytest.mark.parametrize("factr", [1e7, 1e2])
@pytest.mark.parametrize("pgtol", [1e-5, 1e-2])
@pytest.mark.parametrize("epsilon", [1e-8, 1e-2])
@pytest.mark.parametrize("iprint", [-1])
@pytest.mark.parametrize("maxiter", [1, 1500])
@pytest.mark.parametrize("disp", [None])
@pytest.mark.parametrize("maxls", [20,1])
def test_fmin_args0(func_id, x0, approx_grad, m, factr, pgtol,
                    epsilon, iprint, maxiter, disp, maxls):
    check_fmin(func_id=func_id, x0=x0, approx_grad=approx_grad, m=m,
               factr=factr, pgtol=pgtol, epsilon=epsilon, iprint=iprint,
               maxiter=maxiter, disp=disp, maxls=maxls)


## test functions with 1 extra arg and parallel options --------------------
@pytest.mark.parametrize("func_id", ["_arg1"])
@pytest.mark.parametrize("x0", [np.array([-1]), np.array([13, 221])])
@pytest.mark.parametrize("args", [(3,), (-35,)])
@pytest.mark.parametrize("approx_grad", [True, False])
@pytest.mark.parametrize("max_workers", [2, None])
@pytest.mark.parametrize("forward", [True, False])
def test_fmin_args1(func_id, x0, args, approx_grad, max_workers, forward):
    check_fmin(func_id=func_id, x0=x0, args=args, approx_grad=approx_grad,
               max_workers=max_workers, forward=forward)


## test functions with 2 extra args and parallel options -----------------
@pytest.mark.parametrize("func_id", ["_arg2"])
@pytest.mark.parametrize("x0", [np.array([-1]), np.array([13, 221])])
@pytest.mark.parametrize("args", [(773, 66), (-3, 0)])
@pytest.mark.parametrize("approx_grad", [True, False])
@pytest.mark.parametrize("max_workers", [2, None])
@pytest.mark.parametrize("forward", [True, False])
def test_fmin_args2(func_id, x0, args, approx_grad, max_workers, forward):
    check_fmin(func_id=func_id, x0=x0, args=args, approx_grad=approx_grad,
               max_workers=max_workers, forward=forward)


## test bounds upper -------------------------------
@pytest.mark.parametrize("func_id", ["_upper0"])
@pytest.mark.parametrize("x0", [np.array([-9]), np.array([-9, -99])])
@pytest.mark.parametrize("approx_grad", [True, False])
@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("bu", [np.inf, 5, 1, 0])
def test_fmin_upper(func_id, x0, approx_grad, forward, bu):
    check_fmin(func_id=func_id, x0=x0, args=(bu,), approx_grad=approx_grad,
               bounds=list(zip(itertools.repeat(-np.inf, len(x0)),
                               itertools.repeat(bu-1e-8, len(x0)))),
               forward=forward)


## test bounds lower -------------------------------
@pytest.mark.parametrize("func_id", ["_lower0"])
@pytest.mark.parametrize("x0", [np.array([9]), np.array([9, 9])])
@pytest.mark.parametrize("approx_grad", [True, False])
@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("bl", [5, 1, 0, -np.inf])
def test_fmin_lower(func_id, x0, approx_grad, forward, bl):
    check_fmin(func_id=func_id, x0=x0, args=(bl,), approx_grad=approx_grad,
               bounds=list(zip(itertools.repeat(bl+1e-8, len(x0)),
                               itertools.repeat(np.inf, len(x0)))),
               forward=forward)

