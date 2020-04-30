import pytest
from minipar.minipar import *

## 
def fun0(x):
    return sum((x-3)**2)
def jac0(x):
    return 2*(x-3)

def fun1(x):
    return sum((x-4)**2)
def jac1(x):
    return 2*(x-4)


@pytest.mark.parametrize("x0", [np.array([1]), np.array([1,2]), np.array([1,2,3])])    
@pytest.mark.parametrize("eps", [1e-8, 1e-4, 1e-2])    
@pytest.mark.parametrize("forward", [True, False])    
@pytest.mark.parametrize("jac_flag", [True, False])    
@pytest.mark.parametrize("fn_id", [key.replace('fun','') for key, value
                                   in globals().items() if 'fun' in key.lower()])    
def test_minimize_parallel(x0, eps, forward, jac_flag, fn_id):

    ## concurrent.futures.ProcessPoolExecutor() requires fun and jac to be globals
    global fun, jac

    ## load parameters of scenario
    fun=globals()['fun' + str(fn_id)]
    if not jac_flag:
        jac=None
    else:
        jac=globals()['jac' + str(fn_id)]
    options={'eps': eps}
    parallel={'forward': True, 'verbose': False}

    
    mp=minimize_parallel(fun=fun, x0=x0,
                         # args,
                         jac=jac,
                         # bounds,
                         # tol,
                         options=options,
                         # callback,
                         parallel=parallel)
    m=minimize(fun=fun, x0=x0, method="L-BFGS-B",
               # args,
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
    
