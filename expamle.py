""" Example of `minimize_parallel()` """
from minipar.minipar import minimize_parallel
from scipy.optimize import minimize
import numpy as np
import time
from timeit import default_timer as timer

def f(x, sleep_secs=.5):
    print('fn')
    time.sleep(sleep_secs)
    return sum((x-14)**2)

o1 = minimize_parallel(fun=f, x0=np.array([10,20]), args=(.5))
print(o1)

## test against scipy.optimize.minimize()
o2 = minimize(fun=f, x0=np.array([10,20]), args=(.5))
all(np.isclose(o1.x, o2.x, atol=1e-5))
np.isclose(o1.fun, o2.fun, atol=1e-5)
all(np.isclose(o1.jac, o2.jac, atol=1e-5))

## timing results
o1_start = timer()
o1 = minimize_parallel(fun=f, x0=np.array([10,20]), args=(.5))
o1_end = timer()
o1_time = o1_end - o1_start
o2_start = timer()
o2 = minimize(fun=f, x0=np.array([10,20]), args=(.5))
o2_end = timer()
o2_time = o2_end - o2_start

print("Time parallel {:2.2}\nTime standard {:2.2} ".
      format(o1_time, o2_time))

## example with gradient 
def g(x, sleep_secs=.5):
    print('gr')
    time.sleep(sleep_secs)
    return 2*(x-14)

o3 = minimize_parallel(fun=f, x0=np.array([10,20]), jac=g, args=(.5))
o4 = minimize(fun=f, x0=np.array([10,20]), jac=g, args=(.5))

all(np.isclose(o3.x, o4.x, atol=1e-5))
np.isclose(o3.fun, o4.fun, atol=1e-5)
all(np.isclose(o3.jac, o4.jac, atol=1e-5))
