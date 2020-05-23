"""
Example of `minimize_parallel()` on a Windows OS.

On a Windows OS it might be necessary to run `minimize_parallel()`
in the main scope. Here is an example.
"""

from optimparallel import minimize_parallel
from scipy.optimize import minimize
import numpy as np
import time

## objective function, has to be global
def f(x, sleep_secs=.5):
    print('fn')
    time.sleep(sleep_secs)
    return sum((x-14)**2)


def main():
    """Function to be called in the main scope."""
    ## start value
    x0 = np.array([10,20])

    ## minimize with parallel evaluation of 'fun'
    ## and its approximate gradient
    o1 = minimize_parallel(fun=f, x0=x0, args=.5, parallel={'loginfo':True})
    print(o1)

    ## test against scipy.optimize.minimize(method='L-BFGS-B')
    o2 = minimize(fun=f, x0=x0, args=.5, method='L-BFGS-B')
    print(o2)
    print(all(np.isclose(o1.x, o2.x, atol=1e-10)),
	  np.isclose(o1.fun, o2.fun, atol=1e-10),
	  all(np.isclose(o1.jac, o2.jac, atol=1e-10)))

    ## timing results
    o1_start = time.time()
    _ = minimize_parallel(fun=f, x0=x0, args=.5)
    o1_end = time.time()
    o2_start = time.time()
    _ = minimize(fun=f, x0=x0, args=.5, method='L-BFGS-B')
    o2_end = time.time()
    print("Time parallel {:2.2}\nTime standard {:2.2} ".
          format(o1_end - o1_start, o2_end - o2_start))


if __name__ == '__main__':
    main()
