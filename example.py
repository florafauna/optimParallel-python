""" Example of `minimize_parallel()` """
from minipar.minipar import minimize_parallel
from scipy.optimize import minimize
import numpy as np
import time
from timeit import default_timer as timer
import matplotlib.pyplot as plt

## objective function
def f(x, sleep_secs=.5):
    print('fn')
    time.sleep(sleep_secs)
    return sum((x-14)**2)

## start value
x0 = np.array([10,20])

## minimze with parallel evaluation of 'fun' and
## its approximate gradient. 
o1 = minimize_parallel(fun=f, x0=x0, args=.5)
print(o1)

## test against scipy.optimize.minimize()
o2 = minimize(fun=f, x0=x0, args=.5)
print(all(np.isclose(o1.x, o2.x, atol=1e-5)),
      np.isclose(o1.fun, o2.fun, atol=1e-5),
      all(np.isclose(o1.jac, o2.jac, atol=1e-5)))

## timing results
o1_start = timer()
_ = minimize_parallel(fun=f, x0=x0, args=.5)
o1_end = timer()
o2_start = timer()
_ = minimize(fun=f, x0=x0, args=.5)
o2_end = timer()
print("Time parallel {:2.2}\nTime standard {:2.2} ".
      format(o1_end - o1_start, o2_end - o2_start))

## loginfo -------------------------------------
o1 = minimize_parallel(fun=f, x0=x0, args=.5, parallel={'loginfo': True})
o1.loginfo['x']
o1.loginfo['fun']
o1.loginfo['jac']

x1, x2 = o1.loginfo['x'][:,0], o1.loginfo['x'][:,1]
plt.plot(x1, x2, '-o')
for i in range(len(x1)):
    plt.text(x1[i]+.2, x2[i], 'f = {a:3.3f}'.format(a=o1.loginfo['fun'][i,0]))
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.xlim(right=x1[-1]+1)
plt.show()

## example with gradient -----------------------
def g(x, sleep_secs=.5):
    print('gr')
    time.sleep(sleep_secs)
    return 2*(x-14)

o3 = minimize_parallel(fun=f, x0=x0, jac=g, args=.5)
o4 = minimize(fun=f, x0=x0, jac=g, args=.5)

print(all(np.isclose(o3.x, o4.x, atol=1e-5)),
      np.isclose(o3.fun, o4.fun, atol=1e-5),
      all(np.isclose(o3.jac, o4.jac, atol=1e-5)))
