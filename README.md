# optimparallel - A parallel version of `scipy.optimize.minimize(method='L-BFGS-B')`

[![DOI](https://zenodo.org/badge/257319138.svg)](https://zenodo.org/badge/latestdoi/257319138)
[![PyPI](https://img.shields.io/pypi/v/optimparallel)](https://pypi.org/project/optimparallel)
[![Build Status](https://travis-ci.org/florafauna/optimParallel-python.svg?branch=master)](https://travis-ci.org/florafauna/optimParallel-python)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9bb33b3e786940af972da1835847c582)](https://www.codacy.com/manual/florafauna/optimParallel-python?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=florafauna/optimParallel-python&amp;utm_campaign=Badge_Grade)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Using `optimparallel.minimize_parallel()` can significantly reduce the
optimization time. For an objective function with an execution time
of more than 0.1 seconds and p parameters the optimization speed
increases by up to factor 1+p when no analytic gradient is specified
and 1+p processor cores with sufficient memory are available.

A similar extension of the L-BFGS-B optimizer exists in the R package *optimParallel*:
*   [optimParallel on CRAN](https://CRAN.R-project.org/package=optimParallel)
*   [R Journal article](https://doi.org/10.32614/RJ-2019-030)

## Installation

To install the package run:

```bash
$ pip install optimparallel
```

## Usage

Replace `scipy.optimize.minimize(method='L-BFGS-B')` by `optimparallel.minimize_parallel()`
to execute the minimization in parallel:

```python
from optimparallel import minimize_parallel
from scipy.optimize import minimize
import numpy as np
import time

## objective function
def f(x, sleep_secs=.5):
    print('fn')
    time.sleep(sleep_secs)
    return sum((x-14)**2)

## start value
x0 = np.array([10,20])

## minimize with parallel evaluation of 'fun' and
## its approximate gradient.
o1 = minimize_parallel(fun=f, x0=x0, args=.5)
print(o1)

## test against scipy.optimize.minimize()
o2 = minimize(fun=f, x0=x0, args=.5, method='L-BFGS-B')
print(all(np.isclose(o1.x, o2.x, atol=1e-10)),
      np.isclose(o1.fun, o2.fun, atol=1e-10),
      all(np.isclose(o1.jac, o2.jac, atol=1e-10)))
```

The evaluated `x` values, `fun(x)`, and `jac(x)` can be returned:

```python
o1 = minimize_parallel(fun=f, x0=x0, args=.5, parallel={'loginfo': True})
print(o1.loginfo)
```

More examples are given in [example.py](https://github.com/florafauna/optimParallel-python/blob/master/example.py) and the Jupyter Notebook [example_extended.ipynb](https://github.com/florafauna/optimParallel-python/blob/master/example_extended.ipynb).

**Note for Windows users**: It may be necessary to run `minimize_parallel()` in the main scope. See [example_windows_os.py](https://github.com/florafauna/optimParallel-python/blob/master/example_windows_os.py).

## Citation

When using this package please cite:

*   Gerber, F. _optimparallel - A parallel version of `scipy.optimize.minimize(method='L-BFGS-B')_, 2020, <http://doi.org/10.5281/zenodo.3888570>
*   Gerber, F., and Furrer, R. _optimParallel: An R package providing a parallel version of the L-BFGS-B optimization method._ The R Journal, 11(1):352–358, 2019. <http://doi.org/10.32614/RJ-2019-030>


### Author

*   Florian Gerber, <flora.fauna.gerber@gmail.com>, <https://user.math.uzh.ch/gerber>.

### Contributor

*   Lewis Blake

### Contributions
Contributions via pull requests are welcome.

*   <https://github.com/florafauna/optimParallel-python>
*   <https://pypi.org/project/optimparallel>

