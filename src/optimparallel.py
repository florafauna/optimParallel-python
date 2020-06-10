"""
A parallel version of the L-BFGS-B optimizer of `scipy.optimize.minimize()`.

Using it can significantly reduce the optimization time. For an objective
function with p parameters the optimization speed increases by up to
factor 1+p, when no analytic gradient is specified and 1+p processor cores
with sufficient memory are available.

Functions
---------
- minimize_parallel: parallel version of `scipy.optimize.minimize()`
- optimParallel: same as `minimize_parallel()`
- optimparallel: same as `minimize_parallel()`
- fmin_l_bfgs_b_parallel: parallel version of `scipy.optimize.fmin_l_bfgs_b()`
"""

import warnings
import concurrent.futures
import functools
import itertools
import numpy as np
from scipy.optimize import minimize
import time

__all__ = ['minimize_parallel', 'fmin_l_bfgs_b_parallel', 'optimParallel', 'optimparallel']

class EvalParallel:
    def __init__(self, fun, jac=None, args=(), eps=1e-8,
                 executor=concurrent.futures.ProcessPoolExecutor(),
                 forward=True, loginfo=False, verbose=False, n=1):
        self.fun_in = fun
        self.jac_in = jac
        self.eps = eps
        self.forward = forward
        self.loginfo = loginfo
        self.verbose = verbose
        self.x_val = None
        self.fun_val = None
        self.jac_val = None
        if not (isinstance(args, list) or isinstance(args, tuple)):
            self.args = (args,)
        else:
            self.args = tuple(args)
        self.n = n
        self.executor = executor
        if self.loginfo:
            self.info = {k:[] for k in ['x', 'fun', 'jac']}
        self.np_precision = np.finfo(float).eps

    ## static helper methods are used for parallel execution with map()
    @staticmethod
    def _eval_approx_args(args, eps_at, fun, x, eps):
        ## 'fun' has additional 'args'
        if eps_at == 0:
            x_ = x
        elif eps_at <= len(x):
            x_ = x.copy()
            x_[eps_at-1] += eps
        else:
            x_ = x.copy()
            x_[eps_at-1-len(x)] -= eps
        return fun(x_, *args)

    @staticmethod
    def _eval_approx(eps_at, fun, x, eps):
        ## 'fun' has no additional 'args'
        if eps_at == 0:
            x_ = x
        elif eps_at <= len(x):
            x_ = x.copy()
            x_[eps_at-1] += eps
        else:
            x_ = x.copy()
            x_[eps_at-1-len(x)] -= eps
        return fun(x_)

    @staticmethod
    def _eval_fun_jac_args(args, which, fun, jac, x):
        ## 'fun' and 'jec; have additional 'args'
        if which == 0:
            return fun(x, *args)
        return np.array(jac(x, *args))

    @staticmethod
    def _eval_fun_jac(which, fun, jac, x):
        ## 'fun' and 'jac' have no additionals 'args'
        if which == 0:
            return fun(x)
        return np.array(jac(x))

    def eval_parallel(self, x):
        ## function to evaluate 'fun' and 'jac' in parallel
        ## - if 'jac' is None, the gradient is computed numerically
        ## - if 'forward' is True, the numerical gradient uses the
        ##       forward difference method,
        ##       otherwise, the central difference method is used
        x = np.array(x)
        if (self.x_val is not None and
            all(abs(self.x_val - x) <= self.np_precision*2)):
            if self.verbose:
                print('re-use')
        else:
            self.x_val = x.copy()
            if self.jac_in is None:
                if self.forward:
                    eps_at = range(len(x)+1)
                else:
                    eps_at = range(2*len(x)+1)

                ## pack 'self.args' into function because otherwise it
                ## cannot be serialized by
                ## 'concurrent.futures.ProcessPoolExecutor()'
                if len(self.args) > 0:
                    ftmp = functools.partial(self._eval_approx_args, self.args)
                else:
                    ftmp = self._eval_approx

                ret = self.executor.map(ftmp, eps_at,
                                    itertools.repeat(self.fun_in),
                                    itertools.repeat(x),
                                    itertools.repeat(self.eps))
                ret = np.array(list(ret))
                self.fun_val = ret[0]
                if self.forward:
                    self.jac_val = (ret[1:(len(x)+1)] - self.fun_val ) / self.eps
                else:
                    self.jac_val = (ret[1:(len(x)+1)]
                                - ret[(len(x)+1):2*len(x)+1]) / (2*self.eps)

            # 'jac' function is not None
            else:
                if len(self.args) > 0:
                    ftmp = functools.partial(self._eval_fun_jac_args, self.args)
                else:
                    ftmp = self._eval_fun_jac

                ret = self.executor.map(ftmp, [0,1],
                                        itertools.repeat(self.fun_in),
                                        itertools.repeat(self.jac_in),
                                        itertools.repeat(x))
                ret = list(ret)
                self.fun_val = ret[0]
                self.jac_val = ret[1]

            self.jac_val = self.jac_val.reshape((self.n,))

            if self.loginfo:
                self.info['fun'].append(self.fun_val)
                if self.n >= 2:
                    self.info['x'].append(self.x_val.tolist())
                    self.info['jac'].append(self.jac_val.tolist())
                else:
                    self.info['x'].append(self.x_val[0])
                    self.info['jac'].append(self.jac_val[0])
        return None

    def fun(self, x):
        self.eval_parallel(x=x)
        if self.verbose:
            print('fun(' + str(x) + ') = ' + str(self.fun_val))
        return self.fun_val

    def jac(self, x):
        self.eval_parallel(x=x)
        if self.verbose:
            print('jac(' + str(x)+ ') = ' + str(self.jac_val))
        return self.jac_val


def minimize_parallel(fun, x0,
                      args=(),
                      jac=None,
                      bounds=None,
                      tol=None,
                      options=None,
                      callback=None,
                      parallel=None):

    """
    A parallel version of the L-BFGS-B optimizer of
    `scipy.optimize.minimize()`. Using it can significantly reduce the
    optimization time. For an objective function with an execution time
    of more than 0.1 seconds and p parameters the optimization speed
    increases by about factor 1+p when no analytic gradient is specified
    and 1+p processor cores with sufficient memory are available.

    Parameters
    ----------
    `fun`, `x0`, `args`, `jac`, `bounds`, `tol`, `options`, and `callback`
    are the same as the corresponding arguments of `scipy.optimize.minimize()`.
    See the documentation of `scipy.optimize.minimize()` and
    `scipy.optimize.minimize(method='L-BFGS-B')` for more information.

    Additional arguments controlling the parallel execution are:

    parallel: dict
        max_workers: The maximum number of processes that can be
            used to execute the given calls. The value is passed
            to the `max_workers` argument of
            `concurrent.futures.ProcessPoolExecutor()`.

        forward: bool. If `True` (default), the forward difference method is
            used to approximate the gradient when `jac` is `None`.
            If `False`, the central difference method is used, which can be more
            accurate.

        verbose: bool. If `True`, additional output is printed to the console.

        loginfo: bool. If `True`, additional log information containing the
            evaluated parameters as well as return values of
            fun and jac is returned.

        time: bool. If `True`, a dict containing the elapsed time (seconds)
            and the elapsed time per step (evaluation of one 'fun' call and
            its jacobian) is returned.

    Note
    ----
    On Windows it may be necessary to run `minimize_parallel()` in the
    main scope. See the example in
    <https://github.com/florafauna/optimParallel-python/blob/master/example_windows_os.py>.

    Because of the parallel overhead, `minimize_parallel()` is only faster than
    `minimize()` for objective functions with an execution time of more than 0.1
    seconds.

    When `jac=None` and `bounds` are specified, it can be advisable to
    increase the lower bounds by `eps` and decrease the upper bounds by `eps`
    because the optimizer might try to evaluate fun(upper+eps) and
    fun(upper-eps). `eps` is specified in `options` and defaults to `1e-8`,
    see `scipy.optimize.minimize(method='L-BFGS-B')`.

    Example
    -------
    >>> from optimparallel import minimize_parallel
    >>> from scipy.optimize import minimize
    >>> import numpy as np
    >>> import time
    >>>
    >>> ## objective function
    ... def f(x, sleep_secs=.5):
    ...     print('*', end='')
    ...     time.sleep(sleep_secs)
    ...     return sum((x-14)**2)
    >>>
    >>> ## start value
    ... x0 = np.array([10,20])
    >>>
    >>> ## minimize with parallel evaluation of 'fun' and
    >>> ## its approximate gradient.
    >>> o1 = minimize_parallel(fun=f, x0=x0, args=.5)
    *********
    >>> print(o1)
           fun: 2.17075906477993e-12
     hess_inv: array([[0.84615401, 0.23076919],
           [0.23076919, 0.65384592]])
          jac: array([1.94333641e-06, 2.23379136e-06])
      message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
         nfev: 3
          nit: 2
       status: 0
      success: True
            x: array([14.00000097, 14.00000111])
    >>>
    >>> ## test against scipy.optimize.minimize()
    >>> o2 = minimize(fun=f, x0=x0, args=.5, method='L-BFGS-B')
    *********
    >>> print(all(np.isclose(o1.x, o2.x, atol=1e-10)),
    ...       np.isclose(o1.fun, o2.fun, atol=1e-10),
    ...       all(np.isclose(o1.jac, o2.jac, atol=1e-10)))
    True True True

    References
    ----------
    When using the package please cite:
    F. Gerber and R. Furrer (2019) optimParallel: An R package providing
    a parallel version of the L-BFGS-B optimization method.
    The R Journal, 11(1):352-358, 2019,
    https://doi.org/10.32614/RJ-2019-030

    R package with similar functionality:
    https://CRAN.R-project.org/package=optimParallel

    Source code of Python module:
    https://github.com/florafauna/optimParallel-python

    References for the L-BFGS-B optimization code are listed in the help
    page of `scipy.optimize.minimize()`.

    Authors
    -------
    Florian Gerber, flora.fauna.gerber@gmail.com
    https://user.math.uzh.ch/gerber/index.html

    Lewis Blake (testing, 'loginfo', 'time' features).
    """

    ## get length of x0
    try:
        n = len(x0)
    except Exception:
        n = 1

    if jac is True:
        raise ValueError("'fun' returning the function AND its "
                         "gradient is not supported.\n"
                         "Please specify separate functions in "
                         "'fun' and 'jac'.")

    ## update default options with specified options
    options_used = {'disp': None, 'maxcor': 10,
                    'ftol': 2.220446049250313e-09, 'gtol': 1e-05,
                    'eps': 1e-08, 'maxfun': 15000,
                    'maxiter': 15000, 'iprint': -1, 'maxls': 20}
    if not options is None:
        if not isinstance(options, dict):
            raise TypeError("argument 'options' must be of type 'dict'")
        options_used.update(options)
    if not tol is None:
        if not options is None and 'gtol' in options:
            warnings.warn("'tol' is ignored and 'gtol' in 'opitons' is used insetad.",
                          RuntimeWarning)
        else:
            options_used['gtol'] = tol

    parallel_used = {'max_workers': None, 'forward': True, 'verbose': False,
                     'loginfo': False, 'time': False}
    if not parallel is None:
        if not isinstance(parallel, dict):
            raise TypeError("argument 'parallel' must be of type 'dict'")
        parallel_used.update(parallel)

    if parallel_used.get('time'):
        time_start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=
                         parallel_used.get('max_workers')) as executor:
        fun_jac = EvalParallel(fun=fun,
                               jac=jac,
                               args=args,
                               eps=options_used.get('eps'),
                               executor=executor,
                               forward=parallel_used.get('forward'),
                               loginfo=parallel_used.get('loginfo'),
                               verbose=parallel_used.get('verbose'),
                               n=n)
        out = minimize(fun=fun_jac.fun,
                       x0=x0,
                       jac=fun_jac.jac,
                       method='L-BFGS-B',
                       bounds=bounds,
                       callback=callback,
                       options=options_used)

    out.hess_inv = out.hess_inv * np.identity(n)

    if parallel_used.get('loginfo'):
        out.loginfo = {k: (lambda x: np.array(x) if isinstance(x[0], list)
                           else np.array(x)[np.newaxis].T)(v) for k, v in fun_jac.info.items()}

    if parallel_used.get('time'):
        time_end = time.time()
        out.time = {'elapsed': time_end - time_start,
                    'step': (time_end - time_start) / out.nfev}

    return out


def fmin_l_bfgs_b_parallel(func, x0, fprime=None, args=(), approx_grad=0,
                           bounds=None, m=10, factr=1e7, pgtol=1e-5,
                           epsilon=1e-8, iprint=-1, maxfun=15000,
                           maxiter=15000, disp=None, callback=None, maxls=20,
                           parallel=None):

    """
    A parallel version of the L-BFGS-B optimizer `fmin_l_bfgs_b()`.
    Using it can significantly reduce the optimization time.
    For an objective function with an execution time of more than
    0.1 seconds and p parameters the optimization speed increases
    by about factor 1+p when no analytic gradient is specified
    and 1+p processor cores with sufficient memory are available.

    Parameters
    ----------
    `func`, `x0`, `fprime`, `args`, `approx_grad`, `bounds`, `m`, `factr`,
    `pgtol`, `epsilon`, `iprint`, `maxfun`, `maxiter`, `disp`, `callback`,
    `maxls` are the same as in  `fmin_l_bfgs_b()`.

    Additional arguments controlling the parallel execution are:

    parallel: dict
        max_workers: The maximum number of processes that can be
            used to execute the given calls. The value is passed
            to the `max_workers` argument of
            `concurrent.futures.ProcessPoolExecutor()`.

        forward: bool. If `True` (default), the forward difference method is
            used to approximate the gradient when `jac` is `None`.
            If `False` the central difference method is used.

        verbose: bool. If `True`, additional output is printed to the console.

        loginfo: bool. If `True`, additional log information containing the
            evaluated parameters as well as return values of
            fun and jac is returned.

        time: bool. If `True`, a dict containing the elapsed time (seconds)
            and the elapsed time per step (evaluation of one 'fun' call and
            its jacobian) is returned.

    Note
    ----
    On Windows it may be necessary to run `minimize_parallel()` in the
    main scope. See the example in
    <https://github.com/florafauna/optimParallel-python/blob/master/example_windows_os.py>.

    Because of the parallel overhead, `minimize_parallel()` is only faster than
    `minimize()` for objective functions with an execution time of more than 0.1
    seconds.

    When `approx_grad=True` and `bounds` are specified, it can be advisable to
    increase the lower bounds by `eps` and decrease the upper bounds by `eps`
    because the optimizer might try to evaluate fun(upper+eps) and if 'forward
    is True, fun(upper-eps).

    References
    ----------
    When using the package please cite:
    F. Gerber and R. Furrer (2019) optimParallel: An R package providing
    a parallel version of the L-BFGS-B optimization method.
    The R Journal, 11(1):352-358, 2019,
    https://doi.org/10.32614/RJ-2019-030

    R package with similar functionality:
    https://CRAN.R-project.org/package=optimParallel

    Source code of Python module:
    https://github.com/florafauna/optimParallel-python

    References for the L-BFGS-B optimization code are listed in the help
    page of `scipy.optimize.minimize()`.

    Authors
    -------
    Florian Gerber, flora.fauna.gerber@gmail.com
    https://user.math.uzh.ch/gerber/index.html

    Lewis Blake (testing, 'loginfo', 'time' features).
    """

    fun = func
    if approx_grad:
        jac = None
    else:
        if fprime is None:
            raise ValueError("'func' returning the function AND its "
                             "gradient is not supported.\n"
                             "Please specify separate functions in "
                             "'func' and 'fprime'.")
        jac = fprime

    # build options
    if disp is None:
        disp = iprint
    options = {'disp': disp,
               'iprint': iprint,
               'maxcor': m,
               'ftol': factr * np.finfo(float).eps,
               'gtol': pgtol,
               'eps': epsilon,
               'maxfun': maxfun,
               'maxiter': maxiter,
               'maxls': maxls}

    res = minimize_parallel(fun=fun, x0=x0, args=args, jac=jac, bounds=bounds,
                            options=options, callback=callback,
                            parallel=parallel)
    x = res['x']
    f = res['fun']
    d = {'grad': res['jac'],
         'task': res['message'],
         'funcalls': res['nfev'],
         'nit': res['nit'],
         'warnflag': res['status']}

    return x, f, d

## create aliases for users looking for an optimparallel function
optimParallel = minimize_parallel
optimParallel.__doc__ = "Same function as `minimize_parallel()`. See `help(minimize_parallel."
optimparallel = minimize_parallel
optimparallel.__doc__ = optimParallel.__doc__
