# optimParallel-python
A parallel computing interface to the L-BFGS-B optimizer.

A parallel version of the L-BFGS-B optimizer of `scipy.optimize.minimize()`.
Using it can significantly reduce the optimization time. For an objective
function with an execution time of more than 0.1 seconds and p parameters
the optimization speed increases by up to factor 1+p when no analytic
gradient is specified and 1+p processor cores with sufficient memory
are available. 

A similar extension of the L-BFGS-B optimizer exists in the R package *optimParallel*:
  - [https://CRAN.R-project.org/package=optimParallel]
  - [https://doi.org/10.32614/RJ-2019-030]

An example is given in [example.py](example.py). 

## Contributions
Contributions via pull requests are welcome. 
