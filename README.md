# optimParallel-python
A parallel computing interface to the L-BFGS-B optimizer.

### Goal: 
Provide a parallel version of `scipy.optimize.minimize(method=’L-BFGS-B’)`.
This is, for each step of the optimization the objective function `fun` and all computations involved to evaluate its gradient `jac` are evaluated in parallel. 

A similar extension of L-BFGS-B exists in the R package *optimParallel*:
- https://CRAN.R-project.org/package=optimParallel 
- https://doi.org/10.32614/RJ-2019-030

### Milestones:
1. [DONE] Create a first working example of `minimize_parallel()` without fancy extra options. 
2. Create unit tests characterizing the desired behavior of `minimize_parallel()`.
Take into account all options of `scipy.optimize.minimize(method=’L-BFGS-B’)`.
Test can be run with `python -m pytest'
3. Add functionalities to `minimize_parallel()` until all tests from 2. work as expected.
4. Write documentation. 

### Contributions:
Contributions via pull requests are welcome. 
