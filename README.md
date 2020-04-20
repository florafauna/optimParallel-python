# optimParallel-python
A parallel computing interface to the L-BFGS-B optimizer.

### Goal: 
Provide a parallel version of `scipy.optimize.minimize(method=’L-BFGS-B’)`. This is, for each step of the optimization the objective function `fn` and all computations involved to evaluate its gradient `gr` are evaluated in parallel. 

A similar extension of L-BFGS-B exists in the R package *optimParallel*:
- https://CRAN.R-project.org/package=optimParallel 
- https://doi.org/10.32614/RJ-2019-030

### Milestones:
1. Create a class `fg`, which takes a function `f` and optionally its gradient `g`. 
  `fg.f(x)` should evaluate `f` and `g` in parallel, store the return values in attributes, and return `f(x)`.
  `fg.g(x)` if `x` was already evaluated via `fg.f(x)`, return `g(x)` without doing any computations. 
2. Create the function `optimParallel()` that evaluates `scipy.optimize.minimize(method=’L-BFGS-B’)` in parallel using the `fg` class. 
3. Create unit tests characterizing the desired behavior of `optimParallel()`. Take into account all options of `scipy.optimize.minimize(method=’L-BFGS-B’)`.
4. Add functionalities to `optimParallel()` and `fg` until all tests from 3. work as expected.
5. Write documentation. 

### Contributions:
Contributions via pull requests are welcome. 
