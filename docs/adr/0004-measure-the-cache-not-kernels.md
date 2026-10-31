# 4. Measure the cache, not kernels

The generated computation calls the same PyTorch operations as eager, with no fusion, and runs
on CPU (no GPU here). Its time over eager is the cost of tracing, code generation and prologue
checks. That is what this project is about: how many programs a workload needs, how expensive
they are to check, and whether they are right. Kernel performance would come from an executor
behind the same trace (nvFuser, Inductor), which is out of scope.
