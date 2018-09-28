# BalancePoint

Implementation of balencing technique for data-flow code. 

1. The Gao's Linear program formulation (https://doi.org/10.1016/0743-7315(89)90041-5). 
This algorithm generates an optimally balanced graph, that is, a graph that requires a minimum amount of buffering.

2. The Apple Linear program formulation. This algorithm who generates the best-balanced graph possible given a budget of buffers. If the budget is unlimited, it's generated a balanced graph.

# Requirement
  - cvoxp (http://cvxopt.org/install/)
  - numpy
  - networkx (optional https://networkx.github.io/documentation/stable/install.html)
