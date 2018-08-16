#!/usr/bin/env python
import numpy as np
import logging # Please change logging info for more information about nomad optimization

from collections import defaultdict
from itertools import chain,tee
from functools import lru_cache, partial

# Python Typing
from typing import List, Dict, Tuple, Set, NewType
Node = NewType('Node', str)
Edge = Tuple[Node, Node]

def lazy_property(f):
    return property(lru_cache()(f))


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Swing(object):
    '''
    This object takes as input a weighted acyclic graph and maximizes the pipelining capabilities of the graph.

    The key two menbers are:
        1- opt_edge_buffer -> Optimal buffering need to balence the graph
        2- constrained_edge_buffer(max_b) -> Optimal constrain balenced graph

    For (1) we use Gao algorithm based on a Linear Programming formulation
    (2) use a minmax formulation where we minimize the maximum difference between each path and the critical path subject to a limited number of buffers.
    In order to reduce the number of parameters, we optimize only the edge created by (1).

    Dependency:

    (1):cvxopt and glpk (conda install cxopt)
    (2): Nomad (https://www.gerad.ca/nomad/) and PyNomad (see Nomad user guide on how to install)
    '''

    def __init__(self, weigh):
        self.weigh = weigh  # Top-> bottom by default.
        logging.info(f'Unconstrained: {len(weigh)} edges to optimize')

    @lazy_property
    def adjacency_list_topdown(self) -> Dict[Node, List[Node]]:
        d = defaultdict(list)
        for i, o in self.weigh:
            d[i].append(o)
        return dict(d)

    @lazy_property
    def adjacency_list_bottomup(self) -> Dict[Node, List[Node]]:
        d = defaultdict(list)
        for k, lv in self.adjacency_list_topdown.items():
            for v in lv:
                #print(k,lv,v)
                d[v].append(k)
        return dict(d)

    @lazy_property
    def l_node(self) -> Set[Node]:
        ''' list of nodes'''
        return self.adjacency_list_topdown.keys() | self.adjacency_list_bottomup.keys()

    @lazy_property
    def order(self) -> Dict[Node, int]:
        '''Arbitrary nodes labeling'''
        return {k: i for i, k in enumerate(self.l_node)}

    @lazy_property
    def leaf(self) -> Node:
        '''
        Return the leaf node. 
        Assume only one leaf for now
        '''
        leafs = self.l_node - self.adjacency_list_topdown.keys()
        if len(leafs) == 1:
            return leafs.pop()
        else:
            raise NotImplementedError("Multiple leafs found. Not implemted yet")

    @lazy_property
    def delta_degree(self) -> Dict[Node, int]:
        '''indegree(i) - outdegree(i) for each node. '''
        in_ = self.adjacency_list_topdown
        out_ = self.adjacency_list_bottomup

        g = lambda d,n: len(d[n]) if n in d else 0 # Handle leaf and root
        return {n: g(in_,n) - g(out_,n) for n in self.l_node}

    @lru_cache(None)
    def path_node(self, cur: Node) -> List[List[Node]]:
        '''Compute all the node form cur to the root of the DAG.
            Assume a unique root'''
        d = self.adjacency_list_bottomup

        if cur in d:
            it = chain.from_iterable(self.path_node(p) for p in d[cur])
        else:
            it = iter([[]])

        return [k + [cur] for k in it]

    @lazy_property
    def path_edges(self) -> List[List[Edge]]:
        return [ list(pairwise(p)) for p in self.path_node(self.leaf)]

    @lazy_property
    def path(self) -> Dict[Tuple[Edge], int]:
        '''
        Compute for all the path ( of the form (e1->e2),(e2->e3) )
        and the sum of the weigh from the leaf to the read
        '''
        d = dict()

        for p in self.path_node(self.leaf):
            path_edge = tuple(pairwise(p))
            d[path_edge] = sum(map(self.weigh.get, path_edge))

        return d

    @lazy_property
    def critical_weigh(self) -> int:
        cw = max(self.path.values())
        logging.info(f'Critical path weigh: {cw}')
        return cw

    @lazy_property
    def non_critical_path(self) -> Dict[Tuple[Edge], int]:
        c = self.critical_weigh
        d = {p: w for p, w in self.path.items() if w != c}
        logging.info(f'{len(d)} paths to optimize')
        return d
    #                        _
    # |  o ._   _   _. ._   |_) ._ _   _  ._ _. ._ _
    # |_ | | | (/_ (_| |    |   | (_) (_| | (_| | | |
    #                                  _|
    #
    # https://doi.org/10.1016/0743-7315(89)90041-5
    #
    # Maximize \sum_i u_i (outdegree(i) - indegree(i))
    # Subject to
    #       u_i - u_j \leq -w_{ij} for all (i,j) \subseteq E
    #       u_n - u_1 = w_{st}
    #       u_i >= 0

    @lazy_property
    def lp_constrain_matrix(self):

        od = self.order
        n_node, n_edge = map(len, (self.l_node, self.weigh))

        A = np.zeros((n_edge, n_node), float)
        for idx, (i, o) in enumerate(self.weigh):
            A[idx, od[i]] = 1
            A[idx, od[o]] = -1

        A_pos = np.identity(n_node) * -1 # Expant with positif contrains
        return np.concatenate((A, A_pos), axis=0)

    @lazy_property
    def lp_constrain_vector(self):
        b = np.array([-k for k in self.weigh.values()], dtype=float)
        b_pos = np.zeros(len(self.l_node))  # Positif contrains
        return np.concatenate((b, b_pos))

    @lazy_property
    def lp_objective_vector(self):
        # CVXopt can only minimazise so the multiply the objective vector by -1
        # Float is required by CVXopt
        return np.array([-1 * self.delta_degree[i] for i in self.order], dtype=float)

    @lazy_property
    def lp_opt_firing(self) -> Dict[Node,int]:
        '''
        Gao's algorithm, using Liner Programming formulation
        Compute the optimal firing of each node for the now balenced graph
        '''

        from cvxopt import matrix, solvers
        # We add the constrain that each objectif need to me postiif
        A, b = map(matrix, (self.lp_constrain_matrix, self.lp_constrain_vector))
        c = matrix(self.lp_objective_vector)

        #Integer solution use GLPK and turn off verbose output
        solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
        sol = solvers.lp(c, A, b, solver='glpk')

        x, x_int = sol['x'], [int(round(i)) for i in sol['x']]
        assert all(i == f for i, f in zip(x_int, x)), 'Some none integer buffers where found'
        return dict(zip(self.order, x_int))  # Assume ordered dict

    @lazy_property
    def opt_edge_buffer(self) -> Dict[Edge, int]:
        '''
        Needed buffer to optimally balance the graph
        '''
        f, w = self.lp_opt_firing, self.weigh
        d =  {(i, o): (f[o] - f[i]) - w for (i, o), w in w.items() if (f[o] - f[i]) != w}
        logging.info(f'Unconstrained: {len(d)} distinct buffers ({sum(d.values())} values in total) are needed to optimaly balence the graph')
        return d

    # 
    # |\/| o ._  |\/|  _.
    # |  | | | | |  | (_| ><
    #
    # Miminize  max(u_n - u_1)
    # Subject to
    #       u_n - u_1 \geq 0
    #       u_i unrestricted
    def bb(self, x, edges_adjacency_matrix, fixed_weighs, max_b) -> int:
        '''Black Box function who compute:.
            f(x) = max(fixed_weighs-sum(edges_adjacency_matrix*x))  (1)     
            Subject to:
                sum(x) > max_b                                      (2)
                fixed_weighs-sum(edges_adjacency_matrix*x) >= 0     (3)
        '''

        dim = x.get_n()
        buffers = [x.get_coord(i) for i in range(dim)]

        delta = fixed_weighs - np.sum(edges_adjacency_matrix * buffers, axis=1)

        x.set_bb_output(0, max(delta))           # (1)
        x.set_bb_output(1, sum(buffers) - max_b)  # (2)

        for i, v in enumerate(delta, 2):
            x.set_bb_output(i, -v)               #(3) Minus sign: \ge 0 -> \le 0 

        return 1

    def constrained_edge_buffer(self, max_b: int, max_bb_eval: int = 100) -> Tuple[Dict[Edge, int], int]:
        '''
        Optimize min(max(cw-w)) where
                cw,w are the sum of weighs of the critical path, and other paths respectively
        under the constrain 
                sum(b) < max_b where bs are the buffers used in the aforesaid paths
        
        max_b <=0, meens no restriction on the number of buffers

        Return the dictionary of buffers needed for eatch edges and the max(cw-w) value.
        '''
        # Initialization
        opt_edg_buf = self.opt_edge_buffer
        if max_b <= 0:
            return opt_edg_buf, 0

        nc_path = self.non_critical_path
        logging.info(f'Constrained: {len(opt_edg_buf)} edges to optimize')

        # Edge adjacency matrix and  fixed weighs
        e_adj = np.array( [[e in pairs for e in opt_edg_buf] for pairs in nc_path], dtype=int)
        weighs = self.critical_weigh - np.array(list(nc_path.values()), dtype=int)

        bb = partial(self.bb, max_b=max_b, edges_adjacency_matrix=e_adj, fixed_weighs=weighs)

        # Nomad parameter 0 < buffer < optimal result
        lb = [0] * len(opt_edg_buf)
        ub = list(opt_edg_buf.values())

        # For now, the starting point of the minmax optimation is set arbitrary to the lower bound (no buffer)
        x0 = ub #lb

        params = [
            f'BB_OUTPUT_TYPE OBJ EB {" ".join(["EB"] * len(nc_path))}',  # Each path have a constrain
            f'MAX_BB_EVAL {max_bb_eval}',
            'LOWER_BOUND * 0',  # Buffer are postif
            'BB_INPUT_TYPE * I',  # All integer
            f'DISPLAY_DEGREE {0 if logging.getLogger().getEffectiveLevel() >= 30 else 1}'
        ]

        import PyNomad
        logging.debug('Starting Nomad evaluation')
        x_return, f_return, h_return, nb_evals, nb_iters, stopflag = PyNomad.optimize(bb, x0, lb, ub, params)

        if nb_evals == max_bb_eval:
            logging.warning(f'The number of maximun evalation ({max_bb_eval}) has been reached.'
                             ' Maybe inscrease max_bb_eval threshold.')

        # Map back to the name
        l_buffer_updated = {k: int(v) for k, v in zip(opt_edg_buf, x_return) if v}
        return l_buffer_updated, f_return

    @lazy_property
    def max_traffic(self) -> List[Edge]:
        d = defaultdict(int)
        for i in self.path_edges:
            for o in i:
                d[o] += 1

        return dict(d)

