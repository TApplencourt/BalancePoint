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
    def critical_paths(self):
        c =  self.critical_weigh
        d = {p: w for p, w in self.path.items() if w == c}
        logging.info(f'{len(d)} critical path')
        return d

    @lazy_property
    def is_balenced(self):
        return len(self.critical_paths) == len(self.path)

    @lazy_property
    def critical_weigh(self) -> int:
        cw = max(self.path.values())
        logging.info(f'Critical path weigh: {cw}')
        return cw

    @lazy_property
    def non_critical_paths(self) -> Dict[Tuple[Edge], int]:
        c = self.critical_weigh
        d = {p: w for p, w in self.path.items() if w != c}
        logging.info(f'{len(d)} paths non critical paths')
        return d

class Gao():

    #                        _
    # |  o ._   _   _. ._   |_) ._ _   _  ._ _. ._ _
    # |_ | | | (/_ (_| |    |   | (_) (_| | (_| | | |
    #                                  _|
    #
    # Gao' paper: https://doi.org/10.1016/0743-7315(89)90041-5
    #
    # Maximize \sum_i u_i (outdegree(i) - indegree(i))
    # Subject to
    #       u_i - u_j \leq -w_{ij} for all (i,j) \subseteq E
    #       u_n - u_1 = w_{st}
    #       u_i >= 0

    def __init__(self,s):
        self.swing = s

            
    @lazy_property
    def constrain_matrix(self):

        s = self.swing

        od = s.order
        n_node, n_edge = map(len, (s.l_node, s.weigh))

        A = np.zeros((n_edge, n_node), dtype=float)
        for idx, (i, o) in enumerate(s.weigh):
            A[idx, od[i]] = 1
            A[idx, od[o]] = -1

        A_pos = np.identity(n_node) # Expant with positif contrains
        return np.concatenate((A, -A_pos), axis=0)

    @lazy_property
    def constrain_vector(self):
        b = -np.array(list(self.swing.weigh.values()), dtype=float)
        b_pos = np.zeros(len(self.swing.l_node))  # Positif contrains
        return np.concatenate((b, b_pos))

    @lazy_property
    def objective_vector(self):
        # CVXopt can only minimazise so the multiply the objective vector by -1
        # Float is required by CVXopt
        return np.array([-1 * self.swing.delta_degree[i] for i in self.swing.order], dtype=float)

    @lazy_property
    def optimal_firing(self) -> Dict[Node,int]:
        '''
        Gao's algorithm, using Linear Programming formulation
        Compute the optimal firing of each node for the now balenced graph
        '''

        from cvxopt import matrix, solvers
        # We add the constrain that each objectif need to me postiif
        A, b, c = map(matrix, (self.constrain_matrix, self.constrain_vector, self.objective_vector))

        #Integer solution use GLPK and turn off verbose output
        solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
        sol = solvers.lp(c, A, b, solver='glpk')

        x, x_int = sol['x'], [int(round(i)) for i in sol['x']]
        assert all(i == f for i, f in zip(x_int, x)), 'Some none integer buffers where found'
        return dict(zip(self.swing.order, x_int))  # Assume ordered dict

    @lazy_property
    def optimal_buffer(self) -> Dict[Edge, int]:
        '''
        Needed buffer to optimally balance the graph
        '''
        f, w = self.optimal_firing, self.swing.weigh
        d =  {(i, o): (f[o] - f[i]) - w for (i, o), w in w.items() if (f[o] - f[i]) != w}
        logging.info(f'Unconstrained: {len(d)} distinct buffers ({sum(d.values())} values in total) are needed to optimaly balence the graph')
        return d

class Apple():

    def __init__(self,s):
        self.swing = s
        self.W = s.weigh
        self.edges = len(self.W)

    @lazy_property
    def constrain_matrix(self):

        CP = self.swing.critical_paths
        P = self.swing.non_critical_paths
        paths = len(P)

        edges_in_cp = set(chain.from_iterable(CP))
        matrix_path = np.matrix([ [int(e in p and e not in edges_in_cp) for e in self.W] for p in P])

        o,z = (f(paths).reshape(-1,1) for f in (np.ones,  np.zeros ))

        # Eta
        e = np.concatenate((-o,-matrix_path),axis=1)
        # Delta >=0
        d = np.concatenate((z,matrix_path),axis= 1)
        # positif constraint
        A_pos = np.identity(self.edges+1) # Expant with positif contrains
        return np.concatenate((e,d,-A_pos))

    @lazy_property
    def constrain_vector(self):
        weight_path = np.array(list(self.swing.non_critical_paths.values()))
        # Sum weight_path
        b = -next(iter(self.swing.critical_paths.values())) + weight_path
        # postof constraint
        b_pos = np.zeros(self.edges+1)
        return np.concatenate((b,-b,b_pos))

    @lazy_property
    def objective_vector(self):
        return np.concatenate( ([1],np.zeros(self.edges)))    

    @lazy_property
    def optimal_buffer(self) -> Dict[Edge, int]:
        from cvxopt import matrix, solvers
        A, b, c = map(matrix, (self.constrain_matrix, self.constrain_vector, self.objective_vector))

        #Integer solution use GLPK and turn off verbose output
        #solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
        delta, *sol=solvers.lp(c,A,b, solver='glpk')['x']

        return delta, {e:b for e,b in zip(self.W,sol) if b} 
