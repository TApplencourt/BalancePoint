import logging 
import time
import numpy as np

from itertools import chain
from BalencePoint.swing import Swing
from cvxopt import matrix, solvers, glpk
from . import lazy_property

class Gao():

    """
     Gao' paper: https://doi.org/10.1016/0743-7315(89)90041-5
    
     Maximize \sum_i u_i (outdegree(i) - indegree(i))
     Subject to
           u_i - u_j \leq -w_{ij} for all (i,j) \subseteq E
           u_n - u_1 = w_{st}
           u_i >= 0
    """

    def __init__(self,s):
        self.swing = s

    @property
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

    @property
    def constrain_vector(self):
        b = -np.array(list(self.swing.weigh.values()), dtype=float)
        b_pos = np.zeros(len(self.swing.l_node))  # Positif contrains
        return np.concatenate((b, b_pos))

    @property
    def objective_vector(self):
        # CVXopt can only minimazise so the multiply the objective vector by -1
        # Float is required by CVXopt
        return np.array([-1 * self.swing.delta_degree[i] for i in self.swing.order], dtype=float)

    @lazy_property
    def optimal_firing(self): 
        '''
        Gao's algorithm, using Linear Programming formulation
        Compute the optimal firing of each node for the now balenced graph
        '''

        # We add the constrain that each objectif need to me postiif
        A, b, c = map(matrix, (self.constrain_matrix, self.constrain_vector, self.objective_vector))

        #Integer solution use GLPK and turn off verbose output
        solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
        sol = solvers.lp(c, A, b, solver='glpk')

        x, x_int = sol['x'], [int(round(i)) for i in sol['x']]
        assert all(i == f for i, f in zip(x_int, x)), 'Some none integer buffers where found'
        return dict(zip(self.swing.order, x_int))  # Assume ordered dict

    @lazy_property
    def optimal_buffer(self):
        '''
        Needed buffer to optimally balance the graph
        '''
        f, w = self.optimal_firing, self.swing.weigh
        d =  {(i, o): (f[o] - f[i]) - w for (i, o), w in w.items() if (f[o] - f[i]) != w}
        logging.info(f'{len(d)} distinct buffers ({sum(d.values())} values in total) are needed to optimaly balence the graph')
        return d



class Apple():

    """
    param E; param P;

    set EDGE := {0..E};
    set PATH := {0..P};

    param m{EDGE,PATH};
    param w{0..E};
    set CP;

    var b{i in EDGE} integer >= 0;

    var eta integer >=0 ;
    minimize obj: eta;

    subject to cp{p in PATH}:  sum{j in CP} w[j] >= sum{i in EDGE} ( ( b[i] + w[i] ) * m[i,p]);
    subject to delta{p in PATH}:  eta >=  sum{j in CP} w[j]  - sum{i in EDGE} ( ( b[i] + w[i] ) * m[i,p]);

    subject to budget: sum{i in EDGE} b[i] <= 20000;
    end;
    """

    def __init__(self,s, budget=None, l_edge=None):
        self.swing = s

        w = self.swing.weigh
        if l_edge is None:
            self.W = w
        else:
            self.W = {e:w[e] for e in l_edge}

        self.edges = len(self.W)
        self.budget = budget

    @property
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


        A =  np.concatenate((e,d,-A_pos))
        if not self.budget:
            return A
        else:
            b  = np.concatenate( ([0],np.ones(self.edges))).reshape(-1,1)
            return np.concatenate((e,d,-A_pos,b.T))

    @property
    def constrain_vector(self):
        weight_path = np.array(list(self.swing.non_critical_paths.values()))
        # Sum weight_path
        b = -next(iter(self.swing.critical_paths.values())) + weight_path
        # postof constraint
        b_pos = np.zeros(self.edges+1)
        b = np.concatenate((b,-b,b_pos))

        if not self.budget:
            return b
        else:
            return np.concatenate( (b, [self.budget]) ) 

    @property
    def objective_vector(self):
        return np.concatenate( ([1],np.zeros(self.edges)))

    @lazy_property
    def optimal_buffer(self):
        A, b, c = map(matrix, (self.constrain_matrix, self.constrain_vector, self.objective_vector))

        # Use the low level interface that allow use to define some variables as Integer
        options = {'msg_lev': 'GLP_MSG_OFF'}
        I = set(range(1,self.edges+1)) if self.budget else set()
        status, x = glpk.ilp(c,A,b, I = I,options=options)
        assert (status == 'optimal')
        delta, *sol = x

        d = {e:b for e,b in zip(self.W,map(int,sol)) if b}
        logging.info(f'{len(d)} distinct buffers ({sum(d.values())} values in total) are needed to balence the graph')

        return int(delta), d


class Solver():

    def __init__(self, w, budget=None, method='Gao+Apple'):
        self.s = Swing(w)
        assert (method in ('Gao','Apple','Gao+Apple')),f'Method {method} not alowed'
        self.method = method
        self.budget = budget

    @property
    def optimal_buffer(self):

        l_edge = None
        if self.budget is None and self.method in ['Gao','Gao+Apple']:
            start = time.time()
            glp = Gao(self.s)
            d = glp.optimal_buffer
            end = time.time()
            logging.info(f"Gao opt: {end - start:.3f} seconds")

            if not self.budget or sum(d.values()) <= self.budget:
                return 0, d
            else:
                l_edge = set(d.keys())

        if self.method in ['Apple', 'Gao+Apple']:
            start = time.time()
            glp = Apple(self.s,self.budget, l_edge)
            delta, d = glp.optimal_buffer
            end = time.time()
            logging.info(f"Apple opt: {end - start:.3f} seconds")
            return delta, d
 
