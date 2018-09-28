#!/usr/bin/env python
from BalencePoint.swing import Swing
from BalencePoint.lp import Apple, Gao

import logging

logging.getLogger().setLevel(logging.DEBUG)

def generate_random_dag(n, p,seed=None, draw=False):

    import random
    import networkx as nx
    from networkx.drawing.nx_agraph import write_dot

    random_graph = nx.fast_gnp_random_graph(n, p, directed=True, seed=seed)
    G = nx.DiGraph( [(u, v) for (u, v) in random_graph.edges() if u < v])
    # Merge all the leaf
    G.add_edges_from([('root',n) for n,d in G.in_degree() if d==0])
    G.add_edges_from([(n,'leaf') for n,d in G.out_degree() if d==0])
 
    assert (nx.is_directed_acyclic_graph(G))

    random.seed(seed)
    for u,v,d in G.edges(data=True):
            d['weight'] = random.randint(1,20)


    if draw:
        from networkx.drawing.nx_agraph import write_dot

        pos=nx.spring_layout(G)
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        write_dot(G,'66666.dot')
    
    w = { tuple([u,v]): d['weight'] for u,v,d in G.edges(data=True)}

    return w

if __name__ == '__main__':

    #Gao fig 8
    '''
    w = {
        ('u1', 'u2'): 1,
        ('u2', 'u4'): 3,
        ('u2', 'u3'): 4,
        ('u3', 'u4'): 4,
        ('u4', 'u5'): 1,
        ('u1', 'u5'): 20,
        ('u3', 'u5'): 5
    }
    '''

    ##Appl fig 1
    w = {
        ('u1', 'u2'): 1,
        ('u1', 'u3'): 1,
        ('u2', 'u6'): 90,
        ('u2', 'u4'): 1,
        ('u4', 'u6'): 80,
        ('u6', 'u8'): 1,
        ('u3', 'u7'): 20,
        ('u3', 'u5'): 1,
        ('u5', 'u7'): 1,
        ('u7', 'u8'): 1
    }
    seed = 1
    import time

    start = time.time()
    w =generate_random_dag(1250,0.001, seed)
    end = time.time()
    print (f"Gen graph: {end - start}")



    s = Swing(w) 

    start = time.time()
    glp = Gao(s)
    glp.optimal_buffer
    end = time.time()
    print (f"Gao opt: {end - start}")
 
    start = time.time()
    glp = Apple(s)
    glp.optimal_buffer
    end = time.time()
    print (f"Apple opt: {end - start}")
