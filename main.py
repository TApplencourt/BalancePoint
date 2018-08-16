#!/usr/bin/env python
from balence import Swing
#import networkx as nx
#import matplotlib.pyplot as plt
#from networkx.drawing.nx_agraph import write_dot

import random
import logging

logging.getLogger().setLevel(logging.DEBUG)

def generate_random_dag(n, p,seed=None):
    random_graph = nx.fast_gnp_random_graph(n, p, directed=True, seed=seed)
    G = nx.DiGraph( [(u, v) for (u, v) in random_graph.edges() if u < v])
    # Merge all the leaf
    G.add_edges_from([('root',n) for n,d in G.in_degree() if d==0])
    G.add_edges_from([(n,'leaf') for n,d in G.out_degree() if d==0])
 
    assert (nx.is_directed_acyclic_graph(G))

    random.seed(seed)
    for u,v,d in G.edges(data=True):
            d['weight'] = random.randint(1,20)


    pos=nx.spring_layout(G)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    
    write_dot(G,'66666.dot')
    
    w = { tuple([u,v]): d['weight'] for u,v,d in G.edges(data=True)}

    return w

if __name__ == '__main__':

    # Weighted graph to balence
    #Gao fig 8
    w = {
        ('u1', 'u2'): 1,
        ('u2', 'u4'): 3,
        ('u2', 'u3'): 4,
        ('u3', 'u4'): 4,
        ('u4', 'u5'): 1,
        ('u1', 'u5'): 20,
        ('u3', 'u5'): 5
    }

    ##Appl fig 1
    #w = {
    #    ('u1', 'u2'): 1,
    #    ('u1', 'u3'): 1,
    #    ('u2', 'u6'): 90,
    #    ('u2', 'u4'): 1,
    #    ('u4', 'u6'): 80,
    #    ('u6', 'u8'): 1,
    #    ('u3', 'u7'): 20,
    #    ('u3', 'u5'): 1,
    #    ('u5', 'u7'): 1,
    #    ('u7', 'u8'): 1
    #}

    max_b, max_bb_eval, seed = 410-1, 100, 1
    #w =generate_random_dag(200,0.001, seed)

    s = Swing(w) 
    

    print(s.max_traffic)
    # operation
    #w[('u1','u2')] = 2

    #s2 =Swing(w)
    # optarion

    #print(s.adjacency_list_topdown)
    #print(f'Max buffer alowed: {max_b}')
    #l_buffer_updated, diff_delay = Swing(w).constrained_edge_buffer(max_b, max_bb_eval)  #adjacency_buffered
    #print(f'list buffer: {l_buffer_updated}')
    #print(f'number of buffer used: {sum(l_buffer_updated.values())}')
    #print(f'max diff delay: {diff_delay}')



