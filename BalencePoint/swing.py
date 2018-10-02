import logging 

from collections import defaultdict
from itertools import chain,tee
from functools import lru_cache
from . import cached_property

# Python Typing
from typing import List, Dict, Tuple, Set, NewType
Node = NewType('Node', str)
Edge = Tuple[Node, Node]
Path = List[Edge]
Weigh = int

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Swing(object):

    def __init__(self, weigh):
        self.weigh = weigh  # Top-> bottom by default.
        logging.info(f'{len(weigh)} edges')

    @cached_property
    def adjacency_list_topdown(self) -> Dict[Node, List[Node]]:
        d = defaultdict(list)
        print(f"in alt {id(self.weigh)}")
        for i, o in self.weigh:
            d[i].append(o)
        return dict(d)

    @cached_property
    def adjacency_list_bottomup(self) -> Dict[Node, List[Node]]:
        d = defaultdict(list)
        for k, lv in self.adjacency_list_topdown.items():
            for v in lv:
                d[v].append(k)
        return dict(d)

    @cached_property
    def l_node(self) -> Set[Node]:
        ''' list of nodes'''
        n = self.adjacency_list_topdown.keys() | self.adjacency_list_bottomup.keys()
        logging.info(f'{len(n)} nodes')
        return n

    @cached_property
    def order(self) -> Dict[Node, int]:
        '''Arbitrary nodes labeling'''
        return {k: i for i, k in enumerate(self.l_node)}

    @cached_property
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

    @cached_property
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

    @cached_property
    def path_edges(self) -> List[Path]:
        return [ list(pairwise(p)) for p in self.path_node(self.leaf)]

    @cached_property
    def path(self) -> Dict[Tuple[Edge], Weigh]:
        '''
        Compute for all the path ( of the form (e1->e2),(e2->e3) )
        and the sum of the weigh from the leaf to the read
        '''
        d = dict()

        for p in self.path_node(self.leaf):
            path_edge = tuple(pairwise(p))
            d[path_edge] = sum(map(self.weigh.get, path_edge))

        logging.info(f'{len(d)} paths')
        return d

    @cached_property
    def critical_paths(self) -> Dict[Tuple[Edge], Weigh]:
        c =  self.critical_weigh
        d = {p: w for p, w in self.path.items() if w == c}
        logging.info(f'{len(d)} critical paths')
        return d

    @cached_property
    def non_critical_paths(self) -> Dict[Tuple[Edge], Weigh]:
        c = self.critical_weigh
        d = {p: w for p, w in self.path.items() if w != c}
        logging.info(f'{len(d)} non critical paths')
        return d

    @cached_property
    def critical_weigh(self) -> Weigh:
        cw = max(self.path.values())
        logging.info(f'critical path weigh {cw}')
        return cw

    @cached_property
    def is_balenced(self):
        return len(self.critical_paths) == len(self.path)
