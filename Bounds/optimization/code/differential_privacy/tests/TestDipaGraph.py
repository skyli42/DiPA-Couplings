import networkx as nx
from classes.DiPAGraph import all_simple_segment_candidates
import matplotlib.pyplot as plt

def test_all_simple_segment_candidates_1():
    """
    Test the all_simple_segment_candidates function on a graph
    with a self-edge from source to target==source.
    """

    G = nx.DiGraph()
    G.add_edge('a', 'b')
    G.add_edge('b', 'c')
    G.add_edge('c', 'a')
    G.add_edge('c', 'd')
    G.add_edge('d', 'c')
    G.add_edge('b', 'd')
    G.add_edge('a', 'a')
    G.add_edge('b', 'a')

    nx.draw(G, with_labels=True)
    plt.show()

    assert set([tuple(seg) for seg in all_simple_segment_candidates(G, 'a', 'a')]) == {
        ('a', 'a'),
        ('a', 'b', 'a'),
        ('a', 'b', 'c', 'a'),
        ('a', 'b', 'd', 'c', 'a'),
    }


def test_all_simple_segment_candidates_2():
    """
    Test that the all_simple_segment_candidates function works as intended
    on a graph with target==source by comparing with the output of nx.all_simple_paths.
    'k' here is a bottleneck node through which all a->a paths must pass.
    """

    G = nx.DiGraph()
    G.add_edge('a', 'b')
    G.add_edge('b', 'c')
    G.add_edge('b', 'd')
    G.add_edge('c', 'd')
    G.add_edge('c', 'f')
    G.add_edge('d', 'g')
    G.add_edge('d', 'h')
    G.add_edge('e', 'c')
    G.add_edge('f', 'i')
    G.add_edge('f', 'k')
    G.add_edge('g', 'j')
    G.add_edge('h', 'j')
    G.add_edge('i', 'k')
    G.add_edge('i', 'e')
    G.add_edge('j', 'k')
    G.add_edge('j', 'i')
    G.add_edge('l', 'i')
    G.add_edge('l', 'j')
    G.add_edge('l', 'k')

    K = G.copy()
    K.add_edge('k', 'a')

    assert set(tuple(path) for path in all_simple_segment_candidates(K, 'a', 'a')) == set(
        tuple(path) + ('a', ) for path in nx.all_simple_paths(G, 'a', 'k')
    )



def test_all_simple_edge_paths():

    G = nx.DiGraph()
    G.add_edge('a', 'b')
    G.add_edge('b', 'c')
    G.add_edge('c', 'a')
    G.add_edge('a', 'c')

    assert set(tuple(path) for path in all_simple_segment_candidates(G, 'a', 'a')) == {
        ('a', 'b', 'c', 'a'),
        ('a', 'c', 'a'),
    }
