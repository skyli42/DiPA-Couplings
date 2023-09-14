from typing import Generator

import networkx as nx

from classes.DiPAClass import DiPA
from helpers.constants import Guards
from classes.PrimitiveClasses import Transition


def all_simple_segment_candidates(G: nx.DiGraph, source: str, target: str) -> Generator[list[str], None, None]:
    """
    Returns all simple paths from source to target in the graph G, yielding
    edges back to source if target==source.
    :param G: A NetworkX directed graph.
    :param source: Specify where the segment must start.
    :param target: Where the segment must end.
    :return: A generator of POSSIBLE segments without checks for number of assignment transitions.
    """

    path = {source: True}
    stack = [iter(G[source])]

    while stack:
        children = stack[-1]
        child = next(children, None)

        if child is None:
            stack.pop()
            path.popitem()
        else:
            if child == target:
                yield list(path.keys()) + [target, ]

            elif child not in path:  # child was not the target, process its children
                path[child] = True
                stack.append(iter(G[child]))


def dfs_edges_modified(graph: nx.DiGraph, source) -> Generator[tuple[str, str], None, None]:
    """
    Perform a depth-first search (DFS) over the nodes of G and yield the edges
    in order, also yielding edges to already visited nodes.
    :param graph: A NetworkX directed graph.
    :param source: Specify starting node for depth-first search and return edges in the component reachable from source.
    :return: A generator of edges in the depth-first-search tree.
    """

    edges = [(None, source)]
    visited = set()

    while edges:
        u, v = edges.pop()

        if u is not None:
            yield u, v

        if v in visited:
            continue

        visited.add(v)

        for w in graph.neighbors(v):
            edges.append((v, w))


def edges_in_cycle(cycle: list['str']) -> list[tuple[str, str]]:
    """
    Returns a list of edges in the cycle.
    :param cycle: A list of nodes in the cycle in the form [a0, a1, ..., an, a0], with each node appearing exactly once except a0.
    :return: An ordered list of edges in the cycle in the form [(a0, a1), (a1, a2), ..., (an, a0)].

    Precondition: cycle is a nonempty list of node labels.
    """
    assert cycle[0] == cycle[-1] and len(cycle) >= 2
    # cycle = cycle + [cycle[0]]
    edges = list()
    for i in range(len(cycle) - 1):
        edges.append((cycle[i], cycle[i + 1]))
    return edges


def check_adjacency(segment1: list[str], segment2: list[str]) -> bool:
    """
    Returns True if segment1 -> segment2, that is, if segment2 follows segment1 in the DiPA graph.
    :param segment1: A nonempty list of node labels a_0 -> ... -> a_n in the first segment.
    :param segment2: A nonempty list of node labels b_0 -> ... -> b_m in the second segment.
    :return: True if segment1 -> segment2, False otherwise. This is equivalent to checking if a_n == b_0.
    """

    return segment1[-1] == segment2[0]


class DiPAGraph(object):

    def __init__(self, dipa: DiPA):
        self.dipa = dipa
        self.dipa_graph_nx = self.generate_dipa_graph_nx()

    def generate_dipa_graph_nx(self) -> nx.DiGraph:

        dipa_graph = nx.DiGraph()

        dipa_graph.add_nodes_from(self.dipa.states.keys())

        elist = []

        for state_name in self.dipa.states:
            for (guard, trans) in self.dipa.states[state_name].transitions.items():
                dest_name = trans.get_dest_state().get_label()
                elist.append((state_name, dest_name))

        dipa_graph.add_edges_from(elist)

        return dipa_graph

    def find_traversable_edges_in_segment(self, segment: list[str]) -> list[tuple[str, str]]:

        def trans_appears_in_segment(segment: list[str], transition: tuple[str, str]) -> bool:
            if transition[0] not in segment or transition[1] not in segment:
                return False

            for i in range(len(segment) - 1):
                if segment[i] == transition[0] and segment[i + 1] == transition[1]:
                    return True

            return False

        a_0 = segment[0]
        a_n = segment[-1]

        traversable_edges_from_a0: set[tuple[str, str]] = set(dfs_edges_modified(self.dipa_graph_nx, source=a_0))
        traversable_edges_from_an: set[tuple[str, str]] = set(edge[::-1] for edge in dfs_edges_modified(
            self.dipa_graph_nx.reverse(), source=a_n))

        traversable_edges = set.intersection(traversable_edges_from_a0, traversable_edges_from_an)
        filtered_transitions = set()

        for edge in traversable_edges:
            if trans_appears_in_segment(segment, edge):  # transition appears in the segment
                filtered_transitions.add(edge)
            else:
                trans = self.dipa.query_transition(edge[0], edge[1])
                if not trans.is_assignment_transition():  # if the transition is NOT an assignment transition. TODO: verify this!
                    filtered_transitions.add(edge)

        return list(filtered_transitions)

    def find_traversable_transitions_in_segment(self, segment: list[str]) -> set[Transition]:
        traversable_edges = self.find_traversable_edges_in_segment(segment)
        return self.convert_edge_collection_to_transition_list(traversable_edges)

    def convert_segment_to_transition_list(self, edges: list[str]) -> list[Transition]:
        t_list = []
        for i in range(len(edges) - 1):
            trans = self.dipa.query_transition(edges[i], edges[i + 1])
            t_list.append(trans)
        return t_list

    def convert_edge_collection_to_transition_list(self, edges: list[tuple[str, str]]) -> list[Transition]:
        t_list = list()
        for edge in edges:
            trans = self.dipa.query_transition(edge[0], edge[1])
            t_list.append(trans)
        return t_list

    def cycle_types(self, cycle: list['str']) -> set[str]:
        types = set()
        for edge in edges_in_cycle(cycle):
            trans = self.dipa.query_transition(edge[0], edge[1])

            if trans.guard == Guards.INSAMPLE_LT_CONDITION:
                types.add("L")
            elif trans.guard == Guards.INSAMPLE_GTE_CONDITION:
                types.add("G")
        return types

    def find_all_segments(self) -> list[list[str]]:
        """
        Finds all segments in the DiPA.

        We do this by looking at all simple paths from an assignment state
        to another assignment state or a terminal state, and choosing all
        paths that have exactly one assignment transition.

        :return: A list of segments.
        """

        a_states = [key for (key, state) in self.dipa.states.items() if state.has_assignment_transition()]
        t_states = [key for (key, state) in self.dipa.states.items() if state.is_terminal_state()]

        segments: list[list[str]] = []

        for a1 in a_states:
            for a2 in a_states + t_states:
                for simple_path in all_simple_segment_candidates(self.dipa_graph_nx, a1, a2):
                    transitions_on_segment = self.convert_segment_to_transition_list(simple_path)
                    num_assignments_on_segment = len(
                        [t for t in transitions_on_segment if t.is_assignment_transition()])
                    if num_assignments_on_segment == 1:

                        first_trans = self.dipa.query_transition(simple_path[0], simple_path[1])

                        if not first_trans.is_assignment_transition():
                            continue

                        segments.append(simple_path)

        return segments

    def get_segment_subgraph(self, segment: list[str]) -> nx.DiGraph:
        """
        Returns a subgraph of the DiPA graph that contains only the traversable
         edges in the segment.
        :param segment: A segment of node labels.
        :return: the subgraph.
        """

        relevant_edges = self.find_traversable_edges_in_segment(segment)
        subgraph = self.dipa_graph_nx.edge_subgraph(relevant_edges)
        return subgraph

    def cycles_in_segment(self, segment: list[str]) -> Generator[list[str], None, None]:
        """
        Returns a generator of all cycles in the segment.
        Each cycle is of the form [a_0, a_1, ..., a_n], and so contains each node label exactly once.
        :param segment: A segment of node labels.
        :return: A generator of cycles.
        """

        subgraph = self.get_segment_subgraph(segment)
        return nx.simple_cycles(subgraph)

    def convert_cycle_to_transition_list(self, cycle) -> list[Transition]:
        """
        Converts a cycle to a list of transitions.
        :param cycle: A cycle in the form [a_0, a_1, ..., a_n], with each node appearing exactly once.
        :return:
        """

        cycle = cycle + [cycle[0], ]
        return self.convert_segment_to_transition_list(cycle)

    def cycle_transitions_in_segment(self, segment: list[str]) -> list[Transition]:
        """
        Returns a list of all transitions in a cycle in the segment.
        :param segment:  A segment of node labels.
        :return: A list of transitions.
        """

        cycles = self.cycles_in_segment(segment)
        cycle_transitions = []
        for cycle in cycles:
            cycle_transitions.extend(self.convert_cycle_to_transition_list(cycle))
        return cycle_transitions

    def get_first_transition(self, segment: list[str]) -> Transition:
        """
        Returns the first transition in the segment.
        :param segment: A segment of node labels.
        :return: The first transition in the segment.
        """
        return self.dipa.query_transition(segment[0], segment[1])

    def get_following_pairs(self, segments: list[str]) -> list[tuple[str, str]]:
        """
        Returns a list of pairs of transitions that follow each other in the segment.
        :param segment: A segment of node labels.
        :return: A list of pairs of transitions.
        """
        pairs = []
        for i, s1 in enumerate(segments):
            for j, s2 in enumerate(segments):
                if s1[-1] == s2[0]:
                    pairs.append((i, j))
        return pairs

    def find_all_segment_sequences(self, segments):
        """
        Generates all possible sequences of segments.
        :param segments: A list of segments.
        :return: A generator of lists of segments.
        """
        segment_following_graph = nx.DiGraph()
        segment_following_graph.add_nodes_from(range(len(segments)))
        segment_following_graph.add_edges_from(self.get_following_pairs(segments))

        leaves = [node for node in segment_following_graph.nodes() if segment_following_graph.out_degree(node) == 0]

        # The segment following graph is a DAG. Return all possible paths from node 0 to leaves.
        return nx.all_simple_paths(segment_following_graph, 0, leaves)