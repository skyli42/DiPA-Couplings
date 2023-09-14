from typing import Any, List

import pysmt.fnode  # type: ignore
from pysmt.shortcuts import Symbol, Int, And, Or, Equals, Implies, Not, LT, is_sat  # type: ignore
from pysmt.typing import INT  # type: ignore

from classes.DiPAClass import DiPA
from classes.DiPAGraph import DiPAGraph, check_adjacency
from helpers.constants import Outputs, Guards

Constraint = pysmt.fnode.FNode


class DiPAValidator(object):
    """
    A class for validating whether DiPAs satisfy differential privacy.
    """
    strategy_sl = Int(0)  # 0: represents the S^L strategy
    strategy_sn = Int(1)  # 1: represents the S^N strategy
    strategy_sg = Int(2)  # 2: represents the S^G strategy

    def __init__(self, dipa: DiPA, print_output=False) -> None:

        self.dipa = dipa
        self.dipa_graph = DiPAGraph(dipa)

        self.constraint_classes: dict[str, List[Constraint]] = {}

    def check_satisfiability(self) -> bool:
        """
        Checks if the constraints are satisfiable.
        :return: True if the constraints are satisfiable, False otherwise.
        """
        return is_sat(self.generate_constraint_formula())

    def generate_constraints_for_valid_couplings(self,
                                                 segments: list[list[str]],
                                                 seg_symbols: list[Symbol]) -> None:
        """
        Generates the constraints for valid couplings. Regrettably, pysmt does not have a
        formula type, so we will have to return Any.
        :param segments: The segments of the DiPA.
        :param seg_symbols: The symbols representing the coupling strategy for each segment.
        :return: A formula encoding the constraints.
        """

        # 1(a)
        for i, segment in enumerate(segments):
            # if the first transition of the segment (trans(seg)) outputs insample, then use S_N.
            first_trans = self.dipa_graph.get_first_transition(segment)
            if first_trans.get_output() == Outputs.INSAMPLE_OUTPUT:
                self.record_constraint(
                    Equals(seg_symbols[i], self.strategy_sn),
                    'valid_coupling_constraint_list'
                )

        # 1(b)
        for i in range(len(segments)):
            for j in range(len(segments)):
                if check_adjacency(segments[i], segments[j]):  # for all (s_i, s_j) such that s_i -> s_j
                    trans_j = self.dipa_graph.get_first_transition(segments[j])
                    if trans_j.guard == Guards.INSAMPLE_LT_CONDITION:
                        self.record_constraint(
                            Implies(Equals(seg_symbols[i], self.strategy_sg),
                                    Equals(seg_symbols[j], self.strategy_sg)),
                            'valid_coupling_constraint_list'
                        )  # If guard(s_j) = insample < x and s_i = S^G, then s_j = S^G

                        self.record_constraint(
                            Implies(Equals(seg_symbols[i], self.strategy_sn),
                                    Not(Equals(seg_symbols[j], self.strategy_sl))),
                            'valid_coupling_constraint_list'
                        )  # If guard(s_j) = insample < x and s_i = S^N, then s_j != S^L

                    elif trans_j.guard == Guards.INSAMPLE_GTE_CONDITION:
                        self.record_constraint(
                            Implies(Equals(seg_symbols[i], self.strategy_sl),
                                    Equals(seg_symbols[j], self.strategy_sl)),
                            'valid_coupling_constraint_list'
                        )  # If guard(s_j) = insample >= x and s_i = S^L, then s_j = S^L

                        self.record_constraint(Implies(
                            Equals(seg_symbols[i], self.strategy_sn), Not(Equals(seg_symbols[j], self.strategy_sg))
                        ), 'valid_coupling_constraint_list'
                        )  # If guard(s_j) = insample >= x and s_i = S^N, then s_j != S^G

        # 1(c): there is no transition trans(ak) in si that is faulty
        for i in range(len(segments)):
            traversable = self.dipa_graph.find_traversable_transitions_in_segment(segments[i])
            for trans in traversable:
                # If s_i contains an insample < x transition that outputs insample, then s_i != S^G
                if trans.guard == Guards.INSAMPLE_LT_CONDITION and trans.output == Outputs.INSAMPLE_OUTPUT:
                    self.record_constraint(
                        Not(Equals(seg_symbols[i], self.strategy_sg)),
                        'valid_coupling_constraint_list'
                    )

                # If s_i contains an insample >= x transition that outputs insample, then s_i != S^L
                if trans.guard == Guards.INSAMPLE_GTE_CONDITION and trans.output == Outputs.INSAMPLE_OUTPUT:
                    self.record_constraint(
                        Not(Equals(seg_symbols[i], self.strategy_sl)),
                        'valid_coupling_constraint_list'
                    )

    def generate_constraints_for_finite_cost(self,
                                             segments: list[list[str]],
                                             seg_symbols: list[Symbol]) -> list[Constraint]:
        # 2 : constraints for finite cost

        for i in range(len(segments)):
            # no cycle has a transition that outputs insample or insample'
            # disclosing cycle
            # TODO

            types = set()
            for cycle in self.dipa_graph.cycles_in_segment(segments[i]):
                cycle = cycle + [cycle[0]]
                types.update(self.dipa_graph.cycle_types(cycle))

            if 'L' in types:
                self.record_constraint(
                    Equals(seg_symbols[i], self.strategy_sl),
                    'finite_cost_constraint_list'
                )
            if 'G' in types:
                self.record_constraint(
                    Equals(seg_symbols[i], self.strategy_sg),
                    'finite_cost_constraint_list'
                )

    def generate_domain_constraints(self, seg_symbols: list[Symbol]) -> None:
        self.constraint_classes['domain_constraint_list'] = [
            Or(Equals(sym, self.strategy_sl),
               Equals(sym, self.strategy_sn),
               Equals(sym, self.strategy_sg))
            for sym in seg_symbols
        ]  # establishes the domains of the symbols, which are the strategies.

    def combine_all_constraints(self) -> pysmt.fnode.FNode:
        """Combines all the constraints into a single formula."""
        all_constraints = []
        for constraint_class in self.constraint_classes.values():
            all_constraints.extend(constraint_class)

        return And(all_constraints)

    def generate_constraint_formula(self) -> pysmt.fnode.FNode:

        segments = self.dipa_graph.find_all_segments()  # build a list of segments.

        seg_symbols = [Symbol(f"S[{i}]", INT) for i in range(len(segments))]  # set up the symbols and domains.
        seg_ordering = [Symbol(f"T[{i}]", INT) for i in range(len(segments))]

        self.generate_domain_constraints(seg_symbols)

        self.generate_constraints_for_valid_couplings(segments, seg_symbols)
        self.generate_constraints_for_finite_cost(segments, seg_symbols)
        self.generate_constraints_for_acyclic_segment_graph(seg_ordering, segments)

        return self.combine_all_constraints()

    def generate_constraints_for_acyclic_segment_graph(self,
                                                       seg_ordering: List[pysmt.shortcuts.Symbol],
                                                       segments: list[list[str]]) -> None:

        for i in range(len(segments)):
            for j in range(len(segments)):
                if check_adjacency(segments[i], segments[j]):
                    self.record_constraint(
                        LT(seg_ordering[i], seg_ordering[j]),
                        'acyclic_segment_graph_constraint_list'
                    )
                    self.record_constraint(
                        LT(seg_ordering[i], seg_ordering[j]),
                        'acyclic_segment_graph_constraint_list'
                    )

    def record_constraint(self,
                          constraint: pysmt.fnode.FNode,
                          constraint_class: str
                          ):
        self.constraint_classes[constraint_class] = self.constraint_classes.get(constraint_class, []) + [constraint]
