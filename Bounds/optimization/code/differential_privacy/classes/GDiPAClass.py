from typing import Set, List, Optional

from classes.DiPAClass import DiPA
from classes.ExtendedPrimitives import ExtendedGuard, ExtendedTransition, ExtendedCondition
from classes.PrimitiveClasses import State, Transition
from helpers.constants import Outputs


class GDiPA(object):

    def __init__(self, init_state: State) -> None:
        """
        Initializes a DiPA with a default start state and no transitions.
        """

        self.init_state = init_state
        self.states: dict[str, State] = {
            init_state.get_label(): init_state
        }
        self.transitions: Set[ExtendedTransition] = set()
        self.vars = set[str]()

    def add_transition(self,
                       source_state: State,
                       target_state: State,
                       guards: ExtendedCondition,
                       assignment_trans: bool,
                       assignment_variable: Optional[str],
                       output: Outputs) -> None:

        assert assignment_trans == (assignment_variable is not None)

        trans = ExtendedTransition(
            source_state,
            target_state,
            guards,
            assignment_trans,
            assignment_variable,
            output
        )
        self.transitions.add(trans)
        if assignment_variable:
            self.vars.add(assignment_variable)

    def add_state(self, state: State):
        self.states[state.get_label()] = state

    def get_init_state(self):
        return self.init_state

    def query_transition(self, state_key_1: str, state_key_2) -> ExtendedTransition:
        """
        Queries the DiPA for a transition between the two given states.
        :param state_key_1: The first state.
        :param state_key_2: The second state.
        :return: The transition between the two states.
        """

        for trans in self.transitions:
            if trans.source_state.label == state_key_1 and trans.target_state.label == state_key_2:
                return trans
        raise Exception("Transition not found")

    def get_transition_for_output(self, cur_state: State, output: Outputs):

        for trans in self.transitions:
            if trans.source_state == cur_state and trans.output == output:
                return trans

        raise ValueError("Transition not found")
