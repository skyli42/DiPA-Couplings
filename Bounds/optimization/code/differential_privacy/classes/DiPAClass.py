from typing import Optional, Literal, Collection, List, Set

from classes.PrimitiveClasses import State, Transition
from helpers.constants import Outputs, Guards
from helpers.string_functions import format_output_history
from classes.Exceptions import TransitionNotFoundException


class DiPA(object):

    def __init__(self, init_state: State) -> None:
        """
        Initializes a DiPA with a default start state and no transitions.
        """

        self.init_state = init_state
        self.states: dict[str, State] = {
            init_state.get_label(): init_state
        }
        self.transitions: Set[Transition] = set()

    def add_state(self, new_state: State) -> None:
        """
        Adds a new state to the DiPA.
        :param new_state: The new state to add.
        :return: None
        """
        self.states[new_state.get_label()] = new_state

    def add_transition_from_states(self,
                                   source_state: State,
                                   dest_state: State,
                                   guard: Guards,
                                   output: Outputs,
                                   assignment_trans: bool
                                   ) -> None:
        """
        Adds a new transition to the DiPA given the source and destination states.
        :param source_state: The source state of the transition.
        :param dest_state: The destination state of the transition.
        :param guard: The guard of the transition.
        :param output: The output (either BOT or TOP) of the transition.
        :param assignment_trans: Whether the transition is an assignment transition.
        :return:
        """
        trans = source_state.add_transition(dest_state, guard, output, assignment_trans)
        self.transitions.add(trans)


    def add_transition(self,
                       source_state_key: str,
                       dest_state_key: str,
                       guard: Guards,
                       output: Outputs,
                       assignment_trans: bool
                       ) -> None:
        """
        Adds a new transition to the DiPA given the source and destination state keys.
        :param source_state_key:
        :param dest_state_key:
        :param guard:
        :param output:
        :param assignment_trans:
        :return:
        """
        source_state = self.states[source_state_key]
        dest_state = self.states[dest_state_key]

        self.add_transition_from_states(source_state, dest_state, guard, output, assignment_trans)


    def get_transition_for_output(self, source_state: State, output: Outputs) -> Optional[Transition]:
        """
        Returns the transition from the source state with the given output, if it exists.
        :param source_state: The source state.
        :param output: The output.
        :return: The transition, or None if it does not exist.
        """
        for transition in source_state.transitions.values():
            if transition.get_output() == output:
                return transition
        return None

    def validate(self) -> bool:
        """
        Validates that the DiPA is valid.
        :return: True if the DiPA is valid, False otherwise.
        """
        # TODO
        return True

    def get_init_state(self):
        return self.init_state

    def query_transition(self, state_key_1: str, state_key_2) -> Transition:
        """
        Returns the transition object from state_key_1 to state_key_2, if it exists.
        :param state_key_1: The source state key.
        :param state_key_2: The destination state key.
        :return: The transition object, or None if it does not exist.
        """

        if state_key_1 not in self.states:
            raise TransitionNotFoundException(f"State {state_key_1} not found in DiPA.")

        state_1 = self.states[state_key_1]

        for transition in state_1.transitions.values():
            if transition.get_dest_state().get_label() == state_key_2:
                return transition

        raise TransitionNotFoundException(f"Transition from {state_key_1} to {state_key_2} not found.")

    def add_transition_to_dipa(self, source_state_key: str,
                       dest_dipa: "DiPA",
                       guard: Guards,
                       output: Outputs,
                       assignment_trans: bool
                       ) -> None:
        """
        Adds a new transition to the destination DiPA given the source.
        :param source_state_key:
        :param dest_state_key:
        :param guard:
        :param output:
        :param assignment_trans:
        :return:
        """

        for name, state in dest_dipa.states.items():
            i = 2
            while name in self.states:
                state.label = f"{name}_{i}"
                i += 1
            self.add_state(state)

        for transition in dest_dipa.transitions:
            self.transitions.add(transition)

        self.add_transition(
            source_state_key,
            dest_dipa.init_state.get_label(),
            guard,
            output,
            assignment_trans
        )



class DiPATraverser(object):
    def __init__(self, dipa: DiPA, epsilon: float) -> None:
        self.dipa = dipa
        self.epsilon = epsilon

        # Current state variables
        self.current_state = self.dipa.get_init_state()
        self.x = 0

        self.insample = 0
        self.insample_prime = 0

        # Output tracking variable
        self.input_history: List[float] = []
        self.output_history: List[Outputs] = []

    def input(self, input_val: float) -> int:
        """
        Process the input of input_val into the DiPA.

        returns 0 if the query was not processed, 1 if the query was processed.
        """

        if self.current_state.is_terminal_state():
            return 0

        z1, z2 = self.current_state.generate_lap_noise(
            self.epsilon)  # generate the noise for insample and insample_prime

        self.insample = input_val + z1
        self.insample_prime = input_val + z2

        trans = self.current_state.get_next_transition(self.insample >= self.x)

        self.current_state = trans.get_dest_state()
        self.output_history.append(trans.get_output())

        if trans.is_assignment_transition():
            self.x = self.insample

        return 1

    def feed_sequence(self, sequence: Collection[float]) -> int:
        """
        Process the input sequence into the DiPA.
        :param sequence: The input sequence of floats to process.
        :return: The index of the first input that was not processed,
        or len(sequence) if all inputs were processed.
        """
        if len(sequence) == 0:
            return 0
        return sum(self.input(input_val) for input_val in sequence)

    def get_output_string(self):
        return format_output_history(self.output_history)

    def reset_state_variables(self):
        # Current state variables
        self.current_state = self.dipa.get_init_state()
        self.x = 0

        self.insample = 0
        self.insample_prime = 0

        self.input_history = []
        self.output_history = []

    def get_output_history(self) -> List[Outputs]:
        return self.output_history