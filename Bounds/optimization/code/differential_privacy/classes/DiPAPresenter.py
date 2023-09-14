from classes.DiPAClass import DiPA
import graphviz as gv

from classes.DiPAGraph import DiPAGraph
from classes.DiPAValidator import DiPAValidator
from classes.GDiPAClass import GDiPA
from helpers.dipa_constructors import construct_branching_dipa, construct_snake_segment
from helpers.constants import Guards

from IPython.display import Image, display


def format_constraint_class(constraint_class: str) -> str:
    """
    Format a constraint class for printing.
    """

    if constraint_class == 'valid_coupling_constraint_list':
        return 'Constraints for valid couplings'
    elif constraint_class == 'finite_cost_constraint_list':
        return 'Constraints for finite cost'
    elif constraint_class == 'domain_constraint_list':
        return 'Domain constraints'
    elif constraint_class == 'acyclic_segment_graph_constraint_list':
        return 'Constraints for acyclic segment graph'
    else:
        return constraint_class


class DiPAPresenter(object):
    def __init__(self, dipa: DiPA):
        self.dipa = dipa

    def visualize(self):
        dot = gv.Digraph(comment="DiPA")

        # Add states and transitions
        for state_name, state in self.dipa.states.items():
            node_label = f"{state_name}\n--------\n({state.mu}, {state.d})\n({state.mu_prime}, {state.d_prime})"
            node_shape = 'doublecircle' if state.is_terminal_state() else 'circle'
            dot.node(state_name, node_label, shape=node_shape)
            if isinstance(self.dipa, DiPA):
                for cond, transition in state.transitions.items():
                    edge_label = f"{cond.value}\n{transition.get_output().value}, {transition.is_assignment_transition()}"
                    edge_font_colour = 'blue' if cond == Guards.INSAMPLE_LT_CONDITION else (
                        'red' if cond == Guards.INSAMPLE_GTE_CONDITION else 'black')
                    dot.edge(state_name,
                             transition.get_dest_state().get_label(),
                             label=edge_label,
                             color='black',
                             fontcolor=edge_font_colour,
                             penwidth='3.0' if transition.is_assignment_transition() else '1.0'
                             )

        if isinstance(self.dipa, GDiPA):
            for transition in self.dipa.transitions:
                cond = transition.condition
                edge_label = f"{str(cond)}\n{transition.output.value}, {transition.assignment_trans}"
                dot.edge(transition.source_state.label,
                         transition.target_state.label,
                         label=edge_label,
                         color='black',
                         penwidth='3.0' if transition.assignment_trans else '1.0'
                         )

        # Style attributes
        dot.graph_attr['rankdir'] = 'LR'  # Layout from left to right
        dot.node_attr['fontname'] = 'Courier New'  # Set node font to monospace font
        dot.edge_attr['fontname'] = 'Courier New'  # Set node font to monospace font
        dot.node_attr['fontsize'] = '14'  # Nodes are a bit larger
        dot.edge_attr['fontsize'] = '12'  #
        dot.node_attr['filled'] = 'true'
        dot.edge_attr['minlen'] = '2.0'

        # Display the graph

        dot.render('test-output/round-table.gv', view=True)



    def print_dipa(self):
        print("States:")
        for state in sorted(list(self.dipa.states.keys())):
            print(f"\nState {state}")
            print("-" * 10)
            print(f"Transitions: {self.dipa.states[state].transitions}")

    def print_constraints(self) -> None:
        """
        Print the DiPA constraints in a human-readable format.
        """

        validator = DiPAValidator(self.dipa)
        satisfiable: bool = validator.check_satisfiability()

        OKGREEN = '\033[92m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'

        constraint_num = 1

        width = 68

        formatted_sat_string = OKGREEN + "SATISFIABLE" + ENDC \
            if satisfiable \
            else FAIL + "UNSATISFIABLE" + ENDC

        title = f"DiPA CONSTRAINTS: {formatted_sat_string}".center(width + len(OKGREEN) + len(ENDC), ' ')

        print("#" * width)
        print(title)
        print("#" * width)

        for constraint_class, constraint_list in validator.constraint_classes.items():
            print("-" * width)
            print(f"{format_constraint_class(constraint_class)}:\n")

            for constraint in constraint_list:
                print(f"\tConstraint {constraint_num}: {constraint.serialize()}")
                constraint_num += 1

        print("-" * width)
        print("#" * width)

    def print_segments(self) -> None:
        """
        Print the segments of the DiPA in a human-readable format.
        """

        graph = DiPAGraph(self.dipa)
        segments = graph.find_all_segments()

        print("Segments:")

        for i, segment in enumerate(segments):
            print(f"{i}: {segment}")


if __name__ == "__main__":
    dipa = construct_snake_segment(4, 0)
    visualizer = DiPAPresenter(dipa)
    visualizer.visualize()
