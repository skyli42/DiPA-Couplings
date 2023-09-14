import graphviz as gv
from helpers.dipa_constructors import construct_branching_dipa

dipa = construct_branching_dipa()
dot = gv.Digraph(comment="DiPA")
dot.graph_attr['rankdir'] = 'LR'  # Layout from left to right
dot.graph_attr['fontname'] = 'Courier New'  # Set node font to monospace font
dot.node_attr['shape'] = 'doublecircle'
dot.node_attr['fontsize'] = '14'  # Nodes are a bit larger
dot.edge_attr['fontsize'] = '12'  #
dot.node_attr['filled'] = 'true'
dot.edge_attr['minlen'] = '2.0'

def text(s):
    s = s.replace('<', '$\\lt$')
    s = s.replace('>=', '$\\geq$')
    return '\\text{' + s + '}'


for state_name, state in dipa.states.items():
    node_label = f"{state_name}\n--------\n({state.mu}, {state.d})\n({state.mu_prime}, {state.d_prime})"
    dot.node(state_name, node_label)
    for cond, transition in state.transitions.items():
        edge_label = f"{cond}\n{transition.get_output()}, {transition.is_assignment_transition()}"
        dot.edge(state_name,
                 transition.get_dest_state().get_label(),
                 label=edge_label)

print(dot.source)

# save to test-output/dipa.gv
dot.render(view=True, cleanup=True)
# os.system("dot2tex -tmath --autosize --crop dipa.gv > automata.tex")
# os.system("pdflatex automata.tex > /dev/null")

# dot2tex -tmath --autosize --crop dipa.gv > automata.tex
