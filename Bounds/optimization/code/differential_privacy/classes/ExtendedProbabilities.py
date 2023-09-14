import numpy as np
import scipy.integrate
from matplotlib import pyplot as plt

from classes.DiPAPresenter import DiPAPresenter
from classes.ExtendedPrimitives import ExtendedTransition, ORCondition, ExtendedCondition, ANDCondition
from classes.PrimitiveClasses import State
from helpers.constants import Outputs, Guards
from helpers.extended_dipa_constructors import construct_sky_dipa_3
from classes.GDiPAClass import GDiPA
from helpers.laplace import laplace_pdf_without_x, laplace_cdf, integral_of_laplace
from helpers.string_functions import format_output_history


def convert_output_seq_to_state_and_trans_seq(gdipa: GDiPA, output_seq: list[Outputs]) -> tuple[
    list[State], list[ExtendedTransition]]:
    state_sequence = [gdipa.init_state, ]
    trans_sequence = []
    cur_state = gdipa.init_state
    for output in output_seq:
        trans = gdipa.get_transition_for_output(cur_state, output)
        next_state = trans.target_state
        state_sequence.append(next_state)
        trans_sequence.append(trans)
        cur_state = next_state
    return state_sequence, trans_sequence


def compute_trans_probability(dipa_vars: dict[str, float], mu: float, scale_inverse: float,
                              condition: ExtendedCondition) -> float:
    if isinstance(condition, ORCondition):
        return 1.0 - compute_trans_probability(dipa_vars, mu, scale_inverse, condition.negation())
    elif isinstance(condition, ANDCondition):
        prod = 1.0
        for sub_condition in condition.sub_conditions:
            prod *= compute_trans_probability(dipa_vars, mu, scale_inverse, sub_condition)
        return prod
    elif condition.is_leaf:
        guard = condition.payload.guard
        variable = condition.payload.variable
        if guard == Guards.TRUE_CONDITION:
            return 1.0
        elif guard == Guards.FALSE_CONDITION:
            return 0.0
        elif guard == Guards.INSAMPLE_GTE_CONDITION:
            # prob of insample >= var
            return integral_of_laplace(mu, scale_inverse, dipa_vars[variable], np.inf)
        elif guard == Guards.INSAMPLE_LT_CONDITION:
            # prob of insample < var
            return integral_of_laplace(mu, scale_inverse, -np.inf, dipa_vars[variable])
        raise Exception("Unknown guard: " + str(guard))
    else:
        raise Exception("Unknown condition: " + str(condition))


def compute_numerical_probability_extended(dipa: GDiPA,
                                           epsilon: float,
                                           input_history: list[float],
                                           output_history: list[Outputs]) -> float:
    def pr(state_vars: dict[str, float], rho_index: int) -> float:
        if rho_index == n - 1:
            return 1.0

        cur_state = state_seq[rho_index]
        cur_trans = trans_seq[rho_index]

        if not cur_trans.assignment_trans:
            if cur_trans.condition.is_true():  # assume only outputs in the alphabet for now.
                return pr(state_vars, rho_index + 1)
            else:
                ret = compute_trans_probability(
                    state_vars,
                    cur_state.mu + input_history[rho_index],
                    cur_state.d * epsilon,
                    cur_trans.condition
                ) * pr(state_vars, rho_index + 1)
                return ret
        else:
            next_trans = trans_seq[rho_index + 1]
            next_state = state_seq[rho_index + 1]
            if cur_trans.condition.is_true() and next_trans.condition.is_true():
                var1 = cur_trans.assignment_variable
                var2 = next_trans.assignment_variable

                def f(x, y):
                    state_vars[var1] = x
                    state_vars[var2] = y
                    return laplace_pdf_without_x(
                        x - (cur_state.mu + input_history[rho_index]),
                        cur_state.d * epsilon
                    ) * \
                        laplace_pdf_without_x(
                        y - (next_state.mu + input_history[rho_index + 1]),
                        next_state.d * epsilon) * \
                        pr(state_vars, rho_index + 2)

                ret, err = scipy.integrate.dblquad(
                    f,
                    -np.inf,
                    np.inf,
                    -np.inf,
                    np.inf,
                )
                print("err: " + str(err))
                return ret

            elif cur_trans.condition.is_true():
                ret, err = scipy.integrate.quad(
                    lambda x: laplace_pdf_without_x(
                        x - (cur_state.mu + input_history[rho_index]),
                        cur_state.d * epsilon)
                              * pr(state_vars, rho_index + 1),
                    -np.inf,
                    np.inf
                )
                print("err: " + str(err))
                return ret
            else:
                raise NotImplementedError()

    state_seq, trans_seq = convert_output_seq_to_state_and_trans_seq(dipa, output_history)

    n = len(state_seq)
    state_vars = {var: 0.0 for var in dipa.vars}

    return pr(state_vars, rho_index=0)


if __name__ == "__main__":
    dipa = construct_sky_dipa_3()

    # presenter = DiPAPresenter(dipa)
    # presenter.visualize()

    epsilon = 1.0

    ratios = []  # Store ratios for each value of i

    for i in range(20, 41, 1):  # Vary i from 5 to 30 with a step of 5
        X_1 = [0.0, 0.0] + [-j + (-1)**j for j in range(0, i + 1)] + [0.0]
        X_2 = [0.0, 0.0] + [-j for j in range(0, i + 1)] + [0.0]
        output_history = [Outputs.BOT] * 2 + [Outputs.BOT] * i + [Outputs.TOP]

        print("X_1: " + str(X_1))
        print("X_2: " + str(X_2))
        print("output_history: " + format_output_history(output_history))

        p1 = compute_numerical_probability_extended(
            dipa,
            epsilon,
            input_history=X_1,
            output_history=output_history
        )
        p2 = compute_numerical_probability_extended(
            dipa,
            epsilon,
            input_history=X_2,
            output_history=output_history
        )

        print("p1: " + str(p1))
        print("p2: " + str(p2))
        ratio = p1 / p2
        ratios.append(ratio)

    # Generate a colormap for the scatter plot
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(ratios)))

    # Plot the ratios against i as a scatter plot with colorful markers
    plt.scatter(range(20, 41, 1), ratios, c=colors, cmap='viridis')
    plt.xlabel('i')
    plt.ylabel('Ratio')
    plt.title('Ratio vs. i')

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(ratios), vmax=max(ratios)))
    sm.set_array([])
    cbar = plt.colorbar(sm)

    plt.show()


