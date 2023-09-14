from classes.GDiPAClass import GDiPA
from classes.ExtendedPrimitives import ExtendedTransition, ExtendedGuard, ExtendedCondition, TrueCondition, \
    ORCondition, ANDCondition
from classes.PrimitiveClasses import State

from helpers.constants import Outputs, Guards


def construct_sky_dipa_3() -> GDiPA:
    init_state = State("q0", 1, 0, 1, 0)
    q1 = State("q1", 1, 0, 1, 0)
    q2 = State("q2", 1, 0, 1, 0)
    q3 = State("q3", 1, 0, 1, 0)
    q4 = State("q4", 1, 0, 1, 0)

    dipa = GDiPA(init_state)
    dipa.add_state(q1)
    dipa.add_state(q2)
    dipa.add_state(q3)
    dipa.add_state(q4)

    # q0 -> q1, true, BOT, assign x
    dipa.add_transition(
        init_state,
        q1,
        TrueCondition(),
        True,
        "x",
        Outputs.BOT,
    )

    # q1 -> q2, true, BOT, assign y
    dipa.add_transition(
        q1,
        q2,
        TrueCondition(),
        True,
        "y",
        Outputs.BOT,
    )

    # q2 -> q3, in >= x AND in < y, BOT, no assign
    dipa.add_transition(
        q2,
        q3,
        ANDCondition([
            ExtendedCondition(ExtendedGuard(Guards.INSAMPLE_GTE_CONDITION, "x")),
            ExtendedCondition(ExtendedGuard(Guards.INSAMPLE_LT_CONDITION, "y"))
        ]),
        False,
        None,
        Outputs.BOT,
    )

    # q3 -> q2, in < x or in >= y, BOT, no assign
    dipa.add_transition(
        q3,
        q2,
        ORCondition([
            ExtendedCondition(ExtendedGuard(Guards.INSAMPLE_LT_CONDITION, "x")),
            ExtendedCondition(ExtendedGuard(Guards.INSAMPLE_GTE_CONDITION, "y"))
        ]),
        False,
        None,
        Outputs.BOT,
    )

    # q3 -> q4, in >= x AND in < y, TOP, no assign
    dipa.add_transition(
        q3,
        q4,
        ANDCondition([
            ExtendedCondition(ExtendedGuard(Guards.INSAMPLE_GTE_CONDITION, "x")),
            ExtendedCondition(ExtendedGuard(Guards.INSAMPLE_LT_CONDITION, "y"))
        ]),
        False,
        None,
        Outputs.TOP,
    )

    # q2 -> q4, in < x or in >= y, TOP, no assign
    dipa.add_transition(
        q2,
        q4,
        ORCondition([
            ExtendedCondition(ExtendedGuard(Guards.INSAMPLE_LT_CONDITION, "x")),
            ExtendedCondition(ExtendedGuard(Guards.INSAMPLE_GTE_CONDITION, "y"))
        ]),
        False,
        None,
        Outputs.TOP,
    )

    return dipa


if __name__ == "__main__":
    dipa = construct_sky_dipa_3()
