from typing import List

from classes.PrimitiveClasses import Transition, State
from helpers.constants import Guards, Outputs


class ExtendedGuard(object):
    def __init__(self, guard: Guards, variable: str = None):
        self.guard = guard

        assert (variable is None) == (guard == Guards.TRUE_CONDITION)

        self.variable = variable

    def negation(self) -> "ExtendedGuard":

        if self.guard == Guards.INSAMPLE_GTE_CONDITION:
            return ExtendedGuard(Guards.INSAMPLE_LT_CONDITION, self.variable)
        elif self.guard == Guards.INSAMPLE_LT_CONDITION:
            return ExtendedGuard(Guards.INSAMPLE_GTE_CONDITION, self.variable)

        raise AssertionError(f"Cannot negate guard {self.guard}")

    def __repr__(self):
        # replace x with variable
        repr_string = self.guard.value.replace("x", self.variable)
        return repr_string

class ExtendedCondition(object):
    def __init__(self, payload: ExtendedGuard = None):
        self.is_leaf = (payload is not None)
        self.payload = payload

    def is_true(self):
        return False

    def negation(self):
        if self.is_leaf:
            return ExtendedCondition(self.payload.negation())
        else:
            raise NotImplementedError()

    def __repr__(self):
        if self.is_leaf:
            return self.payload.__repr__()
        else:
            raise NotImplementedError()


class FalseCondition(ExtendedCondition):
    def __init__(self):
        super().__init__(ExtendedGuard(Guards.FALSE_CONDITION))

    def is_true(self):
        return False

    def negation(self):
        return TrueCondition()

    def __repr__(self):
        return "False"


class TrueCondition(ExtendedCondition):
    def __init__(self):
        super().__init__(ExtendedGuard(Guards.TRUE_CONDITION))

    def is_true(self):
        return True

    def negation(self):
        return FalseCondition()

    def __repr__(self):
        return "True"


class ORCondition(ExtendedCondition):
    def __init__(self, conditions: List[ExtendedCondition]):
        super().__init__()
        self.sub_conditions = conditions

    def negation(self):
        return ANDCondition([c.negation() for c in self.sub_conditions])

    def __repr__(self):
        return " OR ".join([str(c) for c in self.sub_conditions])


class ANDCondition(ExtendedCondition):
    def __init__(self, conditions: List[ExtendedCondition]):
        super().__init__()
        self.sub_conditions = conditions

    def negation(self):
        return ORCondition([c.negation() for c in self.sub_conditions])

    def __repr__(self):
        return " AND ".join([str(c) for c in self.sub_conditions])


class ExtendedTransition(object):
    def __init__(self,
                 source_state: State,
                 target_state: State,
                 condition: ExtendedCondition,
                 assignment_trans: bool,
                 assignment_variable: str,
                 output: Outputs):
        self.source_state = source_state
        self.target_state = target_state
        self.condition = condition
        self.assignment_trans = assignment_trans
        self.assignment_variable = assignment_variable
        self.output = output
