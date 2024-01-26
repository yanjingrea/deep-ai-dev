import numpy as np
from numpy.random import default_rng

from dataclasses import dataclass
from typing import Any, List


# -----------------------------------------------------------------------------------------------------------------------

# a generic random state generator constructor
# (emulates Levi flights)
#
# params:
#
#   gen_random_state:       function (T) -> random_state
#   gen_random_neighbor:    function (current_state, T) -> neighbor_state
#
#   p:                      frequency of 'long' jumps
#                           (when the generator returns a completely arbitrary state
#                           as opposed to a neighbor of the last generated state)
#
#   random_generator:       [optional]
#
# returns:
#
#   function (current_state, T) -> new_state
#
# note:
#   `T` parameter used above is 'temperature' (see description of `annealing` below)
#
def StateGenerator(
    gen_random_state,
    gen_random_neighbor,
    p=0.1,
    random_generator=default_rng()
):

    def f(current_state=None, T=0):

        if current_state is None or random_generator.random() > 1 - p:
            # 'long' jump
            return gen_random_state(T)

        else:
            # random walk around `current_path`
            return gen_random_neighbor(current_state, T)

    return f


# ------------------------------------------------------------------

# argument type for the `evaluation_predicate`
# (see description of `annealing` below)
#
@dataclass
class EvaluationParams:

    @dataclass
    class SPair:
        state: Any
        value: float

    # ------------------------------------------------------------------

    minimum: SPair
    current: SPair
    transitions: List[SPair]
    n_iter: int
    n_calls: int


# a generic evaluation predicate
# (implements a transition monitoring heuristic for early stopping)
#
def EvaluationPredicate(
    max_calls=1000,
    max_iter=2000,
    window_length=5,
    threshold=5e-3
):

    ratios = lambda a: a[1:] / a[:-1]

    def f(p: EvaluationParams):

        if p.n_iter < max_iter and p.n_calls < max_calls:

            if len(p.transitions) > window_length:

                transition_values = np.fromiter(
                    (abs(q.value) for q in p.transitions),
                    dtype=float
                )

                r = np.abs(1 - ratios(transition_values[-window_length:]))

                return (r >= threshold).any()
            else:
                return True
        else:
            return False

    return f


# ------------------------------------------------------------------

# simulated annealing optimization (a variant of random search)
#
def annealing(
    f,
    state_generator,

    initial_state=None,
    temperature=lambda n: 1 / (1 + n / 10),
    p_transition=lambda current_val, new_val, T: 1 if new_val < current_val else np.exp((current_val - new_val) / T),
    evaluation_predicate=lambda p: p.n_iter < 1000 and p.n_calls < 1000,
    random_generator=default_rng()
):
    """
    input arguments:

       f:   					objective to minimize;
                                should be a real-valued function of one argument `state`
                                of any orderable type (such that it could be used as a dictionary key)

       state_generator:			a function with the following signature:

                                  (current_state: Any, temperature: float) -> new_state: Any

                                this function is used to obtain each new feasible random state
                                given the `current_state` and `temperature`

       initial_state:   		if `None` then `state_generator` is called to obtain the initial state

       temperature:				a function with the following signature:

                                  (n: int) -> float

                                where `n` is the current iteration number;

                                temperature value controls the transition probability (see below)

       p_transition:			a function with the following signature:

                                  (current_val: Any, new_val: Any, temperature: float) -> float

                                defines the probability of transition from current state to a new state
                                based on the objective function values at these states
                                (`current_val` and `new_val` respectively);

                                the lower the `temperature` (~ the more iterations have passed)
                                the lower the probability of transitioning to the state
                                with a greater value should be;

       evaluation_predicate: 	a function with the following signature:

                                  (p: EvaluationParams) -> bool

                                should return True if evaluation should be continued

       random_generator:		numpy random generator
                                (an explicit value may be provided for reproducibility)

    return values:

       minimum_state:			a state with a minimal value across all the evaluated states

       evaluated_states:		a dictionary (`state` -> `value`) of all the evaluated states

       transitions:				a list of all the transitions (including the initial state)
                                as a list of `SPair`-s (see `EvaluationParams` class above)
    """

    SPair = EvaluationParams.SPair

    evaluated_states = {}
    transitions = []

    initial_state = initial_state if initial_state is not None else state_generator(None, temperature(0))

    current_state = initial_state
    current_val = f(initial_state)

    evaluated_states[current_state] = current_val

    transitions.append(SPair(current_state, current_val))

    minimum_state = current_state
    minimum_val = current_val

    n_iter = 0
    n_calls = 1

    while True:

        T = temperature(n_iter)
        state = state_generator(current_state, T)

        if state is None:
            continue

        if state not in evaluated_states:
            evaluated_states[state] = f(state)
            n_calls += 1

        val = evaluated_states[state]

        if val < minimum_val:
            minimum_state = state
            minimum_val = val

        if random_generator.random() <= p_transition(current_val, val, T):

            # state transition accepted
            current_state = state
            current_val = val

            transitions.append(SPair(current_state, current_val))

        n_iter += 1

        if not evaluation_predicate(
                EvaluationParams(
                    minimum=SPair(minimum_state, minimum_val),
                    current=SPair(current_state, current_val),
                    transitions=transitions,
                    n_iter=n_iter,
                    n_calls=n_calls
                )
        ):
            break

    return minimum_state, evaluated_states, transitions
