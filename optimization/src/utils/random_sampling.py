import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm

from itertools import zip_longest

from typing import Optional

from optimization.src.utils.misc import gen_sum

#----------------------------------------------------------------

class PartialTilingError(Exception):
    pass

#----------------------------------------------------------------

class random:

    random_generator = default_rng()

    # random value from an interval:
    #
    #    [a, b) | if a < b
    #    (a, b] | if a > b
    #
    @classmethod
    def interval(cls, a, b, *, dim = None, random_generator = None):

        if random_generator is None:
            random_generator = cls.random_generator

        return a + (b - a)*random_generator.random(size = dim)

    # [discretized version of the `interval` above]
    #
    # generates a random point in a grid with endpoints `a` and `b`
    # given either `n` (number of *points* that partition [a, b])
    # or `step` (which is used to calculate `n`)
    #
    @classmethod
    def grid_interval \
    (
        cls,
        a,
        b,
        random_generator = None,
        *,
        step: Optional[float] = None,
        n:	  Optional[int]	  = None		# number of *points* that partition [a, b]
    ):

        if not (step is None) ^ (n is None):
            raise Exception('Only one of the parameters {`step`, `n`} should be specified')

        if random_generator is None:
            random_generator = cls.random_generator

        if step is not None:
            n = round(abs(a - b)/step) + 1

        return a + (b - a)*random_generator.integers(n)/(n-1) if n > 1 else a

    # random point on a unit n-sphere
    #
    @classmethod
    def sphere(cls, n, random_generator = None):

        if random_generator is None:
            random_generator = cls.random_generator

        normalized = lambda x: x/norm(x)

        return normalized(random_generator.standard_normal(n+1))

    # random point in a unit n-ball
    #
    @classmethod
    def ball(cls, n, random_generator = None):

        if random_generator is None:
            random_generator = cls.random_generator

        return cls.sphere(n-1, random_generator)*(random_generator.random()**(1/n))

    # random point in an n-dimensional standard simplex,
    # i.e. returns `t` such that:
    #
    #   sum(t) == 1
    #   len(t) == n
    #   t[k] >= 0 | for all k
    #
    @classmethod
    def simplex(cls, n, random_generator = None):

        if random_generator is None:
            random_generator = cls.random_generator

        # Dirichlet distribution [alpha = 1]
        #  => uniform distribution of points in a standard simplex
        #
        #t = random_generator.dirichlet((1,)*n)

        # (faster version of the above)
        #
        t = -np.log(random_generator.random(n))
        t /= sum(t)

        return t

    # random point from a projection of an (n+1)-dimensional simplex onto n-dimensional space
    # given the constraints, i.e. returns `a` such that:
    #
    #   sum(a) <= A
    #   len(a) == n
    #   a[k] in [a_min, a_max] | for all k
    #
    # note: this function may use rejection sampling,
    #       generating up to `MAX_TRIAL_SAMPLES` points to produce a single valid one;
    #       (if `MAX_TRIAL_SAMPLES` is None, it is set to the approximate upper bound for the given `n`)
    #
    @classmethod
    def simplex_projection \
    (
        cls,
        n,
        A,
        a_min = 0,
        a_max = None,
        *,
        random_generator = None,
        MAX_TRIAL_SAMPLES = None
    ):

        if random_generator is None:
            random_generator = cls.random_generator

        if a_max is None:
            a_max = A

        if a_min > A/n:
            raise Exception('Sampling space is empty')

        if a_max <= A/n:

            # sampling space is actually a cube
            #
            return a_min + (a_max - a_min)*random_generator.random(n)

        else:

            if a_max + a_min*(n - 1) >= A:

                # sampling space is a 'sub-simplex' projection
                #
                return a_min + (A - n*a_min)*cls.simplex(n+1, random_generator)[:-1]

            else:

                # sampling space is a cube sliced by a simplex
                # -> alternating rejection sampling

                if not MAX_TRIAL_SAMPLES:
                    MAX_TRIAL_SAMPLES = round(2*n*np.sqrt(n))

                # `a_max` for a simplex
                #
                a_max_s = A - a_min*(n - 1)

                # `a_max` for a cube
                #
                a_max_c = A/n

                # if `a_max` is closer to `a_max_c` then sample from cube first
                # otherwise sample from simplex first
                #
                flag = int((a_max - a_max_c) > (a_max_s - a_max))

                for k in range(MAX_TRIAL_SAMPLES):

                    if k % 2 == flag:
                        a = a_min + (a_max - a_min)*random_generator.random(n)
                    else:
                        a = a_min + (A - n*a_min)*cls.simplex(n+1, random_generator)[:-1]

                    if a.sum() <= A and (a < a_max).all():
                        return a

            raise Exception('Unable to sample a valid point')

    # returns an array of arrays `a` such that:
    #
    #   shape(a) 	== (len(a_ranges), ?)
    #   flat_sum(a) <= a_total
    #
    #   a[k, :] in a_ranges[k]	| for all k
    #
    # interpretation and examples:
    #
    #   the generated array `a` can be thought of
    #   as a solution to the following problem:
    #
    #     given a building with cross-section area `a_total`
    #     generate a possible configuration of units per floor such that:
    #
    #       there are `len(a_ranges)` possible unit types
    #
    #       there are `len(a[k])` units of type `k`
    #       with `a[k]` being an array of areas for that unit type
    #       (with each element from range `a_ranges[k]`)
    #
    #       the whole floor is covered by units with the exception of a gap
    #       which cannot accommodate one more whole unit of any type
    #
    # notes:
    #
    #   1. `a_ranges` is supposed to be an array of pairs,
    #       with the minimal element of each pair being the first one
    #
    #   2. `interval_sampler` argument can be supplied to define a custom
    #       sampling method from each interval from `a_ranges`;
    #       if none provided it defaults to `random.interval`
    #
    #   3.	if `frozen` is True, the result is a tuple of tuples,
    #       otherwise it's a list of lists
    #
    @classmethod
    def partial_tiling \
    (
        cls,
        a_total,
        a_ranges,
        *,
        interval_sampler = None,
        frozen = True,
        random_generator = None
    ):

        if random_generator is None:
            random_generator = cls.random_generator

        if interval_sampler is None:
            interval_sampler = cls.interval

        n = len(a_ranges)
        res = [[] for _ in range(n)]

        k_prev = None
        m_prev = None

        while n > 0:

            a_ranges = \
            [
                (r[0], min(r[1], a_total))
                for r in a_ranges
                if r[0] <= a_total
            ]

            n = len(a_ranges)

            if n > 0:

                k = random_generator.integers(n)
                a = interval_sampler(*a_ranges[k], random_generator = random_generator)

                res[k].append(a)

                a_total -= a

                k_prev = k
                m_prev = a_ranges[k][1]

            else:
                # 're-choose' the last sample to be of the maximum possible value
                #
                if k_prev is not None:
                    res[k_prev][-1] = m_prev

        if frozen:
            res = tuple(map(tuple, res))

        return res

    # a more restricted version of `partial_tiling`;
    #
    # additional params:
    #
    #  `n_ranges`:          a list of quantity ranges (a list of pairs) for the corresponding 'tiles', i.e.:
    #                       the amount of tiles of size `a_ranges[k]` should be within `n_ranges[k]` for all `k`
    #
    #  `MAX_TRIAL_SAMPLES`: [see `simplex_projection` description]
    #
    # note:     this function has no `interval_sampler` parameter
    #           as it is tricky to specify an arbitrary interval sampler while sampling from a simplex projection
    #
    # warning:  this function has a side effect of modifying elements of `n_ranges`:
    #            - all null ranges (`None`-s) are replaced with the broadest valid range
    #            - all incomplete ranges (containing `None` as any of the elements) are adjusted
    #
    @classmethod
    def partial_tiling_ex \
    (
        cls,
        a_total,
        a_ranges,
        n_ranges = None,
        *,
        frozen = True,
        random_generator = None,
        MAX_TRIAL_SAMPLES = None
    ):

        if not n_ranges:

            # fall back to a simpler case
            #
            return cls.partial_tiling \
            (
                a_total,
                a_ranges,
                frozen = frozen,
                random_generator = random_generator
            )

        else:

            if random_generator is None:
                random_generator = cls.random_generator

            #---------------------------------------------------------------
            # validation of `n_ranges`

            a_total_min = \
                sum \
                (
                    a[0]*r[0]
                        if r else
                    0
                    for a, r in zip(a_ranges, n_ranges)
                ) \
                    if n_ranges else \
                0

            # ! captures `a_total` and `a_total_min` !
            #
            a_total_adj = lambda: a_total - a_total_min

            if a_total_adj() < 0:
                raise PartialTilingError \
                (
                    "Not enough area to accommodate the minimum requested amount of tiles"
                )

            for idx, (r, a) in enumerate(zip_longest(n_ranges, a_ranges)):

                if r is None:
                    r = (0, None)

                if r[0] is None:
                    r = (0, r[1])

                n_max = r[0] + a_total_adj() // a[0]

                if r[1] is None or r[1] > n_max:
                    n_ranges[idx] = (r[0], n_max)

                if n_ranges[idx][0] > n_ranges[idx][1]:
                    raise PartialTilingError \
                    (
                        f"Invalid range in `n_ranges`: {n_ranges[idx]} [the lower value should be first]"
                    )

            # rebinding to a copy (as this list is going to be modified below)
            #
            n_ranges = n_ranges[:]

            #---------------------------------------------------------------

            range_width = lambda r: abs(r[1] - r[0])

            n_ranges_idx_mask = [True]*len(n_ranges)

            # ! captures `n_ranges` and `n_ranges_idx_mask`!
            #
            most_restricted_range_idx = lambda: min \
            (
                filter(lambda idx: n_ranges_idx_mask[idx], range(len(n_ranges))),
                key = lambda idx: range_width(n_ranges[idx])
            )

            N = len(a_ranges)
            res = [None for _ in range(N)]

            for _ in range(N):

                idx = most_restricted_range_idx()

                n_range = n_ranges[idx]

                a_range = a_ranges[idx]

                a_min = a_range[0]
                a_max = min(a_range[1], a_total)

                q_min = a_range[0]*n_range[0]

                if a_min > a_max:

                    if n_range[0] == 0:
                        a = []
                    else:
                        raise PartialTilingError('Unable to generate a valid partial tiling [area]')
                else:
                    n = random_generator.integers(*n_range, endpoint = True)

                    if n == 0:
                        a = []
                    else:
                        a = cls.simplex_projection \
                        (
                            n,
                            a_total_adj() + q_min,
                            a_min,
                            a_max,
                            MAX_TRIAL_SAMPLES = MAX_TRIAL_SAMPLES
                        )

                n_ranges_idx_mask[idx] = False

                a_total -= gen_sum(a)
                a_total_min -= q_min

                res[idx] = list(a)

                # adjusting `n_ranges`
                #
                for idx, (n_min, n_max) in enumerate(n_ranges):

                    n_max = min(n_max, n_min + a_total_adj() // a_ranges[idx][0])

                    if n_min > n_max:
                        raise PartialTilingError('Unable to generate a valid partial tiling [quantity]')

                    n_ranges[idx] = (n_min, n_max)

            #---------------------------------------------------------------

            FLOAT_ABS_ERROR = 1e-10

            # adjusting if necessary
            #
            if a_total < 0 or a_total > FLOAT_ABS_ERROR:

                gap = a_total

                for k in random_generator.permutation(N):

                    res[k].sort(reverse = True)

                    for idx, val in enumerate(res[k]):

                        new_val = min(max(val + gap, a_ranges[k][0]), a_ranges[k][1])
                        res[k][idx] = new_val

                        gap -= new_val - val

                        if abs(gap) < FLOAT_ABS_ERROR:
                            break

                    if abs(gap) < FLOAT_ABS_ERROR:
                        break

            #---------------------------------------------------------------

            if frozen:
                res = tuple(map(tuple, res))

            return res
