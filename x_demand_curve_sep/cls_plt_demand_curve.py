from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from constants.utils import NatureD


@dataclass
class PltDemandCurve:
    P: np.ndarray
    Q: np.ndarray

    def plot(self, fig=None, ax=None, color=NatureD['blue'], **kwargs):
        if not ax:
            fig, ax = plt.subplots()

        ax.plot(
            self.P,
            self.Q,
            color=color,
            alpha=0.6,
            lw=4,
            **kwargs
        )

        return fig, ax