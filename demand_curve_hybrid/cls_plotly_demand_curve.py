from dataclasses import dataclass

import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from constants.utils import NatureD


@dataclass
class PlotlyDemandCurve:
    P: np.ndarray
    Q: np.ndarray

    def plot(self, fig=None, color=NatureD['blue'], **kwargs):
        if fig is None:
            fig = make_subplots()

        fig.add_trace(
            go.Scatter(
                x=self.P,
                y=self.Q,
                line=dict(
                    color=color,
                    width=4
                ),
                opacity=0.6,
                **kwargs
            )
        )

        return fig
