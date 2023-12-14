from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from constants.utils import NatureD, NatureL, set_plot_format
from optimization.project_config import ProjectConfig

"""
This file contains essential classes describing the results of best-selling path and best land use tests.
"""

set_plot_format(plt)
bed_nums = np.arange(1, 6)


@dataclass
class UnitSalesPath:
    """
    Description of Sales Path Attributes of a single bedroom type:
        - Quantity: Represents the quantity of items sold.
        - PSF (Price Per Square Foot): Indicates the price per square foot.
        - Revenue: Represents the total revenue generated.
    """

    bed_num: int
    quantity_path: np.ndarray
    psf_path: np.ndarray
    price_path: np.ndarray
    revenue_path: np.ndarray
    discounted_revenue_path: np.ndarray
    total_revenue: float
    discounted_total_revenue: float

    @property
    def avg_psf(self):
        return self.psf_path.mean()


@dataclass
class ProjectSalesPaths:

    """
    A dataclass that describes the selling paths of different bedroom types.

    Attributes:
        paths (Mapping[int, UnitSalesPath]): A mapping of bedroom types to their respective sales paths.
    """

    paths: Mapping[int, UnitSalesPath]

    @property
    def avg_psf(self) -> tuple:
        """
        Calculate the average price per square foot (PSF) for each bedroom type.

        Returns:
            Tuple[float]: A tuple containing the average PSF for each bedroom type.
        """

        psf_tuple = ()

        for b in bed_nums:
            if b in self.paths.keys():
                bed_psf = self.paths[b].avg_psf
            else:
                bed_psf = None

            psf_tuple += (bed_psf,)

        return psf_tuple

    @property
    def revenue(self):
        """
            Calculate the total revenue from all bedroom types.
        """
        return sum(self.paths[b].total_revenue for b in self.paths.keys())

    @property
    def discounted_revenue(self) -> float:
        """
            Calculate the total discounted revenue from all bedroom types.
        """
        return sum(self.paths[b].discounted_total_revenue for b in self.paths.keys())

    def plot(self):
        """
        Plot the selling paths for different bedroom types.
        Returns:
            Figure, Axes: Matplotlib Figure and Axes objects.
        """

        n_cols = 1
        n_rows = len(self.paths.keys())

        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(n_cols * 12, n_rows * 3)
        )

        for ax_idx, bed in enumerate(list(self.paths.keys())):

            bed_path = self.paths[bed]

            # plot
            # -------------------------------------------------------
            # x-axis
            t_path = np.arange(1, len(bed_path.psf_path) + 1)

            ax = plt.subplot(n_rows, n_cols, ax_idx + 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax2 = ax.twinx()

            # quantity
            ax2.bar(t_path, bed_path.quantity_path, color=NatureD['blue'], alpha=0.8)

            for idx in range(len(t_path)):
                value = bed_path.quantity_path[idx]
                ax2.text(t_path[idx], value, value)
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

            # price
            ax.plot(t_path, bed_path.psf_path, marker='.', color=NatureL['red'], zorder=2, alpha=0.9)
            for idx in range(len(t_path)):
                value = bed_path.psf_path[idx]
                ax.text(t_path[idx], value, round(value, 2), zorder=1)

            ax.set_title(f'{bed}-bedroom')
            ax.set_xlabel('Launching Period')
            ax.set_ylabel('Price psf')
            ax2.set_ylabel('Sales quantity')
            ax.text(
                0.8,
                0.8,
                f"revenue {bed_path.total_revenue / 10 ** 6: g} millions",
                transform=ax.transAxes, horizontalalignment='left', verticalalignment='bottom'
            )

        return fig, axs

    def to_dataframe(self):
        """
        Convert selling paths data to a DataFrame.

        Returns:
            DataFrame: A DataFrame containing important information about the selling paths.
        """

        res = pd.DataFrame()
        for ax_idx, bed in enumerate(list(self.paths.keys())):
            bed_path = self.paths[bed]

            bed_df = pd.DataFrame(bed_path.__dict__)
            bed_df['launching_period'] = np.arange(1, len(bed_path.psf_path) + 1)

            res = pd.concat([res, bed_df], ignore_index=True)

        return res


@dataclass
class ConfigRevenue:

    """
    A dataclass that takes project's configuration and sales paths to summarize revenue-related information.

    Attributes:
        cfg (ProjectConfig): The project's configuration.
        paths (ProjectSalesPaths): The sales paths for different bedroom types.
    """

    cfg: ProjectConfig
    paths: ProjectSalesPaths

    def __post_init__(self):
        """
        Initialize additional attributes after the object is created.
        Calculates quantity, size, average PSF, and revenue from the provided configuration and sales paths.
        """

        self.quantity = self.cfg.total_unit_count
        self.size = self.cfg.avg_unit_size
        self.avg_psf = self.paths.avg_psf
        self.revenue = self.paths.revenue

    def summary(self):
        """
        Print a summary of the calculated attributes.
        """

        for attr in set(self.__dict__.keys()).difference(self.__dataclass_fields__.keys()):

            value = self.__getattribute__(attr)

            if isinstance(value, tuple):

                value = tuple(int(i) if i is not None else 0 for i in value)

            else:
                value = int(value)

            print(f'{attr}: {value}')

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert revenue-related information to a DataFrame for analysis.

        Returns:
            DataFrame: A DataFrame containing suggested units, area, and estimated selling prices per square foot.
        """

        return pd.DataFrame(
            {
                'Bedroom Type': [f'{i} Bedroom' for i in self.cfg.available_bed],
                'Suggested Units': [self.cfg.get_units_count(i) for i in self.cfg.available_bed],
                'Suggested Area (sqm)': [self.cfg.avg_unit_size_per_bed(i) for i in self.cfg.available_bed],
                'Est Selling Price PSF': [
                    np.round(self.avg_psf[i - 1], 2) for i in self.cfg.available_bed
                ]
            }
        )
