import os
from dataclasses import dataclass
from os.path import dirname, realpath
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from unit_sequence.CoreRanker import CoreRanker, RandomSelection
from adjustText import adjust_text

output_dir = dirname(realpath(__file__)) + os.sep + 'output' + os.sep
figures_dir = output_dir + 'figures/'
tables_dir = output_dir + 'tables/'

COLOR_SCALE = [
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#F2B701",
    "#E73F74",
    "#80BA5A",
    "#E68310"
]


def plot_random_u_curve(fig=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    random_res = RandomSelection().reference

    sns.lineplot(
        x=random_res['percentage'],
        y=random_res['accuracy'],
        alpha=0.5,
        color=COLOR_SCALE[2],
        lw=3,
        label='random selection accuracy',
        ax=ax
    )

    return fig, ax


@dataclass
class RankerEvaluator:
    raw_data: pd.DataFrame
    groupby_keys: list

    actual_label: str
    pred_label: str

    price_difference_col: Optional[str] = None

    def __post_init__(self):
        self.projects_score_table = self._calculate_gain(on='project_name')
        self.projects_bed_score_table = self._calculate_gain(on=self.groupby_keys)

    def _extract_consistent_targets(self, df):

        y_true = df[self.actual_label]
        y_pred = df[self.pred_label]
        if y_true.dtype != y_pred.dtype:
            y_pred = y_pred.apply(lambda a: True if a == 'True' else False)

        return y_true, y_pred

    def _calculate_gain(self, on):
        groupby_data = self.raw_data.groupby(on)
        score = pd.Series(groupby_data.apply(self.calculate_score), name='score')

        q = groupby_data[self.actual_label].apply(lambda a: len(a[a == True])).rename('quantity')
        s = groupby_data['unit'].count().rename('stock')
        q_percent = q / s

        random_score = q_percent.apply(RandomSelection().refer)
        gain = pd.Series(score - random_score, name='gain')

        price_diff = groupby_data[self.price_difference_col]
        MAPD = price_diff.apply(lambda a: np.abs(a).mean()).rename('avg_price_diff')
        MPD = price_diff.mean().apply(lambda a: 'postive' if a > 0 else 'negative').rename('price_diff_sign')

        for series in [gain, q, s, MAPD, MPD]:
            score = pd.merge(score, series, left_index=True, right_index=True)

        return score.reset_index()

    def calculate_score(self, df):
        y_true, y_pred = self._extract_consistent_targets(df)
        return accuracy_score(
            y_true=y_true,
            y_pred=y_pred
        )

    def calculate_corr(self, on, base, other):

        groupby_data = self.raw_data.groupby(on)

        corr = groupby_data.apply(lambda df: df[base].corr(df[other]))

        print(f'mean correlation between {base} and {other}: {corr.mean() * 100 :g}%')
        print(f'median correlation between {base} and {other}: {corr.median() * 100 :g}%')

        return corr

    def plot_confusion_matrix(self):
        y_true, y_pred = self._extract_consistent_targets(self.raw_data)

        fig, ax = plt.subplots()
        matrix_params = dict(cmap=plt.cm.Blues, colorbar=False, normalize='true', ax=ax)
        ConfusionMatrixDisplay.from_predictions(
            y_true=y_true, y_pred=y_pred,
            **matrix_params
        )

    def compare_prediction_and_random(self):
        score_table = self.projects_bed_score_table

        nrows = 2
        ncols = 3

        fig, axs = plt.subplots(nrows, ncols, figsize=(16, 9))

        for idx, bed_num in enumerate(np.arange(1, 6)):
            ax = plt.subplot(nrows, ncols, idx + 1)
            ax.set_title(f'{bed_num}-bedroom', fontsize=10)
            ax.set_xlabel(f'percentage of units sold')
            ax.set_ylabel(f'prediction_accuracy')

            bed_score_table = score_table[score_table['num_of_bedrooms'] == bed_num]

            q_percent = bed_score_table['quantity'] / bed_score_table['stock']

            sns.scatterplot(
                x=q_percent,
                y=bed_score_table['score'],
                hue=bed_score_table['stock'],
                ax=ax
            )

            random_res = RandomSelection().reference
            sns.lineplot(x=random_res['percentage'], y=random_res['accuracy'], alpha=0.5)

        title = f'u curve'
        fig.suptitle(title)
        fig.savefig(figures_dir + title.replace(" ", '-') + '.png')

    def plot_scatter_above_u_curve(
        self,
        *,
        fig=None,
        ax=None,
        control_stock=0,
        highlight_projects: list = None
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        score_table = self.projects_score_table[
            self.projects_score_table['stock'] >= control_stock
            ].copy()
        # score_table = score_table[score_table['avg_price_diff'] <= 0.2]
        score_table['q_percent'] = score_table['quantity'] / score_table['stock']

        x = 'q_percent'
        y = 'score'

        sample_projects = score_table['project_name'].sample(3)

        if highlight_projects:

            color = score_table['project_name'].apply(
                lambda a: COLOR_SCALE[4] if a in highlight_projects else COLOR_SCALE[0]
            )
            final_highlight = np.append(sample_projects, highlight_projects)
        else:
            color = COLOR_SCALE[0]
            final_highlight = sample_projects

        sns.scatterplot(
            data=score_table,
            x=x,
            y=y,
            ax=ax,
            fc="w",
            ec=color,
            s=60,
            lw=3,
            zorder=12,
            alpha=0.7,
            label='model prediction'
        )

        texts = []
        for idx, row in score_table.iterrows():

            project = row['project_name']

            if project not in final_highlight:
                continue

            px, py = score_table[x][idx], score_table[y][idx]
            texts.append(ax.text(px, py, project, fontsize=10))

        adjust_text(
            texts,
            # expand_points=(0.5, 0.5),
            arrowprops=dict(arrowstyle="-", lw=0.5),
            ax=ax
        )

        return fig, ax

    def plot_project_level_fitted_curve(
        self,
        *,
        fig,
        ax,
        control_stock
    ):

        # fitted curve
        # -----------------------------------------------------
        score_table = self.projects_score_table[
            self.projects_score_table['stock'] >= control_stock
            ].copy()
        score_table['percentage'] = score_table['quantity'] / score_table['stock']

        random_table = RandomSelection().reference

        x = 'percentage'
        y = 'score'

        from scipy.interpolate import UnivariateSpline

        spl = UnivariateSpline(
            score_table.sort_values(x)[x],
            score_table.sort_values(x)[y],
            k=2
        )
        new_y = spl(random_table['percentage'])
        new_y = np.clip(new_y, 0, 1)

        sns.lineplot(
            x=random_table['percentage'],
            y=new_y,
            alpha=0.5,
            color=COLOR_SCALE[0],
            lw=3,
            label='model prediction'
        )
        plt.fill_between(
            x=random_table['percentage'],
            y1=new_y,
            y2=random_table['accuracy'],
            color=COLOR_SCALE[2],
            alpha=0.2,
            label="model's gain"
        )

        random_gain_area = np.trapz(
            y=random_table['accuracy'],
            x=random_table['percentage'],
            dx=0.01
        )
        model_gain_area = np.trapz(
            y=new_y,
            x=random_table['percentage'],
            dx=0.01
        )

        model_net_gain = model_gain_area - random_gain_area

        texts = []

        px = random_table.iloc[25][x]

        mask = random_table['percentage'] == px

        random_y = random_table['accuracy'][mask].mean()
        model_y = new_y[mask].mean()

        py = random_y + (model_y - random_y) / 2

        texts.append(
            ax.text(
                px,
                py,
                f"net gain: {model_net_gain * 100: .2f}% \n"
                f"= {model_net_gain / (1 - random_gain_area) * 100: .2f}% of maximum gain",
                fontsize=10
            )
        )

        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-", lw=0.5),
            ax=ax
        )

        ax.legend()

        return fig, ax

    def plot_project_level_u_curve(self, control_stock=0):

        fig, ax = plt.subplots(figsize=(8, 6))

        plot_random_u_curve(fig, ax)

        self.plot_scatter_above_u_curve(control_stock=control_stock, fig=fig, ax=ax)

        return fig, ax


@dataclass
class RankerComparative:
    ranker1: CoreRanker
    ranker2: CoreRanker

    def compare(self, min_stock):
        control_stock = min_stock

        print(f'Rank with raw avm price...')
        raw_avm_score = self.ranker1.test()

        print(f'\nRank with adj avm price...')
        index_mode = 'local'
        adj_avm_score = self.ranker2.test()

        fig, ax = plt.subplots(figsize=(8, 6))
        random_res = RandomSelection().reference

        for label, score_table in zip(
                ['without_stack_index', 'with_stack_index'],
                [raw_avm_score, adj_avm_score]
        ):
            score_table = score_table[score_table['stock'] >= control_stock].copy()

            q_percent = score_table['quantity'] / score_table['stock']
            sns.scatterplot(
                x=q_percent,
                y=score_table['score'],
                ax=ax,
                label=label
            )
        sns.lineplot(x=random_res['percentage'], y=random_res['accuracy'], alpha=0.5)

        # ----------------------------------------------
        avm_res = raw_avm_score.merge(
            adj_avm_score,
            on=['project_name', 'quantity', 'stock']
        )
        # avm_res = avm_res[(avm_res['score_x'] != 1) & (avm_res['gain_x'] != 0)].copy()
        avm_res = avm_res[avm_res['stock'] >= control_stock].copy()

        avm_res['index_gain'] = avm_res['score_y'] - avm_res['score_x']
        avm_res['color'] = avm_res['index_gain'].apply(lambda a: 'red' if a > 0 else 'green')

        pure_gain = avm_res['index_gain'].mean()
        print(f'stack index contribute to pure gain {pure_gain * 100 :g}%')

        plt.vlines(
            x=avm_res['quantity'] / avm_res['stock'],
            ymin=avm_res['score_x'],
            ymax=avm_res['score_y'],
            linestyles='--',
            colors=avm_res['color'],
            alpha=0.8
        )

        plt.savefig(figures_dir + f'gain_comparison_plot_{index_mode}_{self.ranker1.min_year}.png', dpi=300)
