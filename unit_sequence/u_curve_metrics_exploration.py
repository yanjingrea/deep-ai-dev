import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

from DeepAI_weekly_report.scr_get_paths import dev_figure_dir, report_dir
from unit_sequence.CoreRanker import RandomSelection

COLOR_SCALE = [
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#F2B701",
    "#E73F74",
    "#80BA5A",
    "#E68310"
]

min_year = 2020,
min_stock = 150,
n_groups = 5
highlight_projects = [
    'Hillhaven',
    'The Arcady At Boon Keng',
    'Lumina Grand'
]

random_table = RandomSelection().reference

score_table = pd.read_csv(
    f'/Users/wuyanjing/PycharmProjects/app/DeepAI_weekly_report/output/dev/2024-01-26/data'
    f'/u_curve_projects_score_table.csv'
)
score_table['percentage'] = score_table['quantity'] / score_table['stock']


score_table = score_table[
            score_table['stock'] >= min_stock
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


fig, ax = plt.subplots(figsize=(8, 6))

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
    y=random_table['accuracy'],
    alpha=0.5,
    color=COLOR_SCALE[2],
    lw=3,
    label='random selection'
)

sns.lineplot(
    x=random_table['percentage'],
    y=new_y,
    alpha=0.5,
    color=COLOR_SCALE[4],
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
        px + 0.25,
        py - 0.05,
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

title = f'project level u curve'
ax.set_title(title)

report_path = title.replace('-', '_').replace(' ', '_')
plt.savefig(dev_figure_dir + f'{report_path}.png', dpi=300)
plt.savefig(report_dir + f'{report_path}.png', dpi=300)

print()
