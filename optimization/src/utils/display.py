""" Data visualization helpers """

from typing import Union, Iterable, List, Callable, Optional

from itertools import zip_longest

#-----------------------------------------------------------------------------------------------------------------------
## Plain text

# represent an array of `cells` as a grid;
# each `cell` can be either a string (~ a single line)
# or a list of strings (~ multiple lines)
#
def grid_view \
(
	cells: Iterable[Union[str, List[str]]],
	borders = True,
	page_width = 120
):

	cells = \
	[
		(cell,)
			if isinstance(cell, str) else
		cell
		for cell in cells
	]

	max_line_len = max \
	(
		max \
		(
			map(len, cell),
			default = 0
		)
		for cell in cells
	)

	blank = ' ' * max_line_len

	padded_cells = \
	[
		[line.ljust(max_line_len) for line in cell]
		for cell in cells
	]

	sep = '|' if borders else ' '

	full_cell_width = max_line_len + len(sep)
	cells_per_row = (page_width - full_cell_width) // full_cell_width + 1

	if borders:

		bar = '+' + ''.join \
		(
			'-'
				if (k+1) % (max_line_len+1) != 0 else
			'+'
			for k in range(cells_per_row * full_cell_width)
		)
	else:
		bar = ''

	res = f'\n{bar}\n'.join \
	(
		'\n'.join \
		(
			map \
			(
				lambda s: sep + sep.join(s) + sep,
				zip_longest \
				(
					*padded_cells[k:k+cells_per_row],
					fillvalue = blank
				)
			)
		)
		for k in range(0, len(cells), cells_per_row)
	)

	if borders:
		res = bar + '\n' + res + '\n' + bar

	return res

#-----------------------------------------------------------------------------------------------------------------------
## Pandas

try:
	import pandas as pd

	try:
		from bs4 import BeautifulSoup
	except:
		print('BeautifulSoup is not installed, DataFrames may not be rendered properly')

	# 'default' DataFrame to HTML renderer
	#
	def df_to_html \
	(
		df,
		*,
		styler: Optional[Callable] = None,
		path: Optional[str] = None
	):

		df_style = df.style \
			.set_table_styles \
			(
				[
					{
						'selector': 'table',
						'props': 'border-collapse: collapse;'
					},

					{
						'selector': 'th, td',
						'props': \
							f"""
							font-family: monospace;
							font-size: 12pt;

							border: 1px solid gray;
							border-color: #{'a0'*3};

							white-space: pre;
							"""
					},
				]
			) \
			.set_properties \
			(
				**
				{
					'font-family': 'monospace',
					'font-size': '12pt'
				}
			)

		# apply custom styler (if present)
		#
		if styler:
			df_style = df_style.apply(styler)

		html = df_style.render(uuid = '')

		#----------------------------------------------------------------------------------

		# try to remove id-s of `general` selectors
		# (otherwise `border-collapse` CSS property seems to have no effect)
		#
		try:
			html_parsed = BeautifulSoup(html, "html.parser")

			html_style = html_parsed.find('style')
			html_style.string = html_style.string.replace('#T_ ', '')

			html = str(html_parsed)
		except:
			pass

		#----------------------------------------------------------------------------------

		# save to file (if `path` is present)
		#
		if path:

			with open(path, 'w', encoding = 'utf-8') as f:
				f.write(html)

			print(f'Table rendered to {path}')

		#----------------------------------------------------------------------------------

		return html

except:
	pass

#-----------------------------------------------------------------------------------------------------------------------
## Matplotlib

try:
	import numpy as np
	import matplotlib.pyplot as plt

	# generate and plot a demand curve for a `bed`-bedroom units given the `building_config`
	#
	def demand_curve_plot \
	(
		model,
		building_config,
		bed,
		price_format = 'psf',
		price_range = None,
		show_error_margin = None,
		show_training_data = False,
		show_figure_title = True,
		verbose = True
	):

		if show_error_margin is not None:
			raise Exception('`show_error_margin` parameter is deprecated')

		sP, sQ, f = model.demand_curve \
		(
			building_config,
			bed,
			price_format = price_format,
			price_range = price_range,
			full_output = True
		)

		label = f'postal_code: {repr(building_config.postal_code)}, bed: {repr(bed)}'

		if verbose:
			print(f'[{label}]: ', end = '')
			print(f)

		P, Q = f.sample()
		avg_error = 2*f.std_dev

		default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

		color = default_colors[0]
		#color = (0, 0, 0.8)

		fig, ax = plt.subplots()

		ax.plot(P, Q, linewidth = 0.75, color = color)

		if show_training_data:
			ax.scatter(sP, sQ, s = 15, alpha = 0.75, color = color)

		# [needs to be redone]
		#
		if show_error_margin and avg_error:

			q_upper = lambda p: f(p) + avg_error
			q_lower = lambda p: np.maximum(f(p) - avg_error, 0)

			ax.plot(P, q_upper(P), linewidth = 0.5, alpha = 0.1, color = color)
			ax.plot(P, q_lower(P), linewidth = 0.5, alpha = 0.1, color = color)

			ax.fill_between \
			(
				P, q_upper(P), q_lower(P),
				edgecolor = "none",
				alpha = 0.1,
				facecolor = color
			)

		ax.set_xlabel(f'P ({price_format})')
		ax.set_ylabel('Q')

		if show_figure_title:
			fig.suptitle(f'Demand curve ({label})')

		return (fig, ax)

except:
	pass