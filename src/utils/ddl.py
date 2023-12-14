"""
[DUPLICATE]:
	1. https://gitlab.com/reapl1/data/ds-real-estate-developer/-/tree/master/src/utils/ddl.py
	2. https://github.com/Real-Estate-Analytics/feature_pipelines/tree/main/common/utils/ddl.py
"""

import re

import pandas as pd

from dataclasses import dataclass, field
from typing import Dict, Optional

#-----------------------------------------------------------------------------------------------------------------------

# SQL column name definition
#
class TColumnName:

	@dataclass
	class TColumnParams:
		quoted: bool = False
		lower: bool = True

	#------------------------------------------------------

	@staticmethod
	def normalized(name: str):
		return name.lower().strip('"')
		#return re.sub(r'"(.+?)"', r'\1', name.lower())

	def __init__(self, data: str, **kwargs):
		self.data = self.normalized(data)

		if 'quoted' not in kwargs:
			kwargs['quoted'] = re.match(r'"[\w]+"', data) is not None

		if 'lower' not in kwargs:
			kwargs['lower'] = data != data.upper()

		self.properties = self.TColumnParams(**kwargs)

	def __str__(self):
		res = self.data

		if self.properties.lower:
			res = res.lower()
		else:
			res = res.upper()

		if self.properties.quoted:
			res = f'"{res}"'

		return res

	def __repr__(self):
		return str(self)

	# hash only by the normalized name
	def __hash__(self):
		return hash(self.data)

	# compare only by the normalized name
	def __eq__(self, other):
		return self.data == other.data


# partial DDL definition (`CREATE TABLE` statement only)
# (created for simplicity of DataFrame DDL creation/manipulation before pushing to the database)
#
@dataclass
class DDL:

	table_name: Optional[str]  = None
	types: 		Dict[TColumnName, str] = field(default_factory = lambda: {})

	@classmethod
	def from_dict \
	(
		cls,
		d,
		table_name = None
	):

		return cls \
		(
			table_name,
			{
				TColumnName(name): T
				for name, T in d.items()
			}
		)

	@classmethod
	def from_str \
	(
		cls,
		s: str
	):

		_ = r'\s+'

		Header = rf'create{_}table{_}(if{_}not{_}exists{_})?'
		Name = r'"?(?P<table_name>[\w\.]+)"?'
		Columns = r'\((?P<partial_ddl>.*)\)'

		table_name, partial_ddl = re.search \
		(
			rf'{Header}{Name}\s*{Columns}',
			s,
			flags = re.DOTALL | re.IGNORECASE
		) \
		.group('table_name', 'partial_ddl')

		ColumnName = r'(?P<column_name>"?\w+"?)'
		ColumnType = r'(?P<column_type>[a-z0-9\s\(\)]+)'
		Column = rf'\s*{ColumnName}{_}{ColumnType}'

		pure_type = lambda s: re.sub(rf'{_}encode.+$', '', s, flags = re.IGNORECASE)
		statements = partial_ddl.replace('\n', '').split(',')

		try:
			types = dict \
			(
				(
					lambda column_name, column_type:
					(
						TColumnName(column_name),
						pure_type(column_type)
					)
				)
				(
					*re.match(Column, s, flags = re.IGNORECASE) \
					.group('column_name', 'column_type')
				)
				for s in statements
			)

		except Exception as e:
			if isinstance(e, AttributeError):
				raise Exception \
				(
					"Invalid type definition section:\n" + \
					"(\n" +
					',\n'.join('\t' + s.strip() for s in statements) + \
					"\n)"
				)
			else:
				raise e

		return cls(table_name, types)

	@classmethod
	def from_df \
	(
		cls,
		df: pd.DataFrame,
		target_table = None,
		hook = None
	):

		base_ddl = pd.io.sql.get_schema \
		(
			df.reset_index(drop = True),
			target_table if target_table is not None else '_',
			con = hook.sqla_engine if hook else None
		)

		# checking for column names that contain quotation marks;
		# currently, such names are not handled properly
		#
		for column_name in df.columns:
			if re.match('"\w+"', column_name):
				raise Exception(f'column name \'{column_name}\' contains quotation marks')

		res = cls.from_str(base_ddl)

		if target_table is None:
			res.table_name = None

		return res

	# returns a result of a 'left join' of `self` and `d`:
	#  - all the columns from `self` are retained;
	#  - types of those columns that are both in `self` and in `d` are taken from `d`;
	#
	def join(self, d):

		if isinstance(d, dict):
			d = DDL.from_dict(d)

		return type(self) \
		(
			table_name = d.table_name if d.table_name is not None else self.table_name,
			types = \
			{
				column_name: (d.types[column_name] if column_name in d.types else column_type)
				for column_name, column_type in self.types.items()
			}
		)

	# format: 'full' | 'partial'
	#
	def repr \
	(
		self,
		format = 'full',
		**kwargs
	):

		if format == 'full':

			ine = kwargs.get('if_not_exists', False)

			return \
				f'create table{(" if not exists " if ine else " ")}{self.table_name if self.table_name else ""}\n' + \
				'(\n' + \
				',\n'.join \
				(
					f'\t{key} {value}'
					for key, value in self.types.items()
				) + \
				'\n)'

		elif format == 'partial':

			lines = \
			[
				(str(name), dtype)
				for name, dtype in self.types.items()
			]

			max_head_length = max(len(line[0]) for line in lines)

			return ',\n'.join \
			(
				line[0].ljust(max_head_length, ' ') + '\t' + line[-1]
				for line in lines
			)

	def __str__(self):
		return self.repr(format = 'full')
