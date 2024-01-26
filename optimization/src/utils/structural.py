from typing import Optional, Literal, Mapping

from optimization.src.utils.functional import raise_expr

#-----------------------------------------------------------------------------------------------------------------------

# `max_depth` is a 'maximum depth of flattening':
#
#   flattened([1, [[2]], [3]], max_depth = 1) == [1, [2], 3]
#   flattened([1, [[2]], [3]], max_depth = 2) == [1, 2, 3]
#
# when `max_depth` is None, all the levels are flattened
#
def flattened(A, max_depth: Optional[int] = None):

	# checking for iteration stop
	# (strings are a special case: though iterable, treat them as primitive objects)
	#
	if isinstance(A, str) or (max_depth is not None and max_depth < 0):
		yield A
	else:
		for a in A:
			try:
				yield from flattened \
				(
					a,
					max_depth - 1 if max_depth is not None else None
				)
			except TypeError:
				yield a

# extracts some values from a dictionary and returns them along with a 'smaller' dictionary;
# WARNING: does not preserve the original key insertion order;
#
#   `format` specifies how the specified values should be extracted
#    (by default, values are returned 'unpacked')
#
# examples:
#
#   dict_split
#   (
#       {'a': 0, 'b': 1, 'c': 2},
#       ['b', 'a']
#  	)
#  	== (1, 0, {'c': 2})
#
#   dict_split
#   (
#       {'a': 0, 'b': 1, 'c': 2},
#       ['b', 'a'],
#       format = 'dict'
#  	)
#  	== ({'b': 1, 'a': 0}, {'c': 2})
#
#   dict_split
#   (
#       {'a': 0, 'b': 1},
#       ['c'],
#       format = 'dict'
#  	)
#  	== ({}, {'b': 1, 'a': 0})		# possible output - key order is not preserved
#
def dict_split(d, keys, *, format: Literal['values', 'dict'] = 'values'):

	if isinstance(keys, str):
		keys = (keys,)

	comp_keys = d.keys() - keys

	head = \
	{
		key: d[key]
		for key in keys
		if key in d
	}

	tail = \
	{
		key: d[key]
		for key in comp_keys
	}

	return \
	{
		'values': lambda: (*head.values(), tail),
		'dict':   lambda: (head, tail)
	} \
	.get \
	(
		format,
		lambda: raise_expr(f"dict_split: unsupported format: {format!r}")
	) \
	()

# maps `f` over values of `d` going at most `max_levels` levels 'deep'
# (i.e. mapping over a value if it itself is a dictionary)
#
# note:
# 	when integer, `max_levels` should be positive (i.e. > 0);
#   when `None`, all levels are mapped over;
#
# examples:
#
#    dict_map \
#    (
#    	lambda q: 2*q - 1,
#    	{'a': 0, 'b': 2}
#   )
#   == {'a': -1, 'b': 3}
#
#   dict_map \
#   (
#   	lambda val: [val],
#
#   	{
#   		'a':
#   		{
#   			'X': 1,
#   			'Y': 1,
#   			'C': {'r': 0, 'g': 0, 'b': 1}
#   		},
#   		'b':
#   		{
#   			'X': -1,
#   			'Y': 0,
#   			'C': {'r': 1, 'g': 0, 'b': 0}
#   		},
#   		'c': (-1, 1)
#   	},
#
#   	max_levels = 2
#   )
#   ==
#   {
#   	'a':
#   	{
#   		'X': [1],
#   		'Y': [1],
#   		'C': [{'r': 0, 'g': 0, 'b': 1}]
#   	},
#   	'b':
#   	{
#   		'X': [-1],
#   		'Y': [0],
#   		'C': [{'r': 1, 'g': 0, 'b': 0}]
#   	},
#   	'c': [(-1, 1)]
#   }
#
def dict_map(f, d, *, max_levels: Optional[int] = 1):

	return \
	{
		key:
		(
			dict_map
			(
				f, value,
				max_levels = max_levels - 1 if max_levels is not None else None
			)
				if isinstance(value, dict) and (max_levels is None or max_levels > 1) else
			f(value)
		)
		for key, value in d.items()
	}

# joins dictionaries from `iterable` into one by key
# (the corresponding values are collected into a list);
#
# note:
#    all the not dict-like objects in `iterable` are either skipped (when `drop_tail` is True [default])
#    or collected into a list and returned alongside the merged dict (when `drop_tail` is False)
#
# examples:
#
#    dict_join
#    (
#    	[
#    		{'a': 1,   'b': 3},
#    		{'a': '#', 'b': 0, 'c': 15}
#   	]
#    )
#    ==
#    {'a': [1, '#'], 'b': [3, 0], 'c': [15]}
#
#    dict_join
#    (
#    	[
#    		{'a': 0},
#    		None,
#    		{'a': 1},
#    		(-1, 1)
#   	],
#    	drop_tail = False
#    )
#    ==
#    (
#    	{'a': [0, 1]},
#    	[None, (-1, 1)]
#    )
#
def dict_join(iterable, drop_tail = True):

	res = {}
	tail = []

	for s in iterable:
		try:
			for key, value in s.items():
				res.setdefault(key, []).append(value)
		except (AttributeError, TypeError):
			tail.append(s)

	if drop_tail:
		return res
	else:
		return (res, tail)

# a generator that yield values from `obj` at level `level`;
# (0 corresponds to the top level, `None` - to the last ('deepest') one)
#
# examples:
#
#    let obj =
#    {
#    	'one':   {'a': [1],    'b': [0.5]},
#    	'two':   {'a': [1, 2], 'b': [1, None]},
#    	'three': {'a': [None], 'b': [12]}
#   }
#
#   *collect(obj, level = 0) == *obj.values()
#                            == {'a': [1], 'b': [0.5]}, {'a': [1, 2], 'b': [1, None]}, {'a': [None], 'b': [12]}
#
#   *collect(obj, level = 1) == [1], [0.5], [1, 2], [1, None], [None], [12]
#
def collect(obj: Mapping, level: Optional[int] = None):

	try:
		for value in obj.values():
			if level == 0:
				yield value
			else:
				yield from collect \
				(
					value,
					level = level - 1 if level is not None else None
				)
	except AttributeError:
		yield obj

# returns the 'inverse' of a dictionary,
# i.e. creates a `{value -> [key...]}` mapping from a `{key -> value}` one
#
# example:
#
#   dict_inv({'a': 1, 'b': 2, 'c': 1}) == {1: ['a', 'c'], 2: ['b']}
#
def dict_inv(d):

	res = {}

	for key, value in d.items():
		res.setdefault(value, []).append(key)

	return res
