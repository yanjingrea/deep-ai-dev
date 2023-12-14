from typing import Union, Callable, Any, Mapping, Dict, List

import types

import inspect
from functools import partial

import hashlib
import pickle

import json

from itertools import chain

try:
	from itertools import pairwise
except ImportError:

	from itertools import tee

	def pairwise(a):
		a, b = tee(a)
		next(b, None)

		return zip(a, b)

from src.utils.console import warning
from src.utils.functional import try_chain, coalesce, last, composition
from src.utils.structural import dict_split

#-----------------------------------------------------------------------------------------------------------------------
## [general]

# append a mapping `row` to a dictionary of lists `table`
#
def append_row(table: Dict[str, List], row: Dict[str, Any]):

	for key in table:
		table[key].append(row.get(key, None))

#----------------------------------------------------------

# ~ a dictionary that returns the same constant value for any key provided
#
class ConstantMapping(Mapping):

	def __init__(self, value):
		self.value = value

	def __getitem__(self, *args):
		return self.value

	def __iter__(self):
		return iter((self.value,))

	def __len__(self):
		return 1

	def __contains__(self, *args):
		return True

#----------------------------------------------------------

TCode = type((lambda: None).__code__)

def is_primitive(obj):

	return \
	(
		obj is None or
		isinstance(obj, (str, bytes, int, float, type))
	)

# unravels `obj` into a (possibly nested) tuple of 'primitive' objects;
#
# notes:
#   if `obj` is 'primitive' already, wraps it into a tuple
#   if `stringify_types` is true, transforms all types into a tuple of strings
#
def as_tuple(obj, stringify_types = True):

	obj_repr = lambda obj: \
	(
		(
			obj.__class__.__qualname__,
			tuple \
			(
				as_tuple(b, stringify_types)
				for b in obj.__bases__
				if b is not object
			),
			obj.__qualname__
		)
			if isinstance(obj, type) and stringify_types else
		obj
	)

	if is_primitive(obj):
		return (obj_repr(obj),)
	else:

		gen_items = try_chain \
		(
			lambda: obj.items(),
			lambda: iter(obj),
			lambda: obj.__dict__.items(),
			lambda: {}.items()
		) \
		()

		gen_code = \
		(
			obj
				if isinstance(obj, TCode) else
			try_chain(lambda: obj.__code__)()
		)

		code_items = \
		(
			{
				name: getattr(gen_code, name)
				for name in dir(gen_code)
				if name.startswith('co_') and name not in {'co_filename', 'co_firstlineno'}
			}
				if gen_code is not None else
			{}
		) \
		.items()

		return \
		(
			obj_repr(type(obj)),
			tuple
			(
				(
					obj_repr(i)
						if is_primitive(i) else
					as_tuple(i, stringify_types)
				)
				for i in chain(gen_items, code_items)
			)
		)

# a 'stable and extended' version of a standard `hash` function
# (supports hashing mutable objects and lambdas)
#
# note on stability:
#  this function relies on the stability of `pickle`
#  which, technically, is not guaranteed across different language/pickle protocol versions
#
def obj_hash(obj):

	s = try_chain \
	(
		pickle.dumps,
		composition(pickle.dumps, as_tuple)
	) \
	(obj)

	if s is None:
		raise Exception(f"Type {type(obj)} is unhashable")

	return hashlib.sha256(s).hexdigest()

#----------------------------------------------------------

# a 'universal' `<` operator:
#
#   uses the built-in `<` or custom `__lt__` whenever possible,
#   but is properly defined even if `a` and `b` are of different types
#   (i.e. it induces order among heterogeneous data)
#
def universal_lt(a, b):

	try:
		return a < b
	except:

		# if both `a` and `b` are tuples -> map `universal_lt` over
		#
		if isinstance(a, tuple) and isinstance(b, tuple):
			return len(a) < len(b) or all(universal_lt(_a, _b) for (_a, _b) in zip(a, b))

		# if one (and only one) among {`a`, `b`} is a type
		#
		if isinstance(a, type) ^ isinstance(b, type):
			# a type is always `smaller` than an instance

			return isinstance(a, type)
		else:
			# compare by type names/`qualname`-s

			if not isinstance(a, type):
				a = type(a)

			if not isinstance(b, type):
				b = type(b)

			return a.__qualname__ < b.__qualname__

#----------------------------------------------------------

# linear interpolation between array-like objects `a` and `b`
#
def lerp(a, b, t: float):

	if type(a) is type(b):
		T = type(a)
	else:
		T = lambda a: a

	a = np.asarray(a)
	b = np.asarray(b)

	return T(a + (b - a) * t)

#----------------------------------------------------------

def is_generator(obj):
	return \
		isinstance(obj, types.GeneratorType) or \
		isinstance(obj, map) or \
		isinstance(obj, filter) or \
		isinstance(obj, range)

# a generalized version of `sum`
#
gen_sum = try_chain \
(
	lambda a: a.sum(),
	lambda a: sum(a)
)

mean = try_chain \
(
	lambda a: a.mean(),
	lambda a: last \
	(
		q := \
			list(a)
				if is_generator(a) else
			a,

		sum(q)/len(q)
	)
)

def zipmap(f, x):

	if is_generator(x):
		x = list(x)

	return zip(x, map(f, x))

#----------------------------------------------------------

# a quick check for 'pickle-ability':
#    if `type(obj)` supports proper pickling, `obj` should be equal/equivalent to `pickle_identity(obj)`
#
pickle_identity = lambda obj: pickle.loads(pickle.dumps(obj))

#----------------------------------------------------------

# general object 'serialization'
# (converts each instance of a custom class in `obj` to a dictionary or a list)
#
def serialized(obj):

	items = try_chain(lambda: obj.items(), lambda: obj.__dict__.items())()

	if items is not None:
		# `obj` is dict-like
		return \
		{
			key: serialized(value)
			for key, value in items
		}
	else:

		items = try_chain(lambda: iter(obj))()

		if not isinstance(obj, str) and items is not None:
			# `obj` is an iterable (but not a string)
			return \
			[
				serialized(value)
				for value in items
			]
		else:
			return obj

#-----------------------------------------------------------------------------------------------------------------------
## Structures

# creates and returns an anonymous class instance with the specified `fields`
#
instance = lambda **fields: type("", (), fields)()

# ~ a pickleable `instance`
#
class struct:

	def __init__(self, *args, **fields):

		if fields:
			self.__dict__.update(fields)
		elif args and len(args) == 1:
			self.__dict__.update(args[0])

	def __iter__(self):
		return iter(self.__dict__.values())

	def __str__(self):

		return '(' + \
				', '.join \
				(
					f'{name}: {repr(value)}'
					for name, value in self.__dict__.items()
				) + \
				')'

	def __repr__(self):
		return str(self)

	def as_dict(self):
		return self.__dict__

	def as_text \
	(
		self,
		format = 'multiline',
		value_repr = lambda name, value: repr(value)
	):

		return \
		{
			'multiline': lambda obj: '\n'.join \
			(
				f'{name}: {value_repr(name, value)}'
				for name, value in obj.as_dict().items()
			)
		} \
		.get(format, str)(self)

	def serialize(self):

		return json.dumps \
		(
			{
				key: (value, type(value).__qualname__)
				for key, value in self.__dict__.items()
			}
		)

	@classmethod
	def deserialize(cls, data):

		return cls \
		(
			{
				name: eval(p[1])(p[0])
				for name, p in json.loads(data).items()
			}
		)

	def __getitem__(self, idx):

		if isinstance(idx, str):
			return self.__dict__[idx]
		else:
			return type(self) \
			(
				**{key: self.__dict__[key] for key in idx}
			)

	def append(self, **kwargs):
		self.__dict__.update(kwargs)
		return self

	def apply(self, F = None, **F_map):

		for field in (self.__dict__.keys() | F_map.keys()):

			f = F_map.get(field, F)

			if f is not None:
				self.__dict__[field] = f(self.__dict__[field])

		return self

	@classmethod
	def join(cls, iterable):

		res = {}

		for s in iterable:
			for key, value in s.__dict__.items():

				res.setdefault(key, []).append(value)

		return cls(**res)

# a 'silent' None
#
null = instance \
(
	__repr__ 		= lambda self: 'null',

	__call__ 		= lambda self, *args, **kwargs: self,

	__eq__ 			= lambda self, other: other is None or isinstance(other, type(self)),
	__hash__ 		= lambda self: hash(None),

	__ne__ 			= lambda self, other: not (self == other),

	__le__ 			= lambda self, other: self == other,
	__ge__ 			= lambda self, other: self == other,

	__lt__ 			= lambda *_: False,
	__gt__ 			= lambda *_: False,

	__bool__ 		= lambda self: False,

	__add__ 		= lambda self, _: self,
	__sub__ 		= lambda self, _: self,
	__mul__ 		= lambda self, _: self,
	__matmul__ 		= lambda self, _: self,
	__truediv__ 	= lambda self, _: self,
	__floordiv__ 	= lambda self, _: self,
	__mod__ 		= lambda self, _: self,
	__divmod__ 		= lambda self, _: self,
	__pow__ 		= lambda self, *_: self,
	__lshift__ 		= lambda self, _: self,
	__rshift__ 		= lambda self, _: self,
	__and__ 		= lambda self, _: self,
	__xor__ 		= lambda self, _: self,
	__or__ 			= lambda self, _: self,

	__neg__			= lambda self: self,
	__pos__ 		= lambda self: self,
	__abs__ 		= lambda self: self,
	__invert__ 		= lambda self: self
)

# a helper function that returns a tuple-like object of type `base` based on the arguments:
#
#   tuple_constructor(1, 2, 3,   base = tuple) -> (1, 2, 3)
#   tuple_constructor([1, 2, 3], base = tuple) -> (1, 2, 3)
#   tuple_constructor(range(3),  base = tuple) -> (0, 1, 2)
#
# note: `base` can be any type constructor/callable
#
def tuple_constructor(*args, base):

	if len(args) == 1 and \
		(
			is_generator(args[0]) or
			isinstance(args[0], list) or
			isinstance(args[0], tuple)
		):
		return base(args[0])
	else:
		return base(args)

# an `infinite` tuple
# (similar to the `defaultlist`, but immutable)
#
class TInfiniteTuple(tuple):

	def __new__(cls, *args, default_value: Union[Callable, Any] = None):

		tuple_new = super().__new__

		obj = tuple_constructor \
		(
			*args,
			base = lambda q: tuple_new(cls, q)
		)

		obj.default_value = default_value

		return obj

	# defined for correct `pickling`
	#
	def __reduce__(self):

		return \
		(
			type(self),			# reconstructor
			((*self,),),		# reconstructor args
			self.__dict__		# object state
		)

	def __getitem__(self, idx: Union[int, slice]):

		if type(idx) not in {int, slice}:
			raise TypeError \
			(
				f'{type(self).__name__} indices must be integers or slices, not {type(idx).__name__}'
			)

		#----------------------------------------------------------

		if isinstance(idx, int):
			if idx >= 0 and idx < len(self):
				return super().__getitem__(idx)
			else:
				if callable(self.default_value):
					return self.default_value()
				else:
					return self.default_value
		else:
			start = coalesce((idx.start, 	0))
			stop  = coalesce((idx.stop, 	len(self)))
			step  = coalesce((idx.step, 	1))

			# treat negative values for `start` and `stop` specially
			# (in order to partially match the expected 'wrapping around' behaviour)

			if start < 0:
				start = len(self) + start

			if stop < 0:
				stop = len(self) + stop

			return type(self) \
			(
				*(self[k] for k in range(start, stop, step))
			)

# `TInfiniteTuple` wrapper (~ parametrized type surrogate)
#
def InfiniteTuple(default_value: Union[Callable, Any] = None):

	return \
		lambda *args: TInfiniteTuple \
		(
			*args,
			default_value = default_value
		)

#----------------------------------------------------------

# a generalization of `functools.partial`
#
def curried(f):

	s = inspect.signature(f).parameters

	def apply_method(self, f, *args, **kwargs):

		if len(args) > len(s):
			raise Exception \
			(
				f'Too many arguments provided'
			)

		_, s_keyword = dict_split \
		(
			s,
			[
				name
				for _, name in zip(args, s)
			],
			format = 'dict'
		)

		if len(kwargs) > len(s_keyword):
			raise Exception \
			(
				f'Too many arguments provided'
			)

		_, s_residual = dict_split \
		(
			s_keyword,
			kwargs.keys(),
			format = 'dict'
		)

		s_unfilled = \
		{
			name
			for name, value in s_residual.items()
			if \
				value.default is inspect.Parameter.empty \
				and value.kind is not inspect.Parameter.VAR_POSITIONAL
		}

		#-------------------------------------------------------------------

		if not s_unfilled:
			return f(*args, **kwargs)
		else:
			return partial(self, f, *args, **kwargs)

	#-------------------------------------------------------------------

	apply = instance \
	(
		__repr__ = lambda _: '[apply]',
		__call__ = apply_method
	)

	return lambda *args, **kwargs: apply(f, *args, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------
## Numpy

try:
	import numpy as np

	# a more accurate np.arange alternative
	# (also, result includes the `stop` point)
	#
	def arange(start, stop, step = 1):

		return np.linspace \
		(
			start,
			stop,
			1 + int(round((stop - start)/step))
		)

	def date(val):
		return np.datetime64(val)

	def between(a, a_min, a_max):
		return (a >= a_min) & (a <= a_max)

	def minmax(a, **params):
		a = np.asarray(a)
		return (a.min(**params), a.max(**params))

	def nan_minmax(a, **params):
		return (np.nanmin(a, **params), np.nanmax(a, **params))

	def soft_minmax(a, q0 = 0.05, **params):
		return (np.nanquantile(a, q = q0, **params), np.nanquantile(a, q = 1 - q0, **params))

except:
	print(warning('numpy is not installed, some functions will be unavailable'))

#-----------------------------------------------------------------------------------------------------------------------

def format_masterdata_name(name_type: str, masterdata: bool, value: str):
	if name_type == 'query' and masterdata:
		part_list = value.split('_')
		part_list[1] = 'master_' + part_list[1]
		return '_'.join(part_list)
	if name_type == 'table_name' and masterdata:
		return 'master_' + value
	else:
		return value
