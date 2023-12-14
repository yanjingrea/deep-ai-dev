import typing
from typing import Union, Iterable, Callable, Mapping, Type

import types
from types import TracebackType

import dataclasses
from dataclasses import fields, make_dataclass

from functools import cached_property

import re

from itertools import groupby, accumulate

from src.utils.console import warning
from src.utils.functional import try_chain, last, If, raise_expr

#-----------------------------------------------------------------------------------------------------------------------
## context manager

# an exception suppressor [to be used in a context manager]
#
#  parameters:
#
#    *types:   		types of exceptions to suppress (if none provided, all exceptions are suppressed)
#    types_except: 	a tuple of exception types NOT to suppress
#    on_error: 		a function to call when an exception is caught
#    verbose:  		whether to display information about caught exceptions or not
#
class noexcept:

	def __init__ \
	(
		self,
		*types,
		types_except: tuple = (),
		on_error:     Callable[[Type[Exception], Exception, TracebackType], None] = lambda e_type, e, tb: None,
		verbose = True
	):
		self.types = types if types else None
		self.types_except = types_except

		self.on_error = on_error
		self.verbose = verbose

	def __enter__(self):
		pass

	def __exit__(self, *args):

		if args[0] is not None:

			if not issubclass(args[0], self.types_except) and (self.types is None or issubclass(args[0], self.types)):
				if self.verbose:
					print(warning(f'[Exception caught]:\n{repr(args[1])}'))

				self.on_error(*args)
			else:
				return False

		return True

#-----------------------------------------------------------------------------------------------------------------------
## decorators for `dataclass`-like classes

# defines a better representation method `disp` for dataclasses with a lot of fields
# (also overrides `__str__` and `__repr__` with it);
# can also be used as a stand-alone dictionary representation function
#
def dict_repr(*args, prefix_delim = None, **disp_kwargs):

	def disp \
	(
		self,
		*,
		indent_level: int = 0,
		display_name: Union[str, Callable[[str], str]] = lambda qualname: qualname
	):

		cls = type(self)

		if issubclass(cls, dict):
			_dict = self
			_qualname = cls.__qualname__ if cls is not dict else None
		else:
			_dict = \
			{
				attr: value
				for attr, value in self.__dict__.items()
				if not isinstance \
				(
					getattr(cls, attr, None),
					(property, cached_property)
				)
			}

			_qualname = cls.__qualname__

		keys = list(_dict.keys())

		tab = lambda n: '\t'*n

		if prefix_delim is not None:

			prefixes = \
			[
				try_chain(lambda m: m[0])
				(
					re.match \
					(
						fr'\w+?{prefix_delim}',
						f'{key}',
						flags = re.ASCII
					)
				)
				for key in keys
			]

			group_sizes = \
			[
				len(tuple(g[-1]))
				for g in groupby(prefixes)
			]

			index_map = \
			[
				n - 1
				for n in accumulate(group_sizes)
			]

			prefix_breaks = \
			[
				index_map[idx]
				for idx in range(len(group_sizes) - 1)
				if
					group_sizes[idx] > 1 or
					group_sizes[idx+1] > 1
			]

			for idx in reversed(prefix_breaks):
				keys.insert(idx+1, None)

		max_key_length = max \
		(
			(
				len(key)
					if isinstance(key, str) else
				len(str(key))
				for key in keys
				if key is not None
			),
			default = 0
		)

		if _qualname:

			if callable(display_name):
				header = tab(indent_level) + f'{display_name(_qualname)}:\n'
			else:
				if display_name:
					header = tab(indent_level) + f'{display_name}:\n'
				else:
					header = ''

		else:
			header = ''

		item_indent = tab(indent_level + 1)
		sub_item_indent = item_indent + ' '*(max_key_length + 1) + '\t'

		base_value_repr = lambda value: f'{value:g}' \
											if isinstance(value, float) else \
										f'{value}'

		value_repr = lambda value: \
			base_value_repr(value) \
			.replace('\n', '\n' + sub_item_indent)

		return \
			header + \
			'\n'.join \
			(
				''
					if key is None else
				item_indent + \
				f'{key}:'.ljust(max_key_length + 1, ' ') + '\t' + value_repr(_dict[key])

				for key in keys
			) + \
			'\n'

	def _wrapper(cls):

		cls.disp = disp
		cls.__str__ = lambda self: disp(self)
		cls.__repr__ = lambda self: disp(self)

		return cls

	if len(args) == 0:
		return _wrapper
	else:
		if len(args) == 1 and isinstance(args[0], dict):

			disp_kwargs.setdefault('indent_level', 0)
			disp_kwargs['indent_level'] -= 1

			return disp(args[0], **disp_kwargs)

		else:
			return _wrapper(*args)

# turns `cls` into a proper Mapping
# with a __getitem__ method that supports 'slicing'
# (ability to get a 'sub-dictionary' of the object's __dict__ by providing a list of keys)
#
def dict_slicer(cls):

	def _getitem(self, key: Union[str, Iterable[str], slice]):

		if isinstance(key, str):
			return self.__dict__.get(key, None)
			#return getattr(self, key, None)

		elif isinstance(key, slice):

			if key == slice(None):
				return self.__dict__
			else:
				raise NotImplementedError()

		else:
			return {name: self.__dict__.get(name, None) for name in key}
			#return {name: getattr(self, name, None) for name in key}

	new_cls = types.new_class \
	(
		cls.__qualname__,
		(cls, Mapping),
		exec_body = lambda ns: ns.update \
		(
			__getitem__ = _getitem,
			__iter__ = lambda self: iter(self.__dict__),
			__len__ = lambda self: len(self.__dict__)
		)
	)

	new_cls.__module__ = cls.__module__

	return new_cls

# defines a `from_dict` method
# which filters out keys in the input that don't correspond to any of the class fields
# (instead of throwing an exception)
#
def dict_constructor(cls):

	def from_dict(cls, params):

		field_names = {f.name for f in fields(cls)}
		common_names = field_names & params.keys()

		return cls \
		(
			**
			{
				name: params[name]
				for name in common_names
			}
		)

	cls.from_dict = classmethod(from_dict)

	return cls

# defines a `cast` method which allows to 'cast' a dataclass instance to one of its bases
# (~ to get a part of the object)
#
def with_downcast(cls):

	def cast(self, base):

		if issubclass(type(self), base):
			return base \
			(
				**
				{
					field.name: self.__dict__[field.name]
					for field in fields(base)
				}
			)
		else:
			return None

	cls.cast = cast

	return cls

# adds a `join` method
# which allows to construct an object by joining instances of its base classes
#
def with_join(cls):

	def dict_merge(args):

		res = {}

		for arg in args:
			res.update(**arg)

		return res

	cls.join = classmethod \
	(
		lambda cls, *objects: cls \
		(
			**dict_merge \
			(
				obj.__dict__
				for obj in objects
				if issubclass(cls, type(obj))
			)
		)
	)

	return cls

# makes every field optional (with a default of `None` if one is not set already)
#
def defaulted(cls):

	MISSING = dataclasses.MISSING

	cls_fields = \
	[
		(
			field.name,
			field.type,

			dataclasses.field \
			(
				default_factory = field.default_factory
			)
				if field.default_factory is not MISSING else
			dataclasses.field \
			(
				default = field.default
							if field.default is not MISSING else
						  None
			)
		)
		for field in fields(cls)
	]

	dataclass_params = \
	{
		name: getattr(cls.__dataclass_params__, name)
		for name in dir(cls.__dataclass_params__)
		if not name.startswith('__')
	}

	new_cls = make_dataclass \
	(
		cls.__qualname__,
		cls_fields,
		bases = (cls,),
		**dataclass_params
	)

	new_cls.__module__ = cls.__module__

	return new_cls

# adds annotated type enforcing to a dataclass;
#
# params:
#
#   implicit_cast:  whether to try casting an attribute to its annotated type in case of type mismatch
#   universal_None: whether to treat `NoneType` as the universal subtype
#                   i.e. to assume that `None` is a valid value for object of any type
#
def type_checking(cls = None, *, implicit_cast = True, universal_None = True, show_warnings = True):

	if cls is None:
		return lambda cls: type_checking \
		(
			cls,
			implicit_cast = implicit_cast,
			universal_None = universal_None,
			show_warnings = show_warnings
		)
	else:

		dataclass_params = \
		{
			name: getattr(cls.__dataclass_params__, name)
			for name in dir(cls.__dataclass_params__)
			if not name.startswith('__')
		}

		@dataclasses.dataclass(**dataclass_params)
		class new_cls(cls):

			def __post_init__(self):

				try:
					super().__post__init__()
				except AttributeError:
					pass

				instance_check = try_chain \
				(
					lambda val, T: isinstance(val, typing.get_origin(T)),
					on_fail = isinstance
				)

				cast = try_chain \
				(
					lambda val, T: typing.get_origin(T)(val),
					on_fail = lambda val, T: T(val)
				)

				for field in fields(type(self)):

					value = getattr(self, field.name)

					try:
						valid_type = (universal_None and value is None) or instance_check(value, field.type)
					except Exception as e:

						if show_warnings:
							print(warning(f"Unable to validate the type for parameter `{field.name}` [{e}]"))

						continue

					if not valid_type:

						if implicit_cast:
							try:
								new_value = cast(value, field.type)
							except Exception as e:
								raise ValueError(f"Unable to cast {repr(value)} to {field.type} [{e}]")

							object.__setattr__(self, field.name, new_value)
						else:
							raise Exception(f"Parameter `{field.name}` should have type {field.type}, not {type(value)}")

		new_cls.__qualname__ = cls.__qualname__
		new_cls.__module__ = cls.__module__

		return new_cls

#-----------------------------------------------------------------------------------------------------------------------
## general decorators

# represents a function wrapper that should return at most once, i.e.
# when invoked, a `StaticCallable` will call a function once and store its return value
# (subsequent calls of that `StaticCallable` will simply return the stored value)
#
# note: if a function throws then no return value is stored
#       and the next invocation of a `StaticCallable` will call a function again
#
class StaticCallable:

	def __init__(self, constructor = lambda: None):
		self.constructor = constructor

	def __call__(self):

		try:
			self.value
		except AttributeError:
			self.value = self.constructor()

		return self.value

#----------------------------------------------------------

# class wrapper for creating 'parametrized generics'
#
def Generic(*arg_names, **defaults):

	params_repr = lambda params: ', '.join \
	(
		f'{key} = {value!r}'
		for key, value in params.items()
	)

	def type_constructor(cls):

		return lambda *args, **kwargs: last \
		(
			template_params := {**defaults},

			template_params.update(zip(arg_names, args)),
			template_params.update(kwargs),

			missing_args := set(arg_names) - template_params.keys(),

			If \
			(
				missing_args,
				lambda: raise_expr \
				(
					"Undefined parameters: " + ', '.join(map(repr, missing_args))
				)
			),

			valid_keys := cls.__dict__.keys() - type('', (), {}).__dict__.keys(),

			T := type \
			(
				f"{cls.__qualname__}[{params_repr(template_params)}]",
				cls.__bases__,
				dict \
				(
					__template_params__ = template_params,
					**
					{
						key: cls.__dict__[key]
						for key in valid_keys
					}
				)
			),

			setattr(T, '__module__', cls.__module__),

			T
		)

	if not defaults and len(arg_names) == 1 and isinstance(arg_names[0], type):

		cls = arg_names[0]
		arg_names = ()

		return type_constructor(cls)
	else:

		# [argument order matters, so cannot use set difference here]
		#
		arg_names = \
		(
			*arg_names,
			*
			(
				name
				for name in defaults.keys()
				if name not in arg_names
			)
		)

		return type_constructor
