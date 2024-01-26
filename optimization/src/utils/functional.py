from functools import reduce
from itertools import chain

#-----------------------------------------------------------------------------------------------------------------------

# return the last argument
# (for a "last evaluated expression is the return value" idiom)
#
last = lambda *args: args[-1]

# if-then-else expression in a more convenient order
#
If = lambda condition, if_true, if_false = lambda: None: \
				(if_true if condition else if_false)()

# `if-elif-else` chain as an expression
#
def switch(*args):

	for a in args:

		try:
			flag, value = a
		except:
			flag = True
			value = a

		if callable(flag):
			flag = flag()

		if flag:
			return value

# function composition
# (each function is assumed to be a function of one argument)
#
composition = lambda *f: \
				lambda arg: reduce \
				(
					lambda acc, q: q(acc),
					reversed(f), arg
				)

# returns a function, which will try executing functions from `F` one by one until one succeeds
# (will call `on_fail` if all failed)
#
def try_chain(*F, on_fail = lambda *_, **__: None):

	def _wrapped(*args, **kwargs):

		for f in F:
			try:
				return f(*args, **kwargs)
			except Exception:
				continue

		return on_fail(*args, **kwargs)

	return _wrapped

def try_expr(f, on_fail = lambda _: None):

	try:
		return f()
	except Exception as e:
		return on_fail(e)

# just a raise statement wrapper to turn it into expression
#
def raise_expr(e = None):

	if e is None:
		raise
	else:
		if not isinstance(e, Exception):
			e = Exception(e)

		raise e

# just a `None` filter
#
notnull = lambda args: filter \
			(
				lambda arg: arg is not None,
				args
			)

# return first element in `args` that is not `None`
# (if there is no such element returns `None`)
#
coalesce = lambda args: \
				next \
				(
					chain \
					(
						notnull(args),
						iter((None,))
					)
				)
