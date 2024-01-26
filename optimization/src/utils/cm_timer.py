from timeit import default_timer
from typing import Optional

# a simple context manager timer
#
#   yields a timer: a function that returns time elapsed (in seconds) after its creation
#   (i.e. from the start of the `with` block)
#
#   can also be used for wrapping a function into a timed context
#
#   parameters:
#
#     fmt:       format string for printing the elapsed time;
#                if None or an empty string is passed, nothing is printed
#
class cm_timer:

	def __init__(self, fmt: Optional[str] = '{:g} s'):

		self.fmt = fmt
		self.start = None
		self.timer = None

	def __enter__(self):

		self.start = default_timer()
		self.timer = lambda: default_timer() - self.start

		return self.timer

	def __exit__(self, *exception_params):

		if self.fmt:
			print(self.fmt.format(self.timer()))

	# function decorator definition
	#
	def __call__(self, f):

		def _f(*args, **kwargs):

			with type(self)(fmt = f'{f.__name__}: ' + self.fmt):
				return f(*args, **kwargs)

		return _f
