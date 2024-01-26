import numpy as np
from numpy.linalg import norm
from numpy.random import default_rng

from enum import Enum, unique, auto

from dataclasses import dataclass, asdict

from scipy.interpolate import CubicSpline
from scipy.optimize import fmin, fmin_bfgs, fmin_slsqp

from optimization.src.utils.console import highlighted, error

#---------------------------------------------------------

@unique
class Tag(Enum):

	Absolute = auto()
	Relative = auto()

def density(data, width = 0.05, width_type = Tag.Relative, oversampling = 9, **_):

	# gaussian kernel
	#
	G = lambda x0, sigma: \
			lambda x: \
				np.exp(-0.5*((x - x0)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)

	#-----------------------------------------------------------------

	x0, x1 = np.min(data), np.max(data)

	abs_width = \
	{
		Tag.Relative: (x1 - x0)*width,
		Tag.Absolute: width
	} \
	[width_type]

	N = int(oversampling*np.round((x1 - x0)/abs_width) + 1)

	X = np.linspace(x0 - 2*abs_width, x1 + 2*abs_width, N)
	dx = (X[-1] - X[0])/(N - 1)

	d = sum \
	(
		G \
		(
			x,
			0.5*(abs(x*width) if width_type is Tag.Relative else abs_width)
		) \
		(X)
		for x in data
	)

	area = np.trapz(d, dx = dx)
	#area = sum(d)*dx

	return X, d/area

def peaks(x, y, bounds = None):

	if bounds is None:
		bounds = (np.min(x), np.max(x))

	s = CubicSpline(x, y, extrapolate = None)

	ds = s.derivative()
	ds2 = s.derivative(2)

	r = np.fromiter \
	(
		filter \
		(
			lambda x: x >= bounds[0] and x <= bounds[1] and ds2(x) < 0,
			ds.roots()
		),
		dtype = float
	)

	q = ds2.roots()

	sigma = np.fromiter \
	(
		(
			#np.mean \
			np.max \
			(
				np.fromiter \
				(
					map \
					(
						lambda a: np.min(a),
						filter \
						(
							lambda a: a.size != 0,
							[
								np.abs(x - q[q < x]),
								np.abs(x - q[q > x])
							]
						)
					),
					dtype = float
				)
			)
			for x in r
		),
		dtype = float
	)

	return r, s(r), sigma

class GaussianMixture:

	C0 = np.sqrt(2*np.pi)

	@dataclass
	class TArg:

		"""
		represents a Gaussian component with properties A, mu, and sigma.
		It provides methods to
			calculate the Gaussian function, its derivative, and the maximum value of the Gaussian.
			add and subtract two Gaussian components.
		"""

		A: 		float
		mu: 	float
		sigma: 	float

		def __iter__(self):
			return iter((self.A, self.mu, self.sigma))

		def max_value(self):
			return 1/(GaussianMixture.C0*self.sigma)

		def norm(self):
			return GaussianMixture.C0*self.A*self.sigma

		def __call__(self, x, normalized = False):

			N = self.norm() if normalized else 1.0
			return self.A*np.exp(-0.5*((x - self.mu)/self.sigma)**2)/N

		def derivative(self, normalized = False):

			N = self.norm() if normalized else 1.0

			return lambda x: \
						-self.A*((x - self.mu)/self.sigma**2)*np.exp(-0.5*((x - self.mu)/self.sigma)**2)/N

		def __add__(self, arg):

			if isinstance(arg, type(self)):
				return type(self) \
				(
					(self.A + arg.A)/2,
					(self.mu + arg.mu)/2,
					np.sqrt(abs(self.sigma*arg.sigma))		# ~ average sigma
				)

			else:
				raise Exception \
				(
					'operator+(a: {}, b: {}) is not defined'.format \
					(
						type(self).__name__,
						type(arg).__name__
					)
				)

		def __sub__(self, arg):

			if isinstance(arg, type(self)):
				return \
				(
					self.mu - arg.mu,
					max(abs(self.sigma), abs(arg.sigma))
					#np.sqrt(abs(self.sigma*arg.sigma))		# ~ average sigma
				)

			else:
				raise Exception \
				(
					'operator-(a: {}, b: {}) is not defined'.format \
					(
						type(self).__name__,
						type(arg).__name__
					)
				)

		def __str__(self):

			return '(' + \
					',\t'.join \
					(
						f'{name}: {value:g}'
						for name, value in asdict(self).items()
					) + \
					')'

	#-----------------------------------------------------------------

	#args: List[TArg]

	def __init__(self, args = []):

		self.args = list \
		(
			map \
			(
				lambda arg: self.TArg(*arg),
				args
			)
		)

	def norm(self):
		return sum(p.norm() for p in self.args)

	def max_value(self):
		return max(arg.A for arg in self.args)/self.norm()

	def __call__(self, x):
		#return sum(p(x) for p in self.args)/self.norm()
		return sum(p.A * p(x, normalized = True) for p in self.args)/sum(p.A for p in self.args)

	def __iter__(self):
		return iter(self.args)

	def __str__(self):
		return f'{type(self).__name__}:\n' + '\n'.join(map(lambda arg: '\t' + str(arg), self.args))

	def gen_samples(self, N, rg = default_rng()):

		weights = np.fromiter(map(lambda arg: arg.A, self.args), dtype = float)
		idxs = rg.choice(len(self.args), p = weights/sum(weights), size = N)

		return np.fromiter \
		(
			(
				rg.normal(self.args[k].mu, self.args[k].sigma)
				for k in idxs
			),
			dtype = float
		)

	def class_borders(self, return_pairs = True):

		if self.args:

			n_sigma = 3	# [only used at the distribution edges]

			p = []		# inflection(*) points

			p.append(self.args[0].mu - n_sigma*self.args[0].sigma)

			for k in range(len(self.args) - 1):

				d0 = self.args[k].derivative()
				d1 = self.args[k+1].derivative()

				x0 = 0.5*(self.args[k+1].mu + self.args[k].mu)

				L = lambda x: self.args[k](x) + self.args[k+1](x)
				dL = lambda x: d0(x) + d1(x)

				rel_tolerance = 1e-8

				# constrained optimization
				#
				x_min = fmin_slsqp \
				(
					L,
					x0,
					fprime = dL,
					bounds = [(self.args[k].mu, self.args[k+1].mu)],
					acc = rel_tolerance*L(x0),
					disp = False
				)[0]

				p.append(x_min)

			p.append(self.args[-1].mu + n_sigma*self.args[-1].sigma)

			if return_pairs:
				p = [(p[k], p[k+1]) for k in range(len(p)-1)]

			return p

		return []

	def classify(self, data, invalid: int = -1):

		edges = np.array(self.class_borders(return_pairs = False))

		idxs = np.searchsorted(edges, data) - 1
		idxs[(idxs < 0) | (idxs >= len(edges)-1)] = invalid

		return idxs

	@classmethod
	def from_data \
	(
		cls,
		data,
		width = 0.05,
		width_type = Tag.Relative,
		oversampling = 7,
		weight_cutoff = 1e-2,
		rel_tolerance = 1e-6,
		max_iter = 800,
		verbose = True
	):

		# raise Exception('This method is obsolete')

		# constants
		#
		max_overlap = 2		# in [average peak width]

		min_val, max_val = np.min(data), np.max(data)

		#------------------------------------------------------------
		# initialization

		if max_val == 0 or np.abs(1 - min_val/max_val) < 1e-12:

			# all the points are identical -> generate a standard single peak

			f_initial = cls([(1, min_val, 1)])

			return f_initial
		else:

			dw = 0.25*(max_val - min_val)

			X, d = density(data, width, width_type, oversampling = oversampling)
			px, py, pw = peaks(X, d, (min_val - dw, max_val + dw))

			f_initial = cls(zip(py, px, pw))

		#------------------------------------------------------------
		# fitting to the estimated density

		def to_numpy(gm, flatten = True):

			# ret: [A..., mu..., sigma...]

			q = np.vstack([tuple(arg) for arg in gm.args]).T
			return q.ravel() if flatten else q

		from_numpy = lambda a: cls \
		(
			a.reshape(3, -1).T
		)

		def gm_grad(gm, x):

			a, mu, sigma = to_numpy(gm, flatten = False)

			G = gm(x)
			g = np.vstack(tuple(gm.TArg(a, mu, sigma)(c, True) for c in x)).T
			r = np.vstack(tuple((c - mu)/sigma for c in x)).T

			u = np.hstack \
			(
				(
					sigma*(g - G).T,
					a*(g*r).T,
					a*(g*r*r - G).T
				)
			)

			return GaussianMixture.C0*u/gm.norm()

		loss = lambda p: np.mean \
		(
			(d - from_numpy(p)(X))**2
		)

		def loss_grad(p):

			G = from_numpy(p)

			return -2*np.mean \
			(
				(d - G(X))*gm_grad(G, X).T,
				axis = -1
			)

		#---------------------------------------------

		valid = False

		p = to_numpy(f_initial)
		p_refined = p

		f_refined = f_initial

		while not valid:

			valid = True

			res = fmin_bfgs \
			(
				loss,
				p_refined,
				fprime = loss_grad,
				gtol = norm(loss_grad(p_refined))*rel_tolerance,
				maxiter = max_iter,
				disp = verbose,
				full_output = True
			)

			warnflag = res[6]

			# gradient and/or function are not changing -> trying another method
			#
			if warnflag == 2:

				if verbose:
					print(highlighted('BFGS algorithm failed. Trying Nelder-Mead method instead.'))

				res = fmin \
				(
					loss,
					p_refined,
					ftol = loss(p_refined)*rel_tolerance,
					disp = verbose,
					full_output = True
				)

			p_refined = res[0]

			f_refined = from_numpy(p_refined)

			f_refined.args = sorted(f_refined.args, key = lambda a: a.mu)

			if len(f_refined.args) == 1:

				if verbose:
					print(highlighted('A single peak left'))

				break

			# checking for invalid/too close peaks:

			A_max = f_refined.max_value()

			overlapping_idxs = set \
			(
				idx
				for idx, (dm, s) in enumerate \
				(
					f_refined.args[k+1] - f_refined.args[k]
					for k in range(len(f_refined.args)-1)
				)
				if abs(dm) <= max_overlap*s
			)

			small_amplitude_idxs = set \
			(
				idx
				for idx, arg in enumerate(f_refined)
				if arg.A < 0 or (weight_cutoff is not None and arg.A < A_max*weight_cutoff)
			)

			invalid_mu_idxs = set \
			(
				idx
				for idx, arg in enumerate(f_refined)
				if arg.mu < min_val - arg.sigma or arg.mu > max_val + arg.sigma
				#if arg.mu < min_val or arg.mu > max_val
			)

			STD_TOLERANCE = 0.5

			invalid_std_idxs = set \
			(
				idx
				for idx, arg in enumerate(f_refined)
				if arg.sigma <= 0 \
				or arg.sigma < STD_TOLERANCE*0.5*(abs(arg.mu*width) if width_type is Tag.Relative else width)
			)

			invalid_idxs = overlapping_idxs | small_amplitude_idxs | invalid_mu_idxs | invalid_std_idxs

			valid = (len(invalid_idxs) == 0)

			if not valid:

				if len(f_refined.args) == len(invalid_idxs):

					if verbose:
						print(error('All the peaks are invalid!'))

					# leave a single peak
					#
					p_refined = np.array([1, data.mean(), data.std()])

				else:

					# first, merge overlapping peaks
					#
					for idx in overlapping_idxs:

						# (`f_refined.args[idx]` will be discarded)
						#
						f_refined.args[idx+1] = f_refined.args[idx] + f_refined.args[idx+1]

					f_refined.args = \
					[
						arg
						for idx, arg in enumerate(f_refined)
						if idx not in invalid_idxs
					]

					p_refined = to_numpy(f_refined)

		if verbose:
			print()
			print('{}\n\n -> \n\n{}'.format(f_initial, f_refined))

			print()
			print('grad: {} -> {}'.format(norm(loss_grad(p)), norm(loss_grad(p_refined))))
			print()

		return f_refined
