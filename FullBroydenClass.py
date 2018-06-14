import numpy as np
from numpy.linalg import inv, qr, eig, norm, pinv
import math
from math import isclose, sqrt
from Model import Model

class FullBroydenClass:
	"""Class for quasi-Newton limited memory methods
	There are three optimization methods available to initialize this class:
		FULL_BROYDEN_CLASS = 0 # limited memory Full Broyden Class 
		L_BFGS = 1 # limited memory BFGS --> default
		L_SR1 = 2 # limited memory SR1

	- Trust Region subproblem solver
	- Trust Region one iteration algorithm
	- Line search one iteration algorithm
	
	We do not deal with loops in this class.
	why not? Because in practice there is an outer loop for
	shuffling and sampling data and also learning. 
	e.g. reinforcement learning algorithm like Temporal Difference SARSA, etc.

	- This class do not have access to tensorflow.
	- There should be a "Trainer" class that handles the training loop.
	- There should be a "Model" class (containing tf graph) 
		that has trainable parameters, function appriximator and loss function,
		gradients, can evaluate "y" for quasi-Newton matrix and can update w.  
	
	Class connections:
	- Model <--> FullBroydenClass
	- Model <--> Trainer
	- FullBroydenClass --> Trainer
	- DATA or replay memory --> Sampler
	- Sampler --> Trainer
	"""

	# available methods
	FULL_BROYDEN_CLASS = 0 
	L_BFGS = 1 # --> default
	L_SR1 = 2
	METHOD_NAME = ['FULL_BROYDEN_CLASS','L_BFGS','L_SR1']

	def __init__(	self, 
					method=L_BFGS,
					model = None):

		# quasi-Newton method --> default = L_BFGS
		self.method = method
		# Model class
		if model is None:
			self.model = Model()
		else:
			self.model = model

		self.S = np.array([[]])
		self.Y = np.array([[]])
		self.Psi = np.array([[]])
		self.M = np.array([[]])
		self.g = np.array([]) # gradient vector

		self.phi = 0
		self.phi_vec = np.array([])
		
		self.gamma = 1
		self.gamma_vec = np.array([])
		
		self.delta = 3
		self.delta_vec = np.array([])

		self.P_ll = np.array([[]]) # P_parallel 
		self.g_ll = np.array([[]]) # g_Parallel
		self.g_NL_norm = 0
		self.Lambda_1 = np.array([[]])
		self.lambda_min = 0
		print('init -- Full Broyden Class -- OK')
		print('method = {}' .format(self.METHOD_NAME[self.method]))

	def set_phi(self, phi):
		self.phi = phi

	def add_phi(self, phi):
		self.phi_vec = np.append(self.phi_vec, phi)

	def set_delta(self,delta):
		self.delta = delta

	def add_delta(self,delta):
		self.delta_vec = np.append(self.delta_vec, delta)

	def set_gamma(self,gamma):
		self.gamma = gamma

	def add_gamma(self,delta):
		self.gamma = gamma
		self.gamma_vec = np.append(self.gamma_vec, gamma)

