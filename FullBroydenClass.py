import numpy as np
from numpy.linalg import inv, qr, norm, pinv
from scipy.linalg import eig, eigvals

import math
from math import isclose, sqrt
from Model import Model

QUASI_NEWTON_METHODS = ['L_FULL_BROYDEN_CLASS','L_BFGS','L_SR1']
SEARCH_METHODS = ['LINE_SEARCH','TRUST_REGION']

def enqueue(Z,new_val):
	if Z.size == 0:
		Z = new_val.reshape(-1,1)
		return Z
	Z = np.concatenate( (Z,new_val.reshape(-1,1)), axis=1)
	return Z

def dequeue(Z):
	return np.delete(Z, obj=0, axis=1)


class FullBroydenClass:
	"""Class for quasi-Newton limited memory methods
	There are three optimization methods available to initialize this class:
		'L_FULL_BROYDEN_CLASS' # limited memory Full Broyden Class 
		'L_BFGS' # limited memory BFGS --> default
		'L_SR1' # limited memory SR1

	- Trust Region subproblem solver
	- Trust Region one iteration algorithm
	- Line search one iteration algorithm
	
	We do not deal with loops in this class.
	why not? Because in practice there is an outer loop for
	shuffling and sampling data and also learning. 
	e.g. reinforcement learning algorithm like Temporal Difference SARSA, etc.
	
	Class connections:
	- Model <--> FullBroydenClass
	- Model <--> Trainer -- main loop
	- FullBroydenClass --> Trainer -- main loop
	- DATA or replay memory --> Sampler
	- Sampler --> Trainer -- main loop
	"""

	def __init__(	self, 
					quasi_Newton_matrix='L_BFGS',
					search_method = 'TRUST_REGION',
					model = None,
					use_overlap=True,
					**kwargs):

		if not quasi_Newton_matrix in QUASI_NEWTON_METHODS:
			raise Exception("""valid methods: ['L_FULL_BROYDEN_CLASS','L_BFGS','L_SR1']""") 
		if not search_method in SEARCH_METHODS:
			raise Exception("""valid methods: ['LINE_SEARCH','TRUST_REGION']""") 

		# quasi-Newton method --> default = L_BFGS
		self.quasi_Newton_matrix = quasi_Newton_matrix
		# search method --> default = TRUST_REGION
		self.search_method = search_method
		
		# Model class
		if model is None:
			self.model = Model()
		else:
			self.model = model
		self.use_overlap = use_overlap

		# ['common','general-eig-problem']
		self.find_gamma_method = 'general-eig-problem'

		self.m = 10 # memory size
		# Compact representation matrices
		self.S = np.array([[]])
		self.Y = np.array([[]])
		self.Psi = np.array([[]])
		self.M = np.array([[]])
		self.g = np.array([]) # gradient vector

		self.phi = 0 # phi parameter in Full Broyden Class update
		self.all_phi_vec = np.array([])
		self.phi_vec = np.array([])
		
		self.gamma = 1 # B0 = gamma * I
		self.gamma_vec = np.array([])
		
		self.delta = 3 # trust region initial radius
		self.delta_vec = np.array([])

		self.P_ll = np.array([[]]) # P_parallel 
		self.g_ll = np.array([[]]) # g_Parallel
		self.g_NL_norm = 0
		self.Lambda_1 = np.array([[]])
		self.lambda_min = 0

		self.first_iteration = True
		self.iter = 0
		
		self.__dict__.update(kwargs)
		print('-'*60)
		print('init -- Limited Full Broyden Class -- OK')
		print('Quasi-Newton Matrix: ', self.quasi_Newton_matrix)
		print('Quasi-Newton Search: ', self.search_method)
		print('-'*60)

	def set_phi(self, phi):
		self.phi = phi

	def set_delta(self,delta):
		self.delta = delta

	def add_delta(self,delta):
		self.delta_vec = np.append(self.delta_vec, delta)

	def set_gamma(self,gamma):
		self.gamma = gamma

	def add_gamma(self,delta):
		self.gamma = gamma
		self.gamma_vec = np.append(self.gamma_vec, gamma)

	def update_phi_vec(phi):
		if self.phi_vec.size < self.m:
			self.phi_vec = np.append( self.phi_vec, phi )
		else:
			self.phi_vec = np.delete( self.phi_vec, 0 ) 
			self.phi_vec = np.append( self.phi_vec, phi )

		self.all_phi_vec = np.append(self.all_phi_vec, phi)
	
	def backtracking_line_search(self):
		alpha = 1
		rho_ls = 0.9
		c1 = 1E-4
		BLS_COND = False
		g = self.g
		p = -g

		while not BLS_COND:
			new_f = self.model.eval_aux_loss(p_vec=alpha*p)
			old_f = self.model.eval_loss()
			lhs = new_f
			rhs = old_f + c1 * alpha * p @ g
			BLS_COND = lhs <= rhs
			
			if BLS_COND:
				print('Backtracking line search satisfied for alpha = {0:.4f}' \
																.format(alpha))
			if alpha < 0.1:
				print('WARNING! Backtracking line search did not work')
				break
			alpha = alpha * rho_ls
		return alpha*p

	def phi_bar_func(self,sigma,delta):
		g_ll = self.g_ll
		Lambda_1 = self.Lambda_1
		g_NL_norm = self.g_NL_norm

		if np.isclose( -Lambda_1, sigma ).any():
			phi_bar = - 1 / delta
			return phi_bar

		u = sum( (g_ll ** 2) / ((Lambda_1 + sigma) ** 2) ) + \
								(g_NL_norm ** 2) / ( (gamma + sigma) ** 2)
		v = sqrt(u) 

		phi_bar = 1 / v - 1 / delta
		return phi_bar

	def phi_bar_prime_func(self,sigma):
		g_ll = self.g_ll
		Lambda_1 = self.Lambda_1
		g_NL_norm = self.g_NL_norm
		gamma = self.gamma

		u = sum( g_ll ** 2 / (Lambda_1 + sigma) ** 2 ) + \
										g_NL_norm ** 2 / (gamma + sigma) ** 2

		u_prime = sum( g_ll ** 2 / (Lambda_1 + sigma) ** 3 ) + \
										g_NL_norm ** 2 / (gamma + sigma) ** 3
		phi_bar_prime = u ** (-3/2) * u_prime

		return phi_bar_prime

	def solve_newton_equation_to_find_sigma(self,delta):
		# tolerance
		tol = 1E-2
		lambda_min = self.lambda_min
		g_ll = self.g_ll
		Lambda_1 = self.Lambda_1
		g_NL_norm = self.g_NL_norm
		gamma = self.gamma

		if self.phi_bar_func( max(0,-lambda_min), delta) < 0:
			sigma_hat = np.max( abs( g_ll ) / delta - Lambda_1 )
			sigma_hat = max(sigma_hat , (g_NL_norm / delta - gamma) ) 
			sigma = max( 0, sigma_hat)
			counter = 0
			while( abs( self.phi_bar_func(sigma,delta) ) > tol ):
				phi_bar = self.phi_bar_func(sigma,delta)
				phi_bar_prime = self.phi_bar_prime_func(sigma)
				sigma = sigma - phi_bar / phi_bar_prime
				counter += 1
				if counter > 1000:
					print('had to break newton solver')
					break
			sigma_star = sigma
		elif lambda_min < 0:
			sigma_star = - lambda_min
		else:
			sigma_star = 0

		return sigma_star

	def form_Psi(self):
		S = self.S
		Y = self.Y
		gamma = self.gamma

		if self.quasi_Newton_matrix in ['L_FULL_BROYDEN_CLASS','L_BFGS']:
			Psi = np.concatenate( (gamma*S, Y) ,axis=1)
		elif self.quasi_Newton_matrix == 'L_SR1':
			Psi = Y - gamma * S
		self.Psi = Psi
		return
	
	def eval_reduction_ratio(self,p):
		new_f = model.eval_aux_loss(p_vec=p)
		old_f = model.eval_loss()

		ared = old_f - new_f

		S = self.S
		P_ll = self.P_ll
		Lambda_1 = self.Lambda_1
		gamma = self.gamma
		g = self.g

		if S.size is not 0:
			p_ll = P_ll.T @ p
			p_NL_norm = sqrt ( abs( norm(p) ** 2 - norm(p_ll) ** 2 ) )
			p_T_B_p = sum( Lambda_1 * p_ll ** 2)  + gamma * p_NL_norm ** 2
			pred =  - (g @ p  + 1/2 * p_T_B_p)
		else:
			pred =  - 1/2 * g @ p
		
		rho = ared / pred		
		return rho

	def update_S_Y(self,new_s_val,new_y_val):
		Stmp = self.S
		Ytmp = self.Y
		m = self.m
		num_columns_S = Stmp.shape[1]
		num_columns_Y = Stmp.shape[1]
		assert num_columns_S is num_columns_Y, "dimention of S and Y doesn't match"
		if num_columns_S < m:
			Stmp = enqueue(Stmp,new_s_val)
			Ytmp = enqueue(Ytmp,new_y_val)
		else:
			Stmp = dequeue(Stmp)
			Stmp = enqueue(Stmp,new_s_val)
			Ytmp = dequeue(Ytmp)
			Ytmp = enqueue(Ytmp,new_y_val)

		self.S = Stmp
		self.Y = Ytmp
		return 

	def update_M(self):
		if self.quasi_Newton_matrix == 'L_BFGS':
			self.update_M__L_BFGS()
		elif self.quasi_Newton_matrix == 'L_FULL_BROYDEN_CLASS':
			self.update_M__L_FULL_BROYDEN_METHOD()
		elif self.quasi_Newton_matrix == 'L_SR1':
			self.update_M__L_SR1()


	def update_M__L_BFGS(self):
		S = self.S
		Y = self.Y
		gamma = self.gamma

		S_T_Y = S.T @ Y
		L = np.tril(S_T_Y,k=-1)
		U = np.tril(S_T_Y.T,k=-1).T
		D = np.diag( np.diag(S_T_Y) )

		M = - inv( np.block([ 	[gamma * S.T @ S ,	L],
							[    L.T,		       -D] ]) )
		M = (M + M.T) / 2
		self.M = M
		return

	def update_M__L_FULL_BROYDEN_METHOD(self):
		gamma = self.gamma
		S = self.S
		Y = self.Y
		num_columns_S = S.shape[1]
		phi_vec = self.phi_vec

		for k in range(num_columns_S):		
			if k == 0:
				fi = self.phi_vec[0] 
				s0 = S[:,0]
				y0 = Y[:,0]
				alfa = - (1 - fi) / (gamma * s0.T @ s0)
				beta = - fi / (y0.T @ s0)
				deta = (1 + fi * (gamma * s0.T @ s0 )/(y0.T @ s0)) / (y0.T @ s0)
				M = np.array([[alfa, beta],
							  [beta, deta]]) 
			else:
				fi = phi_vec[k]
				sk = S[:,k]
				yk = Y[:,k]
				Psi_old = np.concatenate( (gamma*S[:,:k], Y[:,:k]) , axis=1)
				peta = M @ Psi_old.T @ sk.reshape(-1,1)
				s_T_B_s = gamma*sk.T @ sk + (sk.reshape(-1,1).T @ Psi_old) @ peta
				alfa = - (1 - fi) / (s_T_B_s)
				beta = - fi / (yk.T @ sk)
				deta = ( 1 + fi * (s_T_B_s)/(yk.T @ sk) ) / (yk.T @ sk)
				M = np.block([	[M+alfa*peta@peta.T,	alfa*peta,	beta*peta ],
								[alfa*peta.T,			alfa,		beta],
								[beta*peta.T,			beta,		deta]]) 

		# make sure M is symmetric
		M = (M + M.T) / 2
		self.M = M
		return


	def update_M__L_SR1(self):
		S = self.S
		Y = self.Y
		gamma = self.gamma

		S_T_Y = S.T @ Y
		L = np.tril(S_T_Y,k=-1)
		U = np.tril(S_T_Y.T,k=-1).T
		D = np.diag( np.diag(S_T_Y) )
		M = inv(D + L + L.T - gamma * S.T @ S )
		# make sure M is symmetric
		M = (M + M.T) / 2
		self.M = M
		return

	def find_gamma_common(self):
		sk = self.S[:,-1]
		yk = self.Y[:,-1]
		gamma = (yk.T @ yk) / (sk.T @ yk)
		self.gamma = gamma
		return gamma

	def find_gamma_L_BFGS_general_eig(self):
		S = self.S
		Y = self.Y
		S_T_Y = S.T @ Y
		S_T_S = S.T @ S
		L = np.tril(S_T_Y,k=-1)
		U = np.tril(S_T_Y.T,k=-1).T
		D = np.diag( np.diag(S_T_Y) )

		H = L + D + L.T
		eigen_values_general_problem = eigvals(H, S_T_S)
		eig_min = min(eigen_values_general_problem)
		if eig_min < 0:
			print('no need for safe gaurding')
			gamma = self.find_gamma_common()
			gamma = max( 1, gamma )
		else:
			gamma = 0.9 * eig_min
		self.gamma = gamma

	def find_gamma_L_SR1_general_eig(self):
		pass		

	def find_gamma(self):
		if self.S.size == 0:
			self.gamma = 1
			print('gamma = ', gamma)
			return self.gamma

		if self.find_gamma_method == 'common':
			gamma = self.find_gamma_common()
			print('gamma = ', gamma)
			return gamma
		
		if self.find_gamma_method == 'general-eig-problem':
			if self.quasi_Newton_matrix == 'L_BFGS':
				gamma= self.find_gamma_L_BFGS_general_eig()
				self.gamma = gamma
				print('gamma = ', gamma)
				return gamma

	def satisfy_curvature_condition(self,p):
		BAD_COND = True
		alpha = 1
		while BAD_COND:
			if alpha < 0.1:
				print('could not fix the bad curvature condition')
				return alpha
				break
			s = alpha * p
			new_f = self.model.eval_aux_loss(p_vec=alpha*p)
			y = self.model.eval_aux_gradient_vec()
			BAD_COND = s.T @ y <= 0
			alpha = 0.9 * alpha
		print('curvature condition satisfied for alpha', alpha)
		return alpha

	def satisfy_wolfe_condition(self, p):
		g = self.g
		alpha = 1
		rho_ls = 0.9
		c1 = 1E-4
		c2 = 0.9
		WOLFE_COND_1 = False
		WOLFE_COND_2 = False
		while not ( WOLFE_COND_1 and WOLFE_COND_2): 
			new_f = self.model.eval_aux_loss(p_vec=alpha*p)
			old_f = self.model.eval_loss()
			lhs = new_f
			rhs = old_f + c1 * alpha * p @ g
			WOLFE_COND_1 = lhs <= rhs
			if WOLFE_COND_1:
				print('WOLFE_COND_1 SATISFIED')
			else:
				print('WOLFE_COND_1 NOT SATISFIED')

			new_g = self.model.eval_aux_gradient_vec()
			lhs = new_g @ p
			rhs = c2 * g @ p
			WOLFE_COND_2 = lhs >= rhs
			if WOLFE_COND_2:
				print('WOLFE_COND_2 SATISFIED')
			else:
				print('WOLFE_COND_2 NOT SATISFIED')

			if WOLFE_COND_1 and WOLFE_COND_2:
				print('WOLFE CONDITIONS SATISFIED')
				print('alpha = {0:.4f}' .format(alpha))
			
			if alpha < 0.1:
				print('WARNING! Wolfe Condition did not satisfy')
				break
			alpha = alpha * rho_ls
		return alpha

	def trust_region_subproblem_solver(self):
		# size of w = g.size
		delta = self.delta
		g = self.g
		n = g.size
		S = self.S
		Y = self.Y		
		self.form_Psi()
		Psi = self.Psi
		gamma = self.gamma
		M = self.M

		Q, R = qr(Psi, mode='reduced')

		# check if Psi is full rank or not
		if np.isclose(np.diag(R),0).any():
			rank_deficieny = True
			# find zeros of diagonal of R
			rank_deficient_idx = np.where( np.isclose(np.diag(R),0))[0]
			# deleting the rows of R with a 0 diagonal entry (r * k+1)
			R_cross = np.delete( R, obj=rank_deficient_idx, axis=0 )
			# deleting the columsn of Psi with a 0 diagonal entry on R (n * r)
			Psi_cross = np.delete( Psi, obj=rank_deficient_idx, axis=1 )
			# deleting the rows and columns of R with a 0 diagonal entry (r * r)
			R_cross_cross = np.delete( R_cross, obj=rank_deficient_idx, axis=1 )
			# (n * r)
			Q_hat = Psi_cross @ inv(R_cross_cross)
			# (r * r)
			R_M_R_T = R_cross @ M @ R_cross.T
		else:
			rank_deficieny = False
			R_M_R_T = R @ M @ R.T

		eigen_values, eigen_vectors = eig( R_M_R_T )
		# make sure eigen values are real
		eigen_values = eigen_values.real
		eigen_vectors = eigen_vectors.real

		# sorted eigen values
		idx = eigen_values.argsort()
		eigen_values_sorted = eigen_values[idx]
		eigen_vectors_sorted = eigen_vectors[:,idx]

		Lambda_hat = eigen_values_sorted
		V = eigen_vectors_sorted

		Lambda_1 = gamma + Lambda_hat

		lambda_min = min( Lambda_1.min(), gamma )

		if rank_deficieny:
			P_ll = Psi_cross @ inv(R_cross_cross) @ V
		else:
			P_ll = Psi @ inv(R) @ V # P_parallel 
		g_ll = P_ll.T @ g	# g_Parallel
		g_NL_norm = sqrt ( abs( norm(g) ** 2 - norm(g_ll) ** 2 ) )

		P_ll = self.P_ll
		g_ll = self.g_ll
		g_NL_norm = self.g_NL_norm
		Lambda_1 = self.Lambda_1
		lambda_min = self.lambda_min

		sigma = 0

		if lambda_min > 0 and self.phi_bar_func(0,delta) >= 0:
			sigma_star = 0
			tau_star = gamma
			# Equation (11) of SR1 paper
			p_star = - 1 / tau_star * \
				( g - Psi @ inv( tau_star * inv(M) + Psi.T @ Psi ) @ (Psi.T @ g) )
		elif lambda_min <= 0 and self.phi_bar_func(-lambda_min, delta) >= 0:
			sigma_star = -lambda_min
			if rank_deficieny:
				if ~isclose(sigma_star, -gamma):
					p_star = - Psi_cross @ inv(R_cross_cross) @ U * pinv( np.diag(Lambda_1 + sigma_star) ) @ g_ll - 1 / ( gamma + sigma_star) * ( g - ( Psi_cross @ inv(R_cross_cross) ) @ inv(R_cross_cross).T @ ( Psi_cross.T @ g ) )
				else:
					p_star = - Psi_cross @ inv(R_cross_cross) @ U * inv( np.diag(Lambda_1 + sigma_star) ) @ g_ll

			else:
				# Equation (13) of SR1 paper
				if ~isclose(sigma_star, -gamma):
					p_star = - Psi @ inv(R) @ U * pinv( np.diag(Lambda_1 + sigma_star) ) @ g_ll - 1 / ( gamma + sigma_star) * ( g - ( Psi @ inv(R) ) @ inv(R).T @ ( Psi.T @ g ) )
				else:
					p_star = - Psi @ inv(R) @ U * inv( np.diag(Lambda_1 + sigma_star) ) @ g_ll
			###############???????????????????#########################
			# so-called hard-case
			if lambda_min < 0:
				p_star_hat = p_star.copy()
				# Equation (14) of SR1 paper
				# check if lambda_min is Lambda_1[0]
				if isclose( lambda_min, Lambda_1.min()):
					u_min = P_ll[:,0].reshape(-1,1)
				else:
					for j in range(Lambda_1.size+2):
						e = np.zeros((n,1))
						e[j,0] = 1.0
						u_min = e - P_ll @ P_ll.T @ e
						if ~isclose( norm(u_min), 0.0):
							break
				# find alpha in Eq (14)
				# solve a * alpha^2 + b * alpha + c = 0  
				a = norm(u_min) ** 2
				b = 2 * norm(u_min) * norm(p_star_hat)
				c = norm(p_star_hat) - delta ** 2
				
				alpha_1 = -b + sqrt(b ** 2 - 4 * a * c) / (2 * a)
				alpha_2 = -b - sqrt(b ** 2 - 4 * a * c) / (2 * a)
				alpha = alpha_1
				
				p_star = p_star_hat + alpha * u_min 
		else:
			sigma_star = self.solve_newton_equation_to_find_sigma(delta)
			tau_star = sigma_star + gamma
			# Equation (11) of SR1 paper
			p_star = - 1 / tau_star * \
				( g - Psi @ inv(tau_star * inv(M) + Psi.T @ Psi ) @ (Psi.T @ g))

		return p_star


	def run_one_iteration(self):
		if self.search_method == 'TRUST_REGION':
			self.trust_region_algorithm()

		elif self.search_method == 'LINE_SEARCH':
			self.line_search_algorithm()


	def trust_region_algorithm(self):
		# eta value in Book's trust-region algorithm 6.2 
		eta = 0.9 * 0.001 # eta \in (0,0.001)
		tolerance = 1E-5

		g = self.model.eval_gradient_vec()
		self.g = g
		norm_g = norm(g)	
		print('norm of g = {0:.4f}' .format(norm_g))
		self.norm_g = norm_g

		if norm_g < tolerance:
			print('-'*60)
			print('gradient is smaller than tolerance')

		if self.first_iteration is True:
			p = self.backtracking_line_search()			
			# we should call this function everytime before 
			# evaluation of aux gradient
			new_loss = self.model.eval_aux_loss(p_vec=p) 
			new_y = self.model.eval_y(use_overlap=self.use_overlap)
			new_s = p
			if new_s.T @ new_y <= 0 and self.quasi_Newton_matrix == 'L_BFGS':
				print('curvature condition did not satisfy for L_BFGS ==> danger zone') 
				alpha = self.satisfy_curvature_condition(p)
				# alpha = self.satisfy_wolfe_condition(p)
				new_s = alpha * p
				new_loss = self.model.eval_aux_loss(p_vec=alpha * p) 
				new_y = self.model.eval_y(use_overlap=self.use_overlap)

			self.update_S_Y(new_s,new_y)
			self.update_M()

			self.iter += 1
			self.first_iteration = False
			self.model.update_weights(p_vec=p)
			return
		
		gamma = self.find_gamma()
		p = self.trust_region_subproblem_solver()		
		rho = self.eval_reduction_ratio(p)

		new_y = self.model.eval_y(use_overlap=self.use_overlap)
		new_s = p
		if new_s.T @ new_y <= 0 and self.quasi_Newton_matrix == 'L_BFGS':
			print('s * y <= 0 ==> danger zone') 
		self.update_S_Y(new_s,new_y)
		self.update_M()
		self.iter += 1

		if rho > eta:			
			self.model.update_weights(p_vec=p)
		else:
			print('-'*30)
			print('No update in this iteration')

		if rho > 3/4:
			if norm(new_s) < 0.8 * self.delta:
				self.delta = self.delta
			else:
				print('expanding trust region radius')
				self.delta = 2 * self.delta
		elif rho > 0.1 and rho < 3/4:
			self.delta = self.delta
		else:
			print('shrinking trust region radius')
			self.delta = 0.5 * self.delta
		return

	def line_search_algorithm():
		pass






