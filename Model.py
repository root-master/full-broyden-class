import tensorflow as tf
from networks import dim_w, lenet5_model
import numpy as np
from functools import reduce

class Model:
	"""Class for f(x;w) and L(w): 
		- tensorflow graph
		- computation of loss function
		- computation of accuracy
		- computation of gradients
		- computation of y_{k+1} = g_{k+1} - g_{k}
		- update parameters w
	TODO: how to handle asynchronous multi batch gradient gradient computation 
			on multiple GPU?
	TDOD: maybe computing cumulative multi batch gradient using tensors 
	"""
	def __init__(self,
				f = lenet5_model,
				network_input_shape = (None,784),
				network_output_shape = (None, 10),
				weights_shape_dict = dim_w,
				**kwargs ):
		"""
		f: f(x;w) is the function approximator -- feedforward of ANN
		default values for MNIST DATASET -- LeNet-5 netwrok
		"""
		tf.reset_default_graph()

		if weights_shape_dict is None:
			raise Exception('w is not defined')
		if f is None:
			raise Exception('f is not defined')

		self.weights_shape_dict = weights_shape_dict
		n_W = {}
		for key, shape in weights_shape_dict.items():
			n_W[key] = reduce(lambda i, j: i*j, shape)
		self.n_W = n_W
		
		############################ GRAPH ####################################
		############################ y_ = f(x;w) ##############################
		self.f = f
		self.w_initializer = tf.contrib.layers.xavier_initializer()
		self.x = tf.placeholder(tf.float32,shape=network_input_shape,name='x')					
		self.y = tf.placeholder(tf.float32,shape=network_output_shape, name='y')
		
		w = {}
		for key, _ in weights_shape_dict.items():
			w[key] = tf.get_variable(key, shape=weights_shape_dict[key], 
											initializer=self.w_initializer)
		self.w = w

		self.y_ = f(self.x, self.w) # feedforward model
		############################ loss L(y,y_) #############################
		# Softmax loss
		self.loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels = self.y, 
													logits = self.y_) )
		############################ Accuracy #################################
		self.correct_prediction = tf.equal( tf.argmax(self.y_, 1), 
												tf.argmax(self.y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		######################## Gradients ####################################
		grad_w_tf = {}
		for layer, _ in self.w.items():
			grad_w_tf[layer] = tf.gradients(xs=self.w[layer], ys=self.loss)
		self.grad_w_tf = grad_w_tf

		########################## Auxilary w #################################
		aux_w = {}
		for layer, _ in self.w.items():
			name = layer + 'aux_w_'
			aux_w[layer] = tf.get_variable( name=name, 
											shape=w[layer].get_shape(), 
											initializer=self.w_initializer)
		self.aux_w = aux_w

		aux_w_placeholder = {}
		for layer, _ in self.w.items():
			aux_w_placeholder[layer] = tf.placeholder(dtype="float",
											shape=self.w[layer].get_shape())
		self.aux_w_placeholder = aux_w_placeholder

		aux_w_init = {}
		for layer, _ in self.w.items():
			aux_w_init[layer] = aux_w[layer].assign(self.aux_w_placeholder[layer])

		self.aux_w_init = aux_w_init

		########################## Auxilary y_ ################################
		self.aux_output = f(self.x,self.aux_w)
		
		########################## Auxilary loss ##############################
		self.aux_loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels = self.y, 
													logits = self.aux_output) )
		########################## Auxilary grad ##############################
		aux_grad_w = {}
		for layer, _ in self.w.items():
			aux_grad_w[layer] = tf.gradients(xs=self.aux_w[layer], 
													ys=self.aux_loss)
		self.aux_grad_w = aux_grad_w

		
		########################## update w ###################################
		update_w = {}
		update_w_placeholder = {}
		for layer, _ in self.w.items():
			update_w_placeholder[layer] = tf.placeholder(dtype="float",
												shape=self.w[layer].get_shape())
		self.update_w_placeholder = update_w_placeholder

		for layer, _ in self.w.items():
			update_w[layer] = self.w[layer].assign(self.update_w_placeholder[layer])
		self.update_w = update_w

		############################ SESSION ##################################
		self.saver = tf.train.Saver()
		self.init = tf.global_variables_initializer()

		self.session = tf.Session() # start tf session
		self.session.run(self.init) # init variables
		print('-'*60)
		print('neural network model init --> OK')

		############### Numerical Computation Params ##########################
		self.minibatch = 1000 # needs to be an arguemnt
		self.X = None
		self.Y = None
		self.XO = None
		self.YO = None
		############### Update any other kwargs Params ########################
		self.__dict__.update(kwargs)	
	
	def close_tf_session(self):
		self.session.close()

	def dict_of_weight_matrices_to_single_linear_vec(self, x_dict):
		x_vec = np.array([])
		for key in sorted(self.w.keys()):
			matrix = x_dict[key]
			x_vec = np.append(x_vec,matrix.flatten())	
		return x_vec

	def linear_vec_to_dict_of_weight_matrices(self, x_vec):
		x_dict = {}
		id_start = 0
		id_end   = 0
		for key in sorted(w_tf.keys()):
			id_end = id_start + self.n_W[key]
			vector = x_vec[id_start:id_end]
			matrix = vector.reshape(self.weights_shape_dict[key])
			x_dict[key] = matrix
			id_start = id_end
		return x_dict

	def feed_data(self, X=None, Y=None, XO=None, YO=None):
		'''Feeding data to model -- should be done before any computation
		'X' --> input
		'Y' --> labels
		'XO' --> overlap - input
		'YO' --> overlap - output'''
		self.X = X
		self.Y = Y
		self.XO = XO
		self.YO = YO

	def compute_multibatch_tensor(  self, 
									tensor_tf=None, 
									use_overlap = False):
		''' to coumpute a tensor value over multi batch
		TODO: add a tensor to graph accumulate the sum
		ref: https://stackoverflow.com/questions/45987156/tensorflow-average-gradients-over-several-batches
		ref: https://gist.github.com/Multihuntr/b8cb68316842ff68cab3062740a2a730
		'''
		feed_dict = {}
		total = 0

		if use_overlap:
			X = self.XO
			Y = self.YO
		else:
			X = self.X
			Y = self.Y

		num_minibatches_here = X.shape[0] // self.minibatch
		for j in range(num_minibatches_here):
			index_minibatch = j % num_minibatches_here
			# mini batch 
			start_index = index_minibatch     * self.minibatch
			end_index   = (index_minibatch+1) * self.minibatch
			X_batch = X[start_index:end_index]
			Y_batch = Y[start_index:end_index]
			feed_dict.update({	self.x: X_batch,
								self.y: Y_batch})

			value = self.session.run(tensor_tf, feed_dict=feed_dict)
			total = total + value

		total = total * 1 / num_minibatches_here	
		return total

	def compute_multibatch_gradient(self, 
									grad_tf=None, 
									use_overlap = False):
		''' to coumpute the gradient over multi batch
		TODO: add a tensor to graph accumulate this
		ref: https://stackoverflow.com/questions/45987156/tensorflow-average-gradients-over-several-batches
		ref: https://gist.github.com/Multihuntr/b8cb68316842ff68cab3062740a2a730
		'''
		feed_dict = {}
		gw = {}
		
		if use_overlap:
			X = self.XO
			Y = self.YO
		else:
			X = self.X
			Y = self.Y

		num_minibatches_here = X.shape[0] // self.minibatch
		for j in range(num_minibatches_here):
			index_minibatch = j % num_minibatches_here
			# mini batch 
			start_index = index_minibatch     * self.minibatch
			end_index   = (index_minibatch+1) * self.minibatch
			X_batch = X[start_index:end_index]
			Y_batch = Y[start_index:end_index]
			feed_dict.update({	self.x: X_batch,
								self.y: Y_batch})

			gw_list = self.session.run(grad_tf, feed_dict=feed_dict)
			if j == 0:		
				for layer, _ in self.w.items():
					gw[layer] = gw_list[layer][0]
			else:
				for layer, _ in self.w.items():
					gw[layer] = gw[layer] + gw_list[layer][0]

		for layer, _ in self.w.items():
			gw[layer] = gw[layer] * 1 / num_minibatches_here	
		return gw


	# def test_compute_multibatch_gradient(self, grad_tf):
		# '''form: 
		# https://stackoverflow.com/questions/46772685/how-to-accumulate-gradients-in-tensorflow?rq=1
		# '''
		# tvs = tf.trainable_variables()
		# accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
		# zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
		# ## Calls the compute_gradients function of the optimizer to obtain... the list of gradients
		# gvs = tf.gradient(xs=tvs,ys=self.loss)

		# ## Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
		# accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

		# ## Define the training step (part with variable value update)
		# train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])
		# ## The while loop for training
		# while ...:
		# # Run the zero_ops to initialize it
		# sess.run(zero_ops)
		# # Accumulate the gradients 'n_minibatches' times in accum_vars using accum_ops
		# for i in xrange(n_minibatches):
		# 	sess.run(accum_ops, feed_dict=dict(X: Xs[i], y: ys[i]))
		# 	# Run the train_step ops to update the weights based on your accumulated gradients
		# 	sess.run(train_step)

	def eval_w_dict(self):
		w_dict = self.session.run(self.w)
		w_vec = self.dict_of_weight_matrices_to_single_linear_vec(w_dict)
		self.w_vec = w_vec
		return w_dict

	def eval_loss(self, use_overlap=False):
		loss_val = self.compute_multibatch_tensor(  tensor_tf=self.loss,
													use_overlap=use_overlap)
		return loss_val

	def eval_gradient_vec(self, use_overlap=False):
		"""returns gradient, here only for mode='robust-multi-batch' 
		I should modify to consider all other cases"""
		g_dict = self.compute_multibatch_gradient(grad_tf=self.grad_w_tf,
													use_overlap=use_overlap)
		g_vec = self.dict_of_weight_matrices_to_single_linear_vec(g_dict)
		self.g = g_vec
		return g_vec	

	def eval_aux_loss(self, p_vec):
		w_dict = self.eval_w_dict()
		p_dict = self.linear_vec_to_dict_of_weight_matrices(p_vec)
		feed_dict = {}
		for key,_ in self.w.items():
			feed_dict.update({self.aux_w_placeholder[key]: w_dict[key]+p_dict[key] })
		self.session.run(self.aux_w_init,feed_dict=feed_dict)
		loss_new = self.compute_multibatch_tensor(tensor_tf=self.aux_loss)
		return loss_new

	def eval_aux_gradient_vec(self, use_overlap=False):
		# assuming that eval_aux_loss is being called before this function call
		aux_g_dict = self.compute_multibatch_gradient(grad_tf=self.aux_grad_w,
														use_overlap=use_overlap)
		aux_g_vec = self.dict_of_weight_matrices_to_single_linear_vec(aux_g_dict)
		self.aux_g_vec = aux_g_vec
		return aux_g_vec

	def eval_y(self, use_overlap=False):
		new_g = self.eval_aux_gradient_vec(use_overlap = use_overlap)
		if use_overlap:
			old_g = self.eval_gradient_vec(use_overlap=True)
		else:
			old_g = self.g # because we don't want to recompute this gradient
		
		new_y = new_g - old_g
		self.y = new_y
		return new_y

	def eval_accuracy(self):
		accuracy_val = self.compute_multibatch_tensor(tensor_tf=self.accuracy)
		return accuracy_val

	def update_weights(self,p_vec=None):
		w_dict = self.eval_w_dict(sess)
		p_dict = self.linear_vec_to_dict_of_weight_matrices(p_vec)
		feed_dict = {}
		for key,_ in self.w.items():
			feed_dict.update({self.update_w_placeholder[key]: w_dict[key]+p_dict[key] })
		self.session.run(self.update_w, feed_dict=feed_dict)
		print('weights are updated')



		
