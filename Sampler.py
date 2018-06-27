import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class Sampler():

	def __init__(self, 
					X=None, 
					Y=None,
					use_overlap=True,
					num_half_batch=4,
					use_whole_data=False,
					shuffle_each_epoch=False):
		X, Y = shuffle(X, Y, random_state=0)
		self.X = X
		self.Y = Y
		self.N = X.shape[0]
		self.b = num_half_batch # number of multi-batch intervals
		self.B = self.b - 1 # number of the overlapped multi-batch intervals
		self.mbs = self.N // self.b  # multi-batch size
		self.first_time = True
		self.sample_start = 0
		self.sample_middle = 1
		self.sample_end   = 2
		self.smaple_i = 0 # start index of interval
		self.sample_h = 0 # middle index of interval
		self.sample_j = 0 # end index of interval
		self.use_whole_data = use_whole_data
		self.shuffle_each_epoch = shuffle_each_epoch

	def overlapped_sample(self):
		'''this is for sampling with overlap
		TODO: (1) currently the overlap portion is half of the batch,
		maybe implementing overlap with different size
		TODO (2): right now there is no non-overlap portion, maybe implement 
		sampling with non overlap part

		'''
		if self.use_whole_data:
			return self.X, self.Y, self.X, self.Y
		
		self.sample_i = self.sample_start * self.mbs
		self.sample_m = self.sample_middle * self.mbs
		self.sample_j = self.sample_end * self.mbs
		
		X_sample = self.X[self.sample_i:self.sample_j]
		Y_sample = self.Y[self.sample_i:self.sample_j]
		X_overlap = self.X[self.sample_m:self.sample_j]
		Y_overlap = self.Y[self.sample_m:self.sample_j]

		self.X_sample = X_sample
		self.Y_sample = Y_sample
		self.X_overlap = X_overlap
		self.Y_overlap = Y_overlap
		
		if self.sample_end == self.b: # reached to the end
			X_prev = self.X[0:self.sample_i]
			Y_prev = self.Y[0:self.sample_i]
			self.shuffle_beginning_and_concat_to_end(X_prev, Y_prev)
			# restart indecied -- permutation
			self.sample_start = 0
			self.sample_middle = 1
			self.sample_end = 2

		self.sample_start = self.sample_start + 1
		self.sample_middle = self.sample_middle + 1
		self.sample_end = self.sample_end + 1

		return self.X_sample, self.Y_sample, self.X_overlap, self.Y_overlap 


	def shuffle_beginning_and_concat_to_end(self, X_prev, Y_prev):
		'''shuffles the beginning protion of the data and concat to the 
		last sample'''
		if self.shuffle_each_epoch:
			X_after, Y_after = shuffle(X_prev, Y_prev,random_state=0)
		else:
			X_after = X_prev
			Y_after = Y_prev
		self.X = np.concatenate( (self.X_sample, X_after) )
		self.Y = np.concatenate( (self.Y_sample, Y_after) )

	def simple_sample(self,sample_size=1000):
		data_idx = np.arange(self.X.shape[0])
		random_idx = np.random.choice(data_idx, (sample_size) )
		X_sample = self.X[ random_idx ]  
		Y_sample = self.Y[ random_idx ]
		self.X_sample = X_sample
		self.Y_sample = Y_sample
		return X_sample, Y_sample





