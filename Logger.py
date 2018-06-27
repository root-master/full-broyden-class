class Logger():
	
	def __init__(self, model=None):
		# results -- loss and accuracy
		self.train_loss_vec = []
		self.train_accuracy_vec = []
		self.test_loss_vec = []
		self.test_accuracy_vec = []
		self.model = model


	def eval_test_performance(self):
		test_loss = self.model.eval_loss()
		test_accuracy = self.model.eval_accuracy()
		self.test_loss_vec.append( test_loss )
		self.test_accuracy_vec.append( test_accuracy )
		print('TEST ---- loss: {0:.4f}, accuracy: {1:.4f}' \
						.format(test_loss, test_accuracy))


	def eval_train_performance(self):
		train_loss = self.model.eval_loss()
		train_accuracy = self.model.eval_accuracy()
		self.train_loss_vec.append( train_loss )
		self.train_accuracy_vec.append( train_accuracy )
		print('TRAIN --- loss: {0:.4f}, accuracy: {1:.4f}' \
						.format(train_loss, train_accuracy))



