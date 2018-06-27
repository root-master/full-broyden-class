import time
import pickle

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--storage', '-m', default=10, help='The Memory Storage')
parser.add_argument('--mini_batch','-minibatch', default=1000,
												help='minibatch size')
parser.add_argument('--num_batch_in_data', '-num-batch',default=4,
        							help='number of batches with overlap')
parser.add_argument('--method', '-method',default='TRUST_REGION',
        	help="""Method of optimization ['LINE_SEARCH','TRUST_REGION']""")
parser.add_argument(
        '--whole_gradient','-use-whole-data', action='store_true',default=False,
        help='Compute the gradient using all data')
parser.add_argument(
        '--use_overlap','-use-overlap', action='store_true',default=True,
        help='Compute y using overlap of multibatches')
parser.add_argument('--max_iter', '-maxiter', default=200,help='max iterations')

# python main.py -num-batch=10 -m=20 
args = parser.parse_args()


minibatch = int(args.mini_batch)
m = int(args.storage)
num_half_batch = int(args.num_batch_in_data)
use_whole_data = args.whole_gradient
use_overlap = args.use_overlap
method = str(args.method)
max_iter = int(args.max_iter)

import input_MNIST_data
from Sampler import Sampler
from Model import Model
from Logger import Logger

from FullBroydenClass import FullBroydenClass

# load data X and Y
data = input_MNIST_data.read_data_sets("./data/", one_hot=True)
X_train = data.train_all.images
Y_train = data.train_all.labels
X_test = data.test.images
Y_test = data.test.labels 

# create a shuffler and sampler instance
sampler = Sampler(  X=X_train, Y=Y_train,
					use_overlap=use_overlap, num_half_batch=num_half_batch,
											use_whole_data=use_whole_data)

if use_whole_data: 
	use_overlap = False

# create a model f(x;w) and L(y,y_;w) instance
model = Model(minibatch=minibatch)

# create a logger instance
logger = Logger(model=model)

# create a quasi-Newton method instance
fbd = FullBroydenClass(quasi_Newton_method='L_BFGS',
						search_method = 'TRUST_REGION',
						model = model,use_overlap=use_overlap)


##################### main loop for learning #################################
start = time.time()
for k in range(max_iter):
	print('-'*60)
	print('iteration: {}' .format(k))

	X, Y, XO, YO = sampler.overlapped_sample()
	# feed data to the model

	if k == 0:
		model.feed_data(X=X, Y=Y, XO=XO, YO=YO)
		# print/save training loss, accuracy
		print('-'*20,' initial values ','-'*20)
		logger.eval_train_performance()
		# print/save test loss, accuracy
		model.feed_data(X=X_test,Y=Y_test)
		logger.eval_test_performance()
	
	model.feed_data(X=X, Y=Y, XO=XO, YO=YO)

	# run one iteration of quasi-Newton optimization method
	fbd.run_one_iteration()

	# print/save final training loss, accuracy
	print('-'*20,' performance metrics ','-'*20)
	logger.eval_train_performance()
	# print/save final test loss, accuracy
	model.feed_data(X=X_test,Y=Y_test)
	logger.eval_test_performance()

end = time.time()
loop_time = end - start
each_iteration_avg_time = loop_time / (k+1)

result_file_path = './results/results_experiment_JUNE' + str(method) + '_m_'\
							+ str(m) + '_n_' + str(num_half_batch) + '.pkl'
if use_whole_data:
	result_file_path = './results/results_experiment_JUNE' + str(method) + \
	'_m_' + str(m) + '_n_2' + '.pkl'

# save results
with open(result_file_path, 'wb') as f: 
	pickle.dump([logger.train_loss_vec, logger.train_accuracy_vec], f)
	pickle.dump([logger.test_loss_vec, logger.test_accuracy_vec], f)
	pickle.dump([loop_time, each_iteration_avg_time], f)


