import tensorflow as tf
# number of weights and bias in each layer
n_W = {}
dim_w = {}

# network architecture hyper parameters
input_shape = [-1,28,28,1]
n_classes = 10
W0 = 28
H0 = 28

# Layer 1 -- conv
D1 = 1; F1 = 5; K1 = 20; S1 = 1
W1 = (W0 - F1) // S1 + 1
H1 = (H0 - F1) // S1 + 1
conv1_dim = [F1, F1, D1, K1]
conv1_strides = [1,S1,S1,1] 
n_W['1_w_conv'] = F1 * F1 * D1 * K1
n_W['1_b_conv'] = K1 
dim_w['1_w_conv'] = [F1, F1, D1, K1]
dim_w['1_b_conv'] = [K1]

# Layer 2 -- max pool
D2 = K1; F2 = 2; K2 = D2; S2 = 2
W2 = (W1 - F2) // S2 + 1
H2 = (H1 - F2) // S2 + 1
layer2_ksize = [1,F2,F2,1]
layer2_strides = [1,S2,S2,1]

# Layer 3 -- conv
D3 = K2; F3 = 5; K3 = 50; S3 = 1
W3 = (W2 - F3) // S3 + 1
H3 = (H2 - F3) // S3 + 1
conv2_dim = [F3, F3, D3, K3]
conv2_strides = [1,S3,S3,1] 
n_W['2_w_conv'] = F3 * F3 * D3 * K3
n_W['2_b_conv'] = K3 
dim_w['2_w_conv'] = [F3, F3, D3, K3]
dim_w['2_b_conv'] = [K3]

# Layer 4 -- max pool
D4 = K3; F4 = 2; K4 = D4; S4 = 2
W4 = (W3 - F4) // S4 + 1
H4 = (H3 - F4) // S4 + 1
layer4_ksize = [1,F4,F4,1]
layer4_strides = [1,S4,S4,1]


# Layer 5 -- fully connected
n_in_fc = W4 * H4 * D4
n_hidden = 500
fc_dim = [n_in_fc,n_hidden]
n_W['3_w_fc'] = n_in_fc * n_hidden
n_W['3_b_fc'] = n_hidden
dim_w['3_w_fc'] = [n_in_fc,n_hidden]
dim_w['3_b_fc'] = [n_hidden]
# Layer 6 -- output
n_in_out = n_hidden
n_W['4_w_fc'] = n_hidden * n_classes
n_W['4_b_fc'] = n_classes
dim_w['4_w_fc'] = [n_hidden,n_classes]
dim_w['4_b_fc'] = [n_classes]


for key, value in n_W.items():
	n_W[key] = int(value)

def lenet5_model(x,_w):
	# Reshape input to a 4D tensor 
	x = tf.reshape(x, shape = input_shape)
	# LAYER 1 -- Convolution Layer
	conv1 = tf.nn.relu(tf.nn.conv2d(input = x, 
									filter =_w['1_w_conv'],
									strides = [1,S1,S1,1],
									padding = 'VALID') + _w['1_b_conv'])
	# Layer 2 -- max pool
	conv1 = tf.nn.max_pool(	value = conv1, 
							ksize = [1, F2, F2, 1], 
							strides = [1, S2, S2, 1], 
							padding = 'VALID')

	# LAYER 3 -- Convolution Layer
	conv2 = tf.nn.relu(tf.nn.conv2d(input = conv1, 
									filter =_w['2_w_conv'],
									strides = [1,S3,S3,1],
									padding = 'VALID') + _w['2_b_conv'])
	# Layer 4 -- max pool
	conv2 = tf.nn.max_pool(	value = conv2 , 
							ksize = [1, F4, F4, 1], 
							strides = [1, S4, S4, 1], 
							padding = 'VALID')
	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer
	fc = tf.contrib.layers.flatten(conv2)
	fc = tf.nn.relu(tf.matmul(fc, _w['3_w_fc']) + _w['3_b_fc'])
	# fc = tf.nn.dropout(fc, dropout_rate)

	y_ = tf.matmul(fc, _w['4_w_fc']) + _w['4_b_fc']
	return y_
