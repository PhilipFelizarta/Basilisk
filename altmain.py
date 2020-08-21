#This is our multiprocessing loop!
def play_from(board, pos, moves, res, lock):
	import chess
	import chess.uci
	import numpy as np 
	from env import convert, generate_move_dict, fast_onehot

	stockfish = chess.uci.popen_engine("stockfish")
	stockfish.uci()

	s_pos = []
	s_moves = []
	result = 0.0 #Value is not relative
	move_dict = generate_move_dict()

	while not (board.is_game_over() or board.can_claim_draw()):
		stockfish.position(board)
		stock_move = stockfish.go(movetime=100) #Set stockfish to 10 moves per second

		position = convert(board) #convert to position

		s_pos.append(position)
		s_moves.append(fast_onehot(stock_move[0], move_dict))

		board.push(stock_move[0])

	if board.is_checkmate():
		if board.turn:
			result = -1.0
		else:
			result = 1.0
	else:
		result = 0.0

	#use lock
	lock.acquire()
	index = 0
	end = len(s_pos)
	for p in s_pos:
		coeff = np.power(0.99, end - index - 1)
		res.append(result*coeff)
		pos.append(p)
		moves.append(s_moves[index])
		index += 1
	lock.release()

if __name__ == "__main__":
	#Import statements
	import chess
	import chess.uci
	from tqdm import tqdm
	from time import time
	
	from multiprocessing import Process, Manager, Lock

	import numpy as np
	import tensorflow as tf 

	import os
	from tensorflow.keras.models import Model
	from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, AlphaDropout, Lambda, Attention
	from tensorflow.keras.layers import GlobalAveragePooling2D, Multiply, Permute, Reshape
	from tensorflow.keras.optimizers import SGD
	from tensorflow.keras.initializers import glorot_uniform
	from tensorflow.keras.regularizers import l1, l2
	from tensorflow.keras import backend as K
	import env
	import MCTS

	import matplotlib.pyplot as plt
	import matplotlib.animation as animation
	from matplotlib import style

	K.set_image_data_format('channels_last')
	def save_model(model, modelFile, weightFile): #Save model to json and weights to HDF5
		from tensorflow.keras.models import model_from_json


		model_json = model.to_json()
		with open(modelFile, "w") as json_file:
			json_file.write(model_json)
		model.save_weights(weightFile)
		print("Model saved!")

	def load_model(modelFile, weightFile, update=True): #load model from json and HDF5
		from tensorflow.keras.models import model_from_json


		json_file = open(modelFile, 'r')
		load_model_json = json_file.read()
		json_file.close()
		if not update:
			load_model = model_from_json(load_model_json)
		else:
			_, _, _, load_model, _, _ = create_NN()
			load_model.summary()
		load_model.load_weights(weightFile)
		print("Model Loaded!")
		return load_model

	def shuffle_data(a, b, c):
		import numpy as np
		a_s = np.copy(a)
		b_s = np.copy(b)
		c_s = np.copy(c)

		rand_state = np.random.get_state()

		np.random.set_state(rand_state)
		np.random.shuffle(a_s)
		np.random.set_state(rand_state)
		np.random.shuffle(b_s)
		np.random.set_state(rand_state)
		np.random.shuffle(c_s)
		
		return a_s, b_s, c_s


	def stem(X, filters, stage="stem", size=3, lambd=0.00001):
		stem = Conv2D(filters=filters, kernel_size=(size,size), strides=(1,1), padding='same', data_format='channels_last', 
					  name='Conv_' + stage, kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd))(X)
		stem = BatchNormalization(axis=-1, trainable=True)(stem)
		stem = Activation('relu')(stem)
		return stem

	def res_block(X, filters, block, size=3, lambd=0.00001):
		res = Conv2D(filters=filters, kernel_size=(size,size), strides=(1,1), padding='same', data_format='channels_last', 
					name='res_block1_' + block, kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd))(X)
		res = BatchNormalization(axis=-1, trainable=True)(res)
		res = Activation('relu')(res)
		res = Conv2D(filters=filters, kernel_size=(size,size), strides=(1,1), padding='same', data_format='channels_last', 
					name='res_block2_' + block, kernel_initializer=glorot_uniform(), kernel_regularizer=l2(lambd))(res)

		X = Add()([X, res])
		X = BatchNormalization(axis=-1, trainable=True)(X)
		X = Activation('relu')(X)
		return X
	
	#What if we had a tiny NN
	def create_NN():
		image = Input(shape=(8, 8, 17)) #Chess representation
		X = Flatten()(image)
		latent = Dense(64, activation="relu", name="latentspace", kernel_initializer=glorot_uniform())(X)

		#Create state -> hidden state model
		fmodel = Model(inputs=image, outputs=latent)
		
		#Create policy and value function
		p = Dense(1968, activation="softmax", name="fcPolicy", kernel_initializer=glorot_uniform(), input_shape=(64,))
		v = Dense(1, activation="tanh", name="fcValue", kernel_initializer=glorot_uniform(), input_shape=(64,))
		
		#Create hidden state -> policy/value model
		prev_latent = Input(shape=(64,))
		pol = p(prev_latent)
		val = v(prev_latent)
		gmodel = Model(inputs=prev_latent, outputs=[pol, val])
		
		#Create full model
		policy = p(latent)
		value = v(latent)
		model = Model(inputs=image, outputs=[policy, value])
		
		#Create hidden state -> hidden state model for MCTS in hidden space
		p_latent = Input(shape=(64,))
		new_policy = Input(shape=(1968,))
		concat = Concatenate(axis=-1)([p_latent, new_policy])
		
		hidden = Dense(10, activation="relu")(concat)
		new_latent = Dense(64, activation="relu")(hidden)
		hmodel = Model(inputs=[p_latent, new_policy], outputs=new_latent)
		
		#Create updates for hmodel
		true_latent = K.placeholder(shape=(None, 64), name="truelatent")
		latent_MSE = K.sum(K.square(true_latent - new_latent), axis=1)
		L_h = K.mean(latent_MSE)
		
		opt_h = tf.keras.optimizers.Adam(1e-3)
		updates_h = opt_h.get_updates(params=hmodel.trainable_weights, loss=L_h)
		train_fn_h = K.function(inputs=[hmodel.input[0], hmodel.input[1], true_latent],
							   outputs=[L_h], updates=updates_h)
		
		#full model training function
		p1 = K.clip(policy, 1e-6, 1)
		
		#placeholder
		expert = K.placeholder(shape=(None, 1968), name="expert_policy")
		target = K.placeholder(shape=(None, 1), name="target_value")
					
		MSE = K.mean(K.sum(K.square(target- value), axis=-1))
		#y_true = K.clip(expert, K.epsilon(), 1)
		
		Lclip = -K.mean(K.sum(expert * K.log(p1), axis=-1))
		
		loss = Lclip + 1.0*MSE

		#optimizers
		opt = tf.keras.optimizers.SGD(1e-2, momentum=0.9, nesterov=True) #LEARNING RATE
		updates = opt.get_updates(params=model.trainable_weights, loss=loss)
		train_fn = K.function(inputs=[model.input, expert, target], outputs=[Lclip, MSE], updates=updates)
		
		model.summary()
		hmodel.summary()
		
		return model, fmodel, gmodel, hmodel, train_fn, train_fn_h

	#A NN based on attention
	def create_att(dq, dk, N):
		image = Input(shape=(8, 8, 17)) #Chess representation
		X = Flatten()(image)

		#Attention variables
		Q = Dense(int(dq*N), activation="relu")(X)
		k = Dense(int(dk*N), activation="relu")(X)
		V = Dense(int(dk*N), activation="relu")(X)

		#Reshape for matrix multiplication
		Q = Reshape((dq, N))(Q)
		k = Reshape((dk, N))(k)
		V = Reshape((dk, N))(V)

		#Attention mechanisms
		att = Attention()([Q, k , V])

		latent = Flatten()(att)

		#Create state -> hidden state model
		fmodel = Model(inputs=image, outputs=latent)
		
		#Create policy and value function
		p = Dense(1968, activation="softmax", name="fcPolicy", kernel_initializer=glorot_uniform(), input_shape=(dq*N,))
		v = Dense(1, activation="tanh", name="fcValue", kernel_initializer=glorot_uniform(), input_shape=(dq*N,))
		
		#Create hidden state -> policy/value model
		prev_latent = Input(shape=(dq*N,))
		pol = p(prev_latent)
		val = v(prev_latent)
		gmodel = Model(inputs=prev_latent, outputs=[pol, val])
		
		#Create full model
		policy = p(latent)
		value = v(latent)
		model = Model(inputs=image, outputs=[policy, value])
		
		#Create hidden state -> hidden state model for MCTS in hidden space
		p_latent = Input(shape=(dq*N,))
		new_policy = Input(shape=(1968,))
		concat = Concatenate(axis=-1)([p_latent, new_policy])
		
		hidden = Dense(10, activation="relu")(concat)
		new_latent = Dense(dq*N, activation="relu")(hidden)
		hmodel = Model(inputs=[p_latent, new_policy], outputs=new_latent)
		
		#Create updates for hmodel
		true_latent = K.placeholder(shape=(None, dq*N), name="truelatent")
		latent_MSE = K.sum(K.square(true_latent - new_latent), axis=1)
		L_h = K.mean(latent_MSE)
		
		opt_h = tf.keras.optimizers.Adam(3e-3)
		updates_h = opt_h.get_updates(params=hmodel.trainable_weights, loss=L_h)
		train_fn_h = K.function(inputs=[hmodel.input[0], hmodel.input[1], true_latent],
							   outputs=[L_h], updates=updates_h)
		
		#full model training function
		p1 = K.clip(policy, 1e-6, 1)
		
		#placeholder
		expert = K.placeholder(shape=(None, 1968), name="expert_policy")
		target = K.placeholder(shape=(None, 1), name="target_value")
					
		MSE = K.mean(K.sum(K.square(target- value), axis=-1))
		#y_true = K.clip(expert, K.epsilon(), 1)
		
		Lclip = -K.mean(K.sum(expert * K.log(p1), axis=-1))
		
		loss = Lclip + 1.0*MSE

		#optimizer
		opt = tf.keras.optimizers.Adam(3e-3)
		updates = opt.get_updates(params=model.trainable_weights, loss=loss)
		train_fn = K.function(inputs=[model.input, expert, target], outputs=[Lclip, MSE], updates=updates)
		
		model.summary()
		hmodel.summary()
		
		return model, fmodel, gmodel, hmodel, train_fn, train_fn_h



	def create_model(filters, resblocks=5):
		lambd = 0.0001
		
		image = Input(shape=(8, 8, 17)) #Chess representation
		
		#Stem
		X = stem(image, filters)
		
		#Resnet
		for i in range(resblocks):
			X = res_block(X, filters, str(i + 1))
		
		#Create latent representation
		latent = stem(X, 32, stage="latent", size=1)
		latent = Flatten()(latent)
		latent = Dense(64, activation="relu", name="latentspace", kernel_initializer=glorot_uniform())(latent)
		
		#Create state -> hidden state model
		fmodel = Model(inputs=image, outputs=latent)
		
		#Create policy and value function
		p = Dense(1968, activation="softmax", name="fcPolicy", kernel_initializer=glorot_uniform(), input_shape=(64,))
		v = Dense(1, activation="tanh", name="fcValue", kernel_initializer=glorot_uniform(), input_shape=(64,))
		
		#Create hidden state -> policy/value model
		prev_latent = Input(shape=(64,))
		pol = p(prev_latent)
		val = v(prev_latent)
		gmodel = Model(inputs=prev_latent, outputs=[pol, val])
		
		#Create full model
		policy = p(latent)
		value = v(latent)
		model = Model(inputs=image, outputs=[policy, value])
		
		#Create hidden state -> hidden state model for MCTS in hidden space
		p_latent = Input(shape=(64,))
		new_policy = Input(shape=(1968,))
		concat = Concatenate(axis=-1)([p_latent, new_policy])
		
		hidden = Dense(10, activation="relu")(concat)
		new_latent = Dense(64, activation="relu")(hidden)
		hmodel = Model(inputs=[p_latent, new_policy], outputs=new_latent)
		
		#Create updates for hmodel
		true_latent = K.placeholder(shape=(None, 64), name="truelatent")
		latent_MSE = K.sum(K.square(true_latent - new_latent), axis=1)
		L_h = K.mean(latent_MSE)
		
		opt_h = tf.keras.optimizers.Adam(3e-3)
		updates_h = opt_h.get_updates(params=hmodel.trainable_weights, loss=L_h)
		train_fn_h = K.function(inputs=[hmodel.input[0], hmodel.input[1], true_latent],
							   outputs=[L_h], updates=updates_h)
		
		#full model training function
		p1 = K.clip(policy, K.epsilon(), 1)
		
		#placeholder
		expert = K.placeholder(shape=(None, 1968), name="expert_policy")
		target = K.placeholder(shape=(None, 1), name="target_value")
					
		MSE = K.mean(K.sum(K.square(target- value), axis=-1))
		y_true = K.clip(expert, K.epsilon(), 1)
		
		Lclip = -K.mean(K.sum(y_true * K.log(p1), axis=-1))
		
		loss = Lclip + 1.0*MSE

		#optimizer
		opt = tf.keras.optimizers.SGD(3e-3)
		updates = opt.get_updates(params=model.trainable_weights, loss=loss)
		train_fn = K.function(inputs=[model.input, expert, target], outputs=[Lclip, MSE], updates=updates)
		
		model.summary()
		hmodel.summary()
		
		return model, fmodel, gmodel, hmodel, train_fn, train_fn_h

	model, fmodel, gmodel, hmodel, train_fn, _ = create_NN()
	load = load_model("TinyBasilisk.json", "TinyBasilisk.h5", update=False)
	model.set_weights(load.get_weights())

	board = chess.Board()
	manager = Manager()
	lock = Lock()
	#Training hyperparameters
	cycles = 24001
	num_processes = 200
	num_workers = 32
	epochs = 25
	opening_book = ["e2e4", "d2d4", "g1f3", "c2c4"]
	move_dict = env.generate_move_dict()
	p_loss = []
	m_loss = []

	model_image = []
	expert = []
	targ = []
	appends = 0
	#Training Loop
	plt.ion()
	fig, axs = plt.subplots(2)
	for cycle_num in range(cycles):
		model_image = manager.list()
		expert = manager.list()
		targ = manager.list()
		
		processes = []
		
		board.reset()

		#Begin Simulation of Games
		print("Siumulating Games...")
		ply_number = 1
		for m in tqdm(range(num_processes)):
			board.reset()
			ply_number = 2
			uci_index = np.random.choice(4)
			uci_move = opening_book[uci_index]
			board.push_uci(uci_move)


			mv = m % 50
			mv = np.max([mv, 2])

			while not (board.is_game_over() or ply_number >= mv+1):
				tau = 1.0

				if ply_number > mv-1:
					copy = board.copy()             
					p = Process(target=play_from, args=(copy, model_image, expert, targ, lock))
					p.start()
					processes.append(p)

					for idx in range(len(processes)):
						if not processes[idx].is_alive():
							processes[idx].join()
							processes.pop(idx)
							break

				#Use model to generate base of trajectory
				position = env.convert(board)
				position = np.reshape(position, [-1, 8, 8, 17])
				policy, value = model.predict(position)
				policy = np.squeeze(policy)

				#Implement tau
				mask = env.fast_bitmask(board, move_dict)
				policy = np.clip(policy, 1e-6, 1)
				policy = policy * mask
				policy = policy/np.sum(policy)
				index = np.random.choice(1968, p=policy)
				"""
				policy, value, _ = MCTS.muMCTS(board, board.turn, model, simulations=5)
				policy = np.squeeze(policy)
				#Implement tau
				index = np.random.choice(1968, p=policy)
				"""
				board.push_uci(env.MOVES[index])
				ply_number += 1


		#Clean up processes
		print("Cleaning up...")
		for i in range(len(processes)):
			processes[i].join()

		processes = []

		#Load data
		print("Loading Data...")
		t0 = time()
		model_image = np.reshape(np.array(model_image), [-1, 8, 8, 17])
		expert = np.reshape(np.array(expert), [-1, 1968])
		targ = np.reshape(np.array(targ), [-1, 1])
		print("Time: ", time()-t0)


		n = len(targ)

		print("Shuffling Data...")
		mi, e, ta = shuffle_data(model_image, expert, targ)
		splits = int(len(ta)/1024)
		exp_mi = np.array_split(mi, splits)
		exp_e = np.array_split(e, splits)
		exp_ta = np.array_split(ta, splits)


		print("Training...")
		pe = 0
		mse = 0
		for epoch in range(epochs):
			for k in range(splits):
				loss = train_fn([exp_mi[k], exp_e[k], exp_ta[k]])
				pe += loss[0]
				mse += loss[1]

		p_loss.append(pe/(splits*epochs))
		m_loss.append(mse/(splits*epochs))
		axs[0].plot(p_loss)
		axs[1].plot(m_loss)
		plt.draw()
		plt.pause(0.0001)
		print("Iteration: ", cycle_num, ": policy_loss: ", pe/(splits*epochs), " value loss: ", mse/(splits*epochs),
			" Data: ", len(ta))

		if cycle_num % 2 == 1:
			save_model(model, "TinyBasilisk.json", "TinyBasilisk.h5")
			save_model(fmodel, "TinyBasiliskf.json", "TinyBasiliskf.h5")
			save_model(gmodel, "TinyBasiliskg.json", "TinyBasiliskg.h5")
			save_model(hmodel, "TinyBasiliskh.json", "TinyBasiliskh.h5")

	plt.show(block=True)