import numpy as np
import math
from collections import defaultdict
import random
# !pip3 install git+https://github.com/slremy/netsapi --user --upgrade
from netsapi.challenge import * 

from scipy.stats import beta as beta_dist
from scipy.stats import norm as norm_dist
from sklearn.linear_model import SGDClassifier, LogisticRegression
from scipy.optimize import minimize


class OnlineLogisticRegression:
	""" The implementation of online LR for TS is inspired by the link below.
	https://github.com/gdmarmerola/interactive-intro-rl/blob/master/notebooks/ts_for_contextual_bandit.ipynb
	"""
	def __init__(self, lambda_, alpha, n_dim):
		self.lambda_ = lambda_
		self.alpha = alpha
		self.n_dim = n_dim, 
		self.m = np.zeros(self.n_dim)
		self.q = np.ones(self.n_dim) * self.lambda_
		self.w = np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)
		
	def loss(self, w, *args):
		X, y = args
		return 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.sum([np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])
		
	def grad(self, w, *args):
		X, y = args
		return self.q * (w - self.m) + (-1) * np.array([y[j] *  X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)
	
	def get_weights(self):
		return np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)
	
	def fit(self, X, y):
		self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B", options={'maxiter': 20, 'disp':True}).x
		self.m = self.w
		P = (1 + np.exp(1 - X.dot(self.m))) ** (-1)
		self.q = self.q + (P*(1-P)).dot(X ** 2)
				
	def predict_proba(self, X, mode='sample'):
		self.w = self.get_weights()
		if mode == 'sample':
			w = self.w 
		elif mode == 'expected':
			w = self.m 
		proba = 1 / (1 + np.exp(-1 * X.dot(w)))
		return proba


class CustomAgent(object):
	def __init__(self, env, alpha=5, lambda_=1.0):
		self.env = env
		self.action_resolution = 0.1
		self.alpha=5.0
		self.lambda_ = 1.0
		self.actions = self.actionSpace()
		
		self.last_action = (1, 0)
		self.train_flag = True
		self.eps = 0.1
		
		self.FirstActionValue = {}
		for key in self.actions:
			self.FirstActionValue[key] = (1, 3)
		
		self.online_lr = OnlineLogisticRegression(self.lambda_, self.alpha, 3)
		
		self.ContextValue = {}
		for key in self.actions:
			self.ContextValue[key] = 0
		
		self.ActionContextValue = {}
		for key in self.actions:
			self.ActionContextValue[key] = self.ContextValue
		
	def actionSpace(self):
		xy = []
		for x in np.arange(0,1+self.action_resolution,self.action_resolution):
			for y in np.arange(0,1+self.action_resolution,self.action_resolution):
				xy_sum = x + y
				xy_diff = abs(x - y)
#                 remove bad actions in most scenarios
				if xy_sum >= 0.7 and xy_sum <= 1.5 and xy_diff >= 0.5:
					xy.append((x.round(2), y.round(2)))
		return xy
		
	def get_context(self, action, last_action):
		return np.array((abs(action[0]-last_action[0]), abs(action[0]-last_action[0]), abs(action[0]-action[1])))
						
	def choose_action(self, state):
		if state == 1:
			action = self.choose_first_action()
			if self.train_flag == False:
				self.last_action = action
			return action
		
		samples = {}
		ActionValue = self.ActionContextValue[self.last_action]
		for key in ActionValue:
			x = self.get_context(key, self.last_action)
			prob = self.online_lr.predict_proba(x.reshape(1, -1), mode='sample')
			samples[key] = prob
		max_value =  max(samples, key=samples.get)
		if self.train_flag == False:
			self.last_action = max_value
		return max_value    
	
	def update(self,action,reward):
		x = self.get_context(action, self.last_action)
		self.online_lr.fit(x.reshape(1, -1), np.array([reward/150]))        
		
	def choose_first_action(self):
		if self.train_flag == True:
			samples = {}
			for key in self.FirstActionValue:
				samples[key] = np.random.beta(self.FirstActionValue[key][0], self.FirstActionValue[key][1])
			max_value =  max(samples, key=samples.get)
			return max_value
	
	def update_first_action_value(self, action, reward):
		a, b = self.FirstActionValue[action]
		a = a+reward/150
		b = b + 1 - reward/150
		a = 0.001 if a <= 0 else a
		b = 0.001 if b <= 0 else b
		self.FirstActionValue[action] = (a, b)
	
	def choose_action_eval(self, state):
		if state == 1:
			samples = {}
			for key in self.FirstActionValue:
				a = self.FirstActionValue[key][0]
				b = self.FirstActionValue[key][1]
				samples[key] = a / (a + b)
			action =  max(samples, key=samples.get)
			self.FirstActionValue[action] = (0.001, 0.001)
			self.last_action = action
			return action
		samples = {}
		ActionValue = self.ActionContextValue[self.last_action]
		for key in ActionValue:
			x = self.get_context(key, self.last_action)
			prob = self.online_lr.predict_proba(x.reshape(1, -1), mode='expected')
			samples[key] = prob
		max_value =  max(samples, key=samples.get)
		if self.train_flag == False:
			self.last_action = max_value
		return max_value 
			
	def train(self):
		for _ in range(20): #Do not change
			self.env.reset()
			state = 1
			while True:
				action =  self.choose_action(state)
				nextstate, reward, done, _ = self.env.evaluateAction(list(action))
				if math.isnan(reward):
					reward = -1
				print(state, action, reward)
				if state == 1:
					self.update_first_action_value(action, reward)
				else:
					self.update(action,reward)
				self.last_action = action
				state = nextstate
				if done:
					break

	def generate(self):
		best_policy = None
		best_reward = -float('Inf')
		self.train()
		self.train_flag = False
		
		best_reward = 0
		best_policy = []
		for _ in range(1):
			policy = {state: list(self.choose_action_eval(state)) for state in range(1,6)}
			reward = self.env.evaluatePolicy(policy)
			if reward >= best_reward:
				best_policy = policy
				best_reward = reward
		print(best_policy, best_reward)
		
		return best_policy, best_reward 
