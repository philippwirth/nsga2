import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random

import fitness
import selector
import simulatedbinarycrossover as sbx
import parameterbasedmutation as pbm


# options
_lambda = 128		# number of parents
mu = 128		# number of children
inputDim = 1		# search space dimension
outputDim = 2		# number of objectives
maxGenerations = 100	# maximum number of generations

# ----------------------------- Individual ---------------------------------#
class Individual:

	def __init__(self, _x):
		self.x = _x
		self.fitness = fitness.f(self.x)

		self.inputDim = self.x.size
		self.outputDim = self.fitness.size
		
		
	def dominates(self, other):
		temp = False
		for i in range(self.outputDim):

			# check whether x[i] <= y[i] for all
			if self.fitness[i] > other.fitness[i]:		
				return False

			# check whether exists x[i] < y[i]
			if self.fitness[i] < other.fitness[i]:		
				temp = True

		return temp
	
# -------------------------------- NSGA2 -----------------------------------#

# initialization
print('Initialization.')

initialSigma = 5.0				# initial sigma and initial mean determine
initialMean = np.zeros(inputDim)		# the distribution of the initial population

currentPop = []					# create an initial population in initialMean +- 2*initialSigma
for i in range(_lambda):
	xi = np.random.rand(inputDim)*4*initialSigma - 2*initialSigma
	currentPop.append(Individual(xi))

# loop
for g in range(0,maxGenerations):

	print('Evolution progress: ', 100*(g+1)/maxGenerations, '%', end='\r')

	# step 1: reproduction
	Q = []
	nChildren = 0
	while(nChildren < mu):
		p0 = currentPop[np.random.randint(mu-1)+1]
		p1 = currentPop[np.random.randint(mu-1)+1]
		x0, x1 = sbx.crossover(p0, p1)			# create two children using simulated binary crossover
		c0 = pbm.mutate(Individual(x0))			# mutate the children using parameter based mutation
		c1 = pbm.mutate(Individual(x1))
		
		Q.append(p0)
		Q.append(p1)
		Q.append(c0)
		Q.append(c1)

		nChildren += 2

	# step 3: selection
	currentPop = selector.selectBest(Q, _lambda)

# end
print('')
print('Final Population:')
for i in range(_lambda):
	print('Individual ', i, ': f(', currentPop[i].x, ') = ', currentPop[i].fitness)
print('Finishing.')
