import numpy as np

crossoverEta = 2.0

def crossover(p0, p1):

	n = p0.inputDim		# get dimension of search vector
	c0 = np.zeros(n)	# initialize vectors to store children
	c1 = np.zeros(n)

	for i in range(n):
		# compute spread factor beta according to a polynomial distribution
		u = np.random.rand()
		if u <= 0.5:
			beta = pow(2*u, 1/(crossoverEta+1))
		else:
			beta = pow(1/(2*(1-u)), 1/(crossoverEta+1))

		# use spread factor to compute child solutions
		m = min(p0.x[i], p1.x[i])
		M = max(p0.x[i], p1.x[i])

		c0[i] = 0.5 * ((M+m) - beta*(M-m))
		c1[i] = 0.5 * ((M+m) + beta*(M-m))

	# create children and return
	return c0, c1
