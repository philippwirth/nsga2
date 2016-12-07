import numpy as np

mutationEta = 2.0
deltaMax = 1.0

def mutate(c):

	n = c.inputDim

	for i in range(n):

		# sample delta from the distribution 0.5*(eta + 1)(1 - delta)^eta
		u = np.random.rand()
		if u <= 0.5:
			delta = pow(2*u, (1/(mutationEta + 1))) - 1
		else:
			delta = 1 - pow(2*(1-u), 1/(mutationEta + 1))

		c.x[i] += delta*deltaMax

	return c