import numpy as np

mutationEta = 2.0
deltaMax = 1.0

# mutates and individual using parameter based mutation
def mutate(c):
	
	n = c.inputDim		# get dimension of search vector
	
	for i in range(n):	
		# sample delta from a distribution chosen to imitate binary crossover
		u = np.random.rand()
		if u <= 0.5:
			delta = pow(2*u, (1/(mutationEta + 1))) - 1
		else:
			delta = 1 - pow(2*(1-u), 1/(mutationEta + 1))

		# mutate 
		c.x[i] += delta*deltaMax

	return c
