from my_tools.VectorMatrixClass import Matrix, Vector

def linear_combination(u, coefs) -> Vector:
	"""This function calculates the sum of each vector multiplied by its corresponding
	coefficient, i.e., Σ(coefs[n] * u[n]) for all n from 0 to len(u)-1."""
	
	if len(u) != len(coefs):
		raise ValueError("Number of u must match number of coefficients")
	if not all(len(v.values) == len(u[0].values) for v in u):
		raise ValueError("All u must have the same length")

	result = Vector([0.] * len(u[0].values))
	for i in range(len(u)):
		temp = Vector(u[i].values)
		temp.scl(coefs[i])
		result.add(temp)
	return result