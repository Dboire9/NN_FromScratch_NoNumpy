from my_tools.VectorMatrixClass import Vector

def cross_product(u: Vector, v: Vector) -> Vector:
	"""Return the cross product of two vectors
		which is a vector perpendicular to the other two
		For u⃗=[ux,uy,uz] and v⃗=[vx,vy,vz]:
		|uy*vz - uz*vy|
		|uz*vx - ux*vz|
		|ux*vy - uy*vx|
	"""

	if(len(u.values) != 3 and len(v.values) != 3):
		raise ValueError("The two vectors must be 3-dimensional")
	result = Vector([0.] * len(u.values))
	result[0] = u.values[1] * v.values[2] - u.values[2] * v.values[1]
	result[1] = u.values[2] * v.values[0] - u.values[0] * v.values[2]
	result[2] = u.values[0] * v.values[1] - u.values[1] * v.values[0]
	
	return result