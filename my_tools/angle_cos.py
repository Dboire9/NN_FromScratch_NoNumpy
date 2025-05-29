from my_tools.VectorMatrixClass import Vector

def angle_cos(u: Vector, v: Vector) -> float:
	"""Return the cosine of two vectors"""
	if len(u.values) != len(v.values):
		raise ValueError("Vectors must have the same length")
	if all(val == 0 for val in u.values) or all(val == 0 for val in v.values):
		raise ValueError("Vectors must not be 0")

	u_norm = u.copy().norm()
	v_norm = v.copy().norm()
	u_dot = u.dot(v)
	return round(u_dot / (u_norm * v_norm), 9)