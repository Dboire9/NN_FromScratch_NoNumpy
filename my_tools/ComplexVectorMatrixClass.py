class Vector:
	def __init__(self, values):
		self.values = list(values)

	def __str__(self):
		return '\n'.join([str([comp]) for comp in self.values])
	
	def __getitem__(self, index):
		return self.values[index]

	def __setitem__(self, index, value):
		self.values[index] = value

	def add(self, v):
		"""Add another vector to this vector"""
		if len(self.values) != len(v.values):
			raise ValueError("Vectors must have the same length")
		for i in range(len(self.values)):
			self[i] += v[i]

	def copy(self):
		"""Return a copy of this vector"""
		return Vector(self.values.copy())

	def dot(self, v):
		"""Return the dot product of two vectors"""
		if len(self.values) != len(v.values):
			raise ValueError("Vectors must have the same length")
		result = 0
		# Add the conjugate for complex numbers
		for i in range(len(self.values)):
			self.values[i] = self.values[i].conjugate()
		for i in range(len(self.values)):
			self[i] *= v[i]
			result += self[i]
		return result

	def norm_1(self) -> float:
		result = 0
		for i in range(len(self.values)):
			a = self.values[i].real
			b = self.values[i].imag
			magnitude_squared = a**2 + b**2
			magnitude_squared = magnitude_squared**0.5
			result += magnitude_squared
		return result

	def norm(self) -> float:
		result = 0
		for i in range(len(self.values)):
			a = self.values[i].real
			b = self.values[i].imag
			magnitude_squared = a**2 + b**2
			result += magnitude_squared
		result = result ** 0.5
		return result
	
	def norm_inf(self):
		result = [0.] * len(self.values)
		for i in range(len(self.values)):
			a = self.values[i].real
			b = self.values[i].imag
			magnitude_squared = a**2 + b**2
			magnitude_squared = magnitude_squared**0.5
			result[i] = magnitude_squared
		return max(result)

	def scl(self, K):
		"""Scale this vector by a scalar value"""
		for i in range(len(self.values)):
			self[i] *= K

	def sub(self, v):
		"""Substract another vector to this vector"""
		if len(self.values) != len(v.values):
			raise ValueError("Vectors must have the same length")
		for i in range(len(self.values)):
			self[i] -= v[i]




class Matrix:
	def __init__(self, rows):
		self.rows = [list(row) for row in rows]
		self.num_rows = len(rows)
		self.num_cols = len(rows[0]) if rows else 0
		if not all(len(row) == self.num_cols for row in rows):
			raise ValueError("All rows must have the same length")

	def __str__(self):
		return '\n'.join([f"[{', '.join(map(str, row))}]" for row in self.rows])

	def __getitem__(self, index):
		"""Allow accessing rows using m[i]"""
		return self.rows[index]
	
	def __setitem__(self, index, value):
		"""Allow setting rows using m[io turn in] = value"""
		if len(value) != self.num_cols:
			raise ValueError("Assigned row must match matrix column count")
		self.rows[index] = list(value)

	def add(self, v):
		"""Add another matrix to this matrix"""
		if self.num_rows != v.num_rows:
			raise ValueError("Matrices must have same dimensions")
		for i in range(self.num_rows):
			for j in range(self.num_cols):
				self.rows[i][j] += v.rows[i][j]

	def copy(self):
		"""Return a copy of this vector"""
		return Matrix([row.copy() for row in self.rows])

	def determinant(self):
		"""Return the determinant of a given matrix"""
		if self.num_cols != self.num_rows:
			raise ValueError("Determinant can only be calculated for a square matrix")
		total = 1
		self, swaps = self.row_reduction()
		for j in range(self.num_cols):
			total *= self.rows[j][j]
		if swaps != 0:
			total *= swaps * -1
		return total
	
	def inverse(self):
		"""Return the inverse of a given square matrix"""
		if self.num_cols != self.num_rows:
			raise ValueError("Inverse of a matrix must be done for a square matrix")
		result = self.copy()
		if self.determinant() != 0:
			result = result.augmented_matrix()
			result = result.reduced_row_echelon()
			result = result.get_inverse()
			return result

#Utils for the inverse
	def augmented_matrix(self):
		new = Matrix([[0.] * (self.num_cols * 2) for _ in range(self.num_rows)])
		for j in range(self.num_rows):
			for i in range(self.num_cols):
				new.rows[j][i] = self.rows[j][i]
		for k in range(self.num_cols, new.num_cols):
			new.rows[k % self.num_rows][k] = 1.0
		return new

	def get_inverse(self):
		result = self.copy()
		new = Matrix([[0.] * (self.num_rows) for _ in range(self.num_rows)])
		for j in range(new.num_rows):
			for i in range(new.num_cols):
				new.rows[j][i] = result.rows[j][i + self.num_rows]
		return new


	def mul_vec(self, vec: Vector) -> Vector:
		"""Multiply the matrix by a vector"""
		if self.num_cols != len(vec.values):
			raise ValueError("Length of the vector and the columns in the matrix must be equal")
		result = Vector([0.] * len(vec.values))
		for i in range(self.num_rows):
			for j in range(self.num_cols):
				result[i] += self.rows[i][j] * vec.values[j]
		return result

	def mul_mat(self, mat):
		"""Multiply the two matrices"""
		if self.num_cols != mat.num_rows:
			raise ValueError("Length of the column matrix and the rows in the other must be equal")
		result = Matrix([[0.] * mat.num_cols for _ in range(self.num_rows)])
		for i in range(self.num_rows):
			for j in range(self.num_cols):
				for k in range(mat.num_rows):
					result.rows[i][j] += self.rows[i][k] * mat.rows[k][j]
		return result

	def rank(self):
		rank = self.num_rows
		result = self.copy()
		result = result.reduced_row_echelon()
		for j in range(result.num_rows):
			if all(element == 0 for element in result.rows[j]):
				rank -= 1
		return rank

	def reduced_row_echelon(self):
		"""Return the reduced row echelon of a given matrix"""
		result = self.copy()
		# Find the pivot and rearrange the rows
		for i in range(self.num_cols):
			for j in range(self.num_rows):
				if self.rows[j][i] != 0: break
			if self.rows[j][i]: break
		if j != 0:
			for i in range(self.num_cols):
				result.rows[0][i] = self.rows[j][i]
				result.rows[j][i] = self.rows[0][i]
		# Row echelons
		for k in range(self.num_rows):
			j = 0
			result = result.norm_row_echelon(k)
			for j in range(result.num_rows):
				if j != k:
					pivot = result.search_pivot(k)
					result = result.sub_row_echelon(j, k, pivot)
		return result

# Utils for reduced row echelon

	def search_pivot(result, k):
		for i in range(k, result.num_cols):
			# print(f"k = 1 : {result.rows[i][k]}")
			if result.rows[i][k] != 0:
				return i
		return 0

	def sub_row_echelon(result, j, k, pivot):
		if result.rows[k][pivot] != 0:
			factor = result.rows[j][pivot] / result.rows[k][pivot]
			for i in range(result.num_cols):
				result.rows[j][i] -= factor * result.rows[k][i]
		return result

	def norm_row_echelon(result, k):
		for j in range(k, result.num_rows):
			for i in range(result.num_cols):
				if result.rows[j][i] != 0:
					divider = 1 / result.rows[j][i]
					start_cols = i
					for i in range(start_cols, result.num_cols):
						result.rows[j][i] *= divider
					return result
		return result

	def row_reduction(self):
		result = self.copy()  
		num_rows = result.num_rows
		swaps = 0
		
		for k in range(num_rows):
			# Find the pivot in the current column
			pivot_row = result.search_pivot(k)
			# If pivot not found at [k][k] swaps the pivot row with the k row
			if pivot_row != k:
				temp = result.copy()
				for i in range(self.num_cols):
					result.rows[k][i] = temp.rows[pivot_row][i]
					result.rows[pivot_row][i] = temp.rows[k][i]
				swaps += 1

			if result.rows[pivot_row][k] == 0:
				continue

			# Eliminate entries below the pivot
			for j in range(k + 1, num_rows):
				result = result.sub_row_echelon(j, k, pivot_row)
		return result, swaps

	def scl(self, K):
		"""Scale this matrix by a scalar value"""
		for i in range(self.num_rows):
			for j in range(self.num_cols):
				self.rows[i][j] *= K

	def shape(self):
		"""Return the shape of the matrix in arguments"""
		return f"Matrix {self.num_rows}x{self.num_cols}"

	def sub(self, v):
		"""Substract another matrix to this matrix"""
		if self.num_rows != v.num_rows:
			raise ValueError("Matrices must have same dimensions")
		for i in range(self.num_rows):
			for j in range(self.num_cols):
				self.rows[i][j] -= v.rows[i][j]

	def trace(self) -> float:
		"""Compute the trace of the matrix"""
		if self.num_cols != self.num_rows:
			raise ValueError("The matrix must be a square to do a trace")
		result = 0
		for i in range(self.num_cols):
			result += self.rows[i][i]
		return result

	def transpose(self):
		"""Return the transpose of the matrix"""
		result = Matrix([[0.] * self.num_rows for _ in range(self.num_cols)])
		for i in range(self.num_rows):
			for j in range(self.num_cols):
				result.rows[j][i] = self.rows[i][j]
		return result