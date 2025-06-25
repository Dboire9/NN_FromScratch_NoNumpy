from math import tan, pi, exp

class Vector:
    def __init__(self, values):
        # Flatten if values is a list of lists
        if values and isinstance(values[0], list):
            self.values = [v[0] for v in values]
        else:
            self.values = list(values)

    def __str__(self):
        return '\n'.join([str([comp]) for comp in self.values])
    
    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value
    
    def __len__(self):
        return len(self.values)
    
    def __sub__(self, other):
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must have the same length for subtraction")
            return Vector([self.values[i] - other.values[i] for i in range(len(self))])
        elif isinstance(other, (int, float)):
            return Vector([v - other for v in self.values])
        else:
            raise TypeError("Unsupported type for Vector subtraction")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Vector([other - v for v in self.values])
        else:
            raise TypeError("Unsupported type for Vector subtraction")

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
        for i in range(len(self.values)):
            self[i] *= v[i]
            result += self[i]
        return result

    def norm_1(self) -> float:
        result = 0
        for i in range(len(self.values)):
            if(self.values[i] < 0):
                self.values[i] *= -1
            result += self.values[i]
        return result

    def norm(self) -> float:
        result = 0
        for i in range(len(self.values)):
            result += self.values[i]**2
        return result**0.5
    
    def norm_inf(self):
        for i in range(len(self.values)):
            if(self.values[i] < 0):
                self.values[i] *= -1
            self.values[i] = self.values[i]
        return max(self.values)

    def reshape(self, rows, cols):
        if len(self.values) != rows * cols:
            raise ValueError("Vector size does not match the desired matrix shape")
        data = [self.values[i*cols:(i+1)*cols] for i in range(rows)]
        return Matrix(data)

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
    
    def __len__(self):
        # Returns the number of columns if matrix is not empty, else 0
        if self.rows and len(self.rows[0]) > 1:
            return len(self.rows[0])
        elif self.rows:
            return len(self.rows)
        else:
            return 0

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

    def __matmul__(self, other):
        return self.mul_mat(other)

    def __mul__(self, other):
        # Element-wise multiplication with scalar or another Matrix
        if isinstance(other, Matrix):
            if self.shape() != other.shape():
                raise ValueError("Shapes must match for element-wise multiplication")
            return Matrix([
                [self.rows[i][j] * other.rows[i][j] for j in range(self.num_cols)]
                for i in range(self.num_rows)
            ])
        else:  # scalar
            return Matrix([
                [self.rows[i][j] * other for j in range(self.num_cols)]
                for i in range(self.num_rows)
                ])

    def __rmul__(self, other):
        # For scalar * Matrix
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if self.shape() != other.shape():
                raise ValueError("Shapes must match for element-wise subtraction")
            return Matrix([
                [self.rows[i][j] - other.rows[i][j] for j in range(self.num_cols)]
                for i in range(self.num_rows)
            ])
        else:  # scalar
            return Matrix([
                [self.rows[i][j] - other for j in range(self.num_cols)]
                for i in range(self.num_rows)
            ])

    def __rsub__(self, other):
        # For scalar - Matrix
        return Matrix([
            [other - self.rows[i][j] for j in range(self.num_cols)]
            for i in range(self.num_rows)
        ])

    def add(self, v):
        """Add another matrix to this matrix"""
        if self.num_rows != v.num_rows:
            raise ValueError("Matrices must have same dimensions")
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.rows[i][j] += v.rows[i][j]

    def add_vector_to_matrix(matrix, vector):
        """
        Adds a vector (of length equal to matrix rows) to each column of the matrix.
        matrix: Matrix object
        vector: Vector object (length == matrix.num_rows)
        Returns a new Matrix object.
        """
        if matrix.num_rows != len(vector):
            raise ValueError("Vector length must match the number of matrix rows")
        result = []
        for i in range(matrix.num_rows):
            v = vector[i]
            if isinstance(v, list):  # Extract float if v is a list
                v = v[0]
            result.append([matrix.rows[i][j] + v for j in range(matrix.num_cols)])
        return Matrix(result)

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
    
    def exp(self):
        return Matrix([[exp(v) for v in row] for row in self.rows])
    
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
        result_rows = [
            [sum(self.rows[i][k] * mat.rows[k][j] for k in range(self.num_cols))
             for j in range(mat.num_cols)]
            for i in range(self.num_rows)
        ]
        return Matrix(result_rows)

    def projection(self, fov, ratio, near, far):
        result = self.copy()
        fov = fov * (pi / 180)
        # Scales the x-coordinate based on the fov
        result.rows[0][0] = 1 / (ratio * tan(fov/2))
        # Scales the y-coordinate based on the fov
        result.rows[1][1] = 1 / (tan(fov/2))
        # Maps z-coordinates (depth) into a normalized range
        result.rows[2][2] = -1 * ((far + near) / (far - near))
        result.rows[2][3] = -1 * ((2 * far * near) / (far - near))
        # Sets the output w-coordinate to −z
        result.rows[3][2] = -1
        id_matrix = Matrix([[1.0 if i == j else 0.0 for j in range(result.num_cols)] for i in range(result.num_rows)])
        
        # Creating a homogeonous vector from the ones we have in the .obj (e.g line 10)
        v = Vector([-0.000578, -0.064495, 0.100000, 1.0])
        # Matrix Vector multiplication
        dot_result = [0.0] * 4
        for j in range(result.num_rows):
            dot_prod = Vector([0.0, 0.0, 0.0, 0.0])
            for i in range(result.num_cols):
                dot_prod.values[i] = result.rows[j][i]
            dot_result[j] = dot_prod.copy().dot(v)
        # Checking if x2d and y2d are between -1 and 1, and if not changing the z
        while (dot_result[0] / dot_result[3] > 1) or (dot_result[1] / dot_result[3] > 1):
            v[2] -= 1
            id_matrix.rows[2][3] -= 1
            for j in range(0, result.num_rows):
                dot_prod = Vector([0.0, 0.0, 0.0, 0.0])
                for i in range(0, result.num_cols):
                    dot_prod.values[i] = result.rows[j][i]
                dot_result[j] = dot_prod.dot(v)
        while (dot_result[0] / dot_result[3] < -1) or (dot_result[1] / dot_result[3] < -1):
            v[2] += 1
            id_matrix.rows[2][3] += 1
            for j in range(0, result.num_rows):
                dot_prod = Vector([0.0, 0.0, 0.0, 0.0])
                for i in range(0, result.num_cols):
                    dot_prod.values[i] = result.rows[j][i]
                dot_result[j] = dot_prod.dot(v)
        # print(dot_result[0] / dot_result[3], dot_result[1] / dot_result[3])
        result = result.mul_mat(id_matrix)
        print (result)
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
        # print(f"Matrix {self.num_rows}x{self.num_cols}")
        return (self.num_rows, self.num_cols)

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