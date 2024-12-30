import numpy as np

def generate_full_rank_matrix(n, max_attempts=1000):
   
    for _ in range(max_attempts):
        A = np.random.randint(-10, 10, size=(n, n))
        if np.linalg.matrix_rank(A) == n:
            return A
    raise ValueError(f"Failed to generate a full-rank matrix after {max_attempts} attempts")

def generate_invertible_symmetric_matrix(A):
    return A.T @ A

def findEigenSpace(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    return eigenValues, eigenVectors

def reconstructMatrix(eigenValues, eigenVectors):
    D = np.diag(eigenValues)
    return eigenVectors @ D @ eigenVectors.T

n = int(input("Enter the dimension of the matrix: "))

A = generate_full_rank_matrix(n)
A = generate_invertible_symmetric_matrix(A)

print("The matrix A is:")
print(A)

eigenValues, eigenVectors = findEigenSpace(A)

print("The eigen values are:")
print(eigenValues)
print("The eigen vectors are:")
print(eigenVectors)

B = reconstructMatrix(eigenValues, eigenVectors)

print("The reconstructed matrix B is:")
print(B)

if np.allclose(A, B):
    print("The matrix is reconstructed successfully!")
else:
    print("The matrix is not reconstructed successfully!")

