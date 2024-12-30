import numpy as np

def generate_invertible_matrix(n):
    
    while True:
        A = np.random.randint(-10, 10, size=(n, n))
       
        try:
            np.linalg.inv(A)
            return A
        except np.linalg.LinAlgError:
            continue

def findEigenSpace(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    return eigenValues, eigenVectors

def reconstructMatrix(eigenValues, eigenVectors):
    D = np.diag(eigenValues)
    return eigenVectors @ D @ np.linalg.inv(eigenVectors)

def verifyMatrix(A, B):
    return np.allclose(A, B)

# Taking input from the user
n = int(input("Enter the dimension of the matrix: "))

# Generating the matrix
A = generate_invertible_matrix(n)

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

if verifyMatrix(A, B):
    print("The matrix is reconstructed successfully!")
else:
    print("The matrix is not reconstructed successfully!")
