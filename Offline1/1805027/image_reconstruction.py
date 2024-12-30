import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_and_convert_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image

def SVD_decomposition(gray_image):
    U, S, Vt = np.linalg.svd(gray_image, full_matrices=False)

    return U, S, Vt

def k_rank_approximation(U, S, Vt, k):
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    A_k = U_k @ S_k @ Vt_k

    return A_k

def reconstruct_image(gray_image):
   
    n, m = gray_image.shape
    initial_k_values = [1, 5, 10, 20, 30, 40, 45, 50, 100]

    k = initial_k_values[-1]
    while k * 2 <= min(n, m):
        k *= 2
        initial_k_values.append(k)

    # Your final k values
    k_values = np.array(initial_k_values)

    # Calculating the number of rows and columns for the subplots
    num_k_values = len(k_values)
    cols = 4  # max number of columns
    rows = num_k_values // cols + (num_k_values % cols > 0)

    for i, k in enumerate(k_values, 1):
        
        U, S, Vt = SVD_decomposition(gray_image)
        A_k = k_rank_approximation(U, S, Vt, k)

        plt.subplot(rows, cols, i)  
        plt.imshow(A_k, cmap='gray')
        plt.title(f'k = {k}')
        plt.axis('on')
    plt.tight_layout()
    plt.savefig('reconstructed_image.png')
    plt.show()


def main():
   
    gray_image = read_and_convert_image('image.jpg')
    reconstruct_image(gray_image)

main()