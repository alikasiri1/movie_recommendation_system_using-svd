from locale import normalize
from turtle import shape
import numpy as np
import pandas as pd

movies_df = pd.read_csv("movies.csv")

ratings_df = pd.read_csv("ratings.csv")

dic = {}
for i in range(9742):
    dic[i] = movies_df['movieId'][i]

df = np.loadtxt("S.txt")


normalized_df = df - np.asarray([(np.mean(df, 1))]).T


def power_iteration(matrix, num_iterations):

    vector = np.ones(len(matrix))

    for _ in range(num_iterations):
        matrix_vector_product = np.dot(matrix, vector)


        vector = matrix_vector_product / np.linalg.norm(matrix_vector_product)

    eigenvalue = np.dot(matrix_vector_product, vector)

    return eigenvalue, vector


def eigen_decomposition(A,d=9742 ,num_iterations=10):
    # n = len(A)
    n = 50
    eigenvalues = np.zeros(n)
    eigenvectors = np.zeros((d, d))
    
    for i in range(n):
        print(i)

        eigenvalue, eigenvector = power_iteration(A, num_iterations)

        eigenvalues[i] = eigenvalue
        eigenvectors[:, i] = eigenvector
    
        A = A - eigenvalue * np.outer(eigenvector, eigenvector)
    
    return eigenvalues, eigenvectors


def svd(matrix_2 ,A, eigenvalues , eigen_vector):

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigen_vector = eigen_vector[:, sorted_indices]

    matrix = np.dot(matrix_2,matrix_2.T)
    eigenvalues_M , U = eigen_decomposition(matrix , d=610,num_iterations=500)
    sorted_indices = np.argsort(eigenvalues_M)[::-1]
    U = U[:, sorted_indices]

    singular_list = []
    for eigen_value in eigenvalues:
        if eigen_value > 0:
            singular_list.append(np.sqrt(eigen_value))
    singular_values = np.array(singular_list)

    num_positive_singular = len(singular_values)

    # U = np.dot(A, eigen_vector[:, :num_positive_singular])
    
    # U /= singular_values 


    # sorted_indices = np.argsort(eigenvalues_M)[::-1]
    
    Sigma = singular_values
    
    return U[: , :num_positive_singular], Sigma, eigen_vector[:, :num_positive_singular].T


def top_cosine_similarity(U, data, user_id, top_n=10):
    index = user_id - 1 
    user_row = U[index, :]
    mag_user_row = np.linalg.norm(user_row)
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(user_row, data.T) / (mag_user_row * magnitude)
    sort_indexes = np.argsort(-similarity)
    print(similarity.shape)
    print(sort_indexes , sort_indexes.max())
    return sort_indexes[:top_n]

def top_cosine_similarity2(U, V, user_id, top_n=10):
    user_index = user_id - 1
    user_row = U[user_index, :]

    cosine_distances = np.zeros(V.shape[0]) 
    for i in range(0, V.shape[0]):
        dot_product = np.dot(user_row, V[i])

        user_row_magnitude = np.linalg.norm(user_row)

        row_magnitude = np.linalg.norm(V[i])

        cosine_distance = dot_product / (user_row_magnitude * row_magnitude)

        cosine_distances[i] = cosine_distance

    sort_indexes = np.argsort(-cosine_distances)
    return sort_indexes[:top_n]




matrix = np.dot(normalized_df.T, normalized_df)


eigenvalues, eigenvectors = eigen_decomposition(matrix)
for i in range(0 , len(eigenvalues)):
    eigenvalues[i] = round(eigenvalues[i], 2)


print("Eigenvalue:\n", eigenvalues)
print()
print("Eigenvector:\n", eigenvectors)
print('\n\n')

U, Sigma, Vt = svd(normalized_df,matrix, eigenvalues, eigenvectors)



print("___________________________________________________________________________")
print("U:\n", U)
print()
print("___________________________________________________________________________")
print("Sigma:\n", Sigma) 
print()
print("___________________________________________________________________________")
print("Vt:\n", Vt)
print("___________________________________________________________________________")

k = 50
user_id = 156
top_n = 50

sliced_V = Vt.T[:, :k] 
print(sliced_V.shape)
indexes = top_cosine_similarity2(U, sliced_V, user_id, top_n)
print(U.shape[0])
print(U.shape[1])
col = U.shape[1]
for i in indexes:
    if df[user_id , i] == 0:
        print(movies_df[movies_df["movieId"] == dic[i]].title.values[0])
print("___________________________________________________________________________")
k = 50
user_id = 156
top_n = 50
U, Sigma, Vt = np.linalg.svd(normalized_df)
print("___________________________________________________________________________")
print("U:\n", U[: , :50])
print()
print("___________________________________________________________________________")
print("Sigma:\n", Sigma) 
print()
print("___________________________________________________________________________")
print("Vt:\n", Vt)
print("___________________________________________________________________________")
sliced_V = Vt.T[:, :k] 
indexes = top_cosine_similarity2(U[: , :50], sliced_V, user_id, top_n)

row = df[156 , :]
for i in indexes:
    if df[user_id , i] == 0:
        print(movies_df[movies_df["movieId"] == dic[i]].title.values[0])