import numpy as np
import pandas as pd

movies_df = pd.read_csv("movies.csv")

ratings_df = pd.read_csv("ratings.csv")

dic = {}
for i in range(9742):
    dic[i] = movies_df['movieId'][i]

#create 610*9742 matrix that rows are users and columns are movies
   
# movie_id_2= np.ndarray( shape=(len(ratings_df.movieId)) , dtype=np.int32)
# for i in range(len(ratings_df.movieId)):
#     movie_id_2[i] = dic[int(ratings_df.movieId[i])]   
# movie_id= np.ndarray( shape=(610,9742) , dtype=np.int32)
# for j in range(610):
#     for i in range(len(ratings_df.userId)):
#         if ratings_df.userId[i] == j+1 :
#             movie_id[j][movie_id_2[i]] = ratings_df.rating[i]
# np.savetxt('myarray.txt', movie_id)
#####################################################################
df = np.loadtxt("data.txt") # read 610*9742 matrix that rows are users and columns are movies that already created


normalized_df = df - np.asarray([(np.mean(df, 1))]).T


def power_iteration(matrix, num_iterations):

    vector = np.ones(len(matrix))

    for _ in range(num_iterations):
        matrix_vector_product = np.dot(matrix, vector)


        vector = matrix_vector_product / np.linalg.norm(matrix_vector_product)

    eigenvalue = np.dot(matrix_vector_product, vector)

    return eigenvalue, vector


def eigen_decomposition(A, d=9742 ,num_iterations=10):
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


def svd(A , eigenvalues , eigen_vector):

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigen_vector = eigen_vector[:, sorted_indices]

    matrix_AAT = np.dot(A,A.T) # AAT matrix
    eigenvalues_M , U = eigen_decomposition(matrix_AAT , d=610,num_iterations=500)
    sorted_indices = np.argsort(eigenvalues_M)[::-1]
    U = U[:, sorted_indices]

    singular_list = []
    for eigen_value in eigenvalues:
        if eigen_value > 0:
            singular_list.append(np.sqrt(eigen_value))
    singular_values = np.array(singular_list)

    num_positive_singular = len(singular_values)
    
    Sigma = singular_values
    
    return U[: , :num_positive_singular], Sigma, eigen_vector[:, :num_positive_singular].T


def top_cosine_similarity(U, V, user_id, top_n=10):
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

U, Sigma, Vt = svd(normalized_df, eigenvalues, eigenvectors)



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
indexes = top_cosine_similarity(U, sliced_V, user_id, top_n)

for i in indexes:
    if df[user_id , i] == 0:
        print(movies_df[movies_df["movieId"] == dic[i]].title.values[0])
