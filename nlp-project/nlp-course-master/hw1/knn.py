import numpy as np

def knn(vector, matrix, k=10):
    """
    Finds the k-nearest rows in the matrix with comparison to the vector.
    Use the cosine similarity as a distance metric.

    Arguments:
    vector -- A D dimensional vector
    matrix -- V x D dimensional numpy matrix.

    Return:
    nearest_idx -- A numpy vector consists of the rows indices of the k-nearest neighbors in the matrix
    """

    nearest_idx = []

    ### YOUR CODE HERE
    norm_of_matrix = np.linalg.norm(matrix, axis=1)
    similarity = np.dot(matrix, vector) / norm_of_matrix
    nearest_idx = np.argpartition(similarity, -k)[-k:]
    nearest_idx = nearest_idx[np.argsort(similarity[nearest_idx])][::-1]
    ### END YOUR CODE
    return nearest_idx

def test_knn():
    """
    Use this space to test your knn implementation by running:
        python knn.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    nearest_idx_1 = knn(np.array([1, 2, 3]), np.array([[1, 2, 2], [1, 2, 3], [0, 0, 20], [5, 0, 5]]), 3)
    assert np.allclose(nearest_idx_1, [1, 0, 2])
    nearest_idx_2 = knn(np.array([1, 0, 0]), np.array([[1, 0, 0], [0, 1.02, 0], [-2, 0, 0], [-1.01, 0, 0], [-6, 0, 3]]),
                        4)
    assert np.allclose(nearest_idx_2, [0, 1, 4, 3])
    ### END YOUR CODE


def cosine_sim(a, b):
    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    test_knn()


