import numpy as np
from numpy.linalg import eig


def eof(M):
    """
    Calculates the Empirical Orthogonal Functions (EOFs) of a given field.
    If the number of spatial points far exceeds the number of temporal points, 
    performance will become an issue. 
    Thus, this function should only be used for approximately square fields.

    Args:
        M (numpy.ndarray): A 3D array representing the input field. Dimensions are (time, latitude, longitude).

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            - Eigenvalues normalized by their sum.
            - EOFs (Canonical Correlation patterns) reshaped to match the original domain.
            - Principal Components (PCs) corresponding to the EOFs.
    """
    # Convert NaN values to zero
    F = np.nan_to_num(M)

    # Extract dimensions
    num_times = len(F[:,0,0])
    nlat = len(F[0,:,0])
    nlon = len(F[0,0,:])

    # Reshape the field to a 2D matrix (time x space)
    F = F.reshape(num_times, nlat*nlon)

    # Subtract the mean from each column (space)
    n = nlat * nlon
    for i in range(n):
        F[:,i] -= np.mean(F[:,i])

    # Compute the covariance matrix R = F.T @ F
    R = F.T @ F

    # Compute eigenvalues (vals) and eigenvectors (vecs)
    vals, vecs = np.linalg.eig(R)

    # Sort eigenvectors by largest eigenvalue
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:,idx]

    # Project data matrix onto eigenvectors to find PCs
    pcas = np.zeros([num_times, n])
    for i in range(n):
        pcas[:,i] = F @ vecs[:,i]

    # Reshape vecs to match the original domain (latitude x longitude)
    vecs = vecs.reshape(nlat, nlon, -1)

    # Normalize eigenvalues by their sum
    normalized_vals = vals / np.sum(vals)

    return normalized_vals, vecs, pcas


def nonsquare_eof(F):
    """
    Calculates the Empirical Orthogonal Functions (EOFs) of a given field.
    The field can be nonsquare, i.e. have many more points in space than in time,
    without this causing significant slowdown.
    This function is designed for use for such nonsquare matrices.

    Args:
        F (numpy.ndarray): A 4D array representing the input field. Dimensions are (time, level, latitude, longitude).

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            - Eigenvalues normalized by their sum.
            - EOFs reshaped to match the original field shape.
            - Principal Components (PCs) corresponding to the EOFs.
    """
    # Extract dimensions
    nlev = len(F[0,:,0,0])
    num_times = len(F[:,0,0,0])
    nlat = len(F[0,0,:,0])
    nlon = len(F[0,0,0,:])

    # Reshape the field to a 2D matrix (time x space)
    F = F.reshape(num_times, nlev*nlat*nlon)

    # Subtract the mean from each column (space)
    for i in range(nlev*nlat*nlon):
        F[:,i] -= np.mean(F[:,i])

    # Compute the covariance matrix L = F @ F^T
    L = F @ F.T

    # Compute eigenvalues (Lambda) and eigenvectors (B)
    Lambda, B = np.linalg.eig(L)

    # Sort eigenvectors by largest eigenvalue
    idx = Lambda.argsort()[::-1]
    Lambda = Lambda[idx]
    B = B[:,idx]

    # Compute D = F^T @ B
    D = F.T @ B

    # Compute square root of eigenvalues
    eps = 1e-10  # Small constant to avoid division by zero
    sq = np.zeros([D.shape[0], D.shape[1]])

    for i in range(D.shape[1]):
        sq[:,i] = np.sqrt(Lambda[i] + eps)

    # Compute Canonical Correlation patterns (CC)
    CC = D / sq

    # Compute Principal Components (PCs)
    pcas = np.zeros([F.shape[0], F.shape[1]])
    for i in range(CC.shape[1]):
        pcas[:,i] = F @ CC[:,i]

    # Reshape CC to match the original field shape
    CC = CC.reshape(nlev, nlat, nlon, -1)

    # Normalize eigenvalues by their sum
    normalized_lambda = Lambda / np.sum(Lambda)

    return normalized_lambda, CC, pcas


