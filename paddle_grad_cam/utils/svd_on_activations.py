import numpy as np


def get_2d_projection(activation_batch):
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()

        reshaped_activations = (reshaped_activations - reshaped_activations.mean(axis=0))
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)

        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)
