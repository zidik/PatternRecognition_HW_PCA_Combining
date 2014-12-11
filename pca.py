__author__ = 'Mark Laane'

import numpy

class PCA:
    def train(self, training_samples: numpy.ndarray, no_of_components: int=None):
        # Guard(s)
        if no_of_components is not None:
            if no_of_components < training_samples.shape[0]:
                raise ValueError("no_of_components cannot be larger than number of samples in training data")

        # Calculate the average face
        average_sample = training_samples.mean(axis=0)
        # Calculate eigenvectors
        train_diff = training_samples - average_sample
        eig_values, eig_vectors = numpy.linalg.eig(numpy.dot(train_diff, train_diff.transpose()))
        eig_vectors = numpy.dot(eig_vectors, train_diff)

        #Reduce dimensionality:
        if no_of_components is not None:
            # Sort eigenvectors by eigenvalues
            idx = numpy.argsort(-eig_values)
            eig_vectors = eig_vectors[idx]
            # Take only top "principal_components" of eigenvectors
            eig_vectors = eig_vectors[:no_of_components]

        # Calculate weights for each image
        train_weights = numpy.dot(eig_vectors, train_diff.transpose()).transpose()

        return average_sample, eig_vectors, train_weights
