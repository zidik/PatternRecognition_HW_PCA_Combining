__author__ = 'Mark Laane'

import numpy


class PCA:
    def __init__(self, training_samples_per_class):
        self.training_samples_per_class = training_samples_per_class
        self.average_sample = None
        self.eig_vectors = None
        self.train_weights = None

    def train(self, training_samples: numpy.ndarray, no_of_components: int=None):
        # Calculate the average face
        self.average_sample = training_samples.mean(axis=0)
        # Calculate eigenvectors
        train_diff = training_samples - self.average_sample
        eig_values, self.eig_vectors = calculate_eigen_vects_values(train_diff)

        # Reduce dimensionality
        eig_values, self.eig_vectors = reduce_dimensionality(eig_values, self.eig_vectors, no_of_components)

        # Calculate weights for each image
        self.train_weights = numpy.dot(self.eig_vectors, train_diff.transpose()).transpose()

    def classify_samples(self, testing_samples):
        predictions = classify_samples(testing_samples, self.train_weights, self.training_samples_per_class,
                                       self.average_sample, self.eig_vectors)
        return predictions


def classify_samples(test_samples, train_weights, training_samples_per_class, average_sample, eig_vectors):
    test_diff = test_samples - average_sample
    test_weights = numpy.dot(eig_vectors, test_diff.transpose()).transpose()

    predictions = numpy.empty((test_samples.shape[0], 2))
    for sample_no, sample_weights in enumerate(test_weights):
        closest_training_sample_no, distance = find_closest_training_sample(sample_weights, train_weights)
        predicted_class_no = closest_training_sample_no // training_samples_per_class
        predictions[sample_no] = predicted_class_no, distance
    return predictions


def find_closest_training_sample(test, trainings):
    closest_training_no = None
    closest_distance = float("inf")
    for count, training in enumerate(trainings):
        distance = numpy.sqrt(numpy.sum((test - training) ** 2))
        if distance < closest_distance:
            closest_distance = distance
            closest_training_no = count
    return closest_training_no, closest_distance


def calculate_eigen_vects_values(train_diff):
    eig_values, eig_vectors = numpy.linalg.eig(numpy.dot(train_diff, train_diff.transpose()))
    eig_vectors = numpy.dot(eig_vectors, train_diff)
    return eig_values, eig_vectors


def reduce_dimensionality(eig_values, eig_vectors, no_of_components):
    if no_of_components is None:
        return eig_values, eig_vectors
    if no_of_components < eig_values.shape[0]:
        raise ValueError("no_of_components is larger than number of existing components")

    # Sort eigenvectors and -values by eigenvalues
    idx = numpy.argsort(-eig_values)
    eig_vectors = eig_vectors[idx]
    eig_values = eig_values[idx]
    # Take only top "principal_components"
    eig_vectors = eig_vectors[:no_of_components]
    eig_values = eig_values[:no_of_components]
    return eig_values, eig_vectors