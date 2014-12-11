__author__ = 'Mark Laane'

import logging

import cv2
import numpy
import sys

from combining_classifications import combine_majority_vote, combine_mean_rule, combine_minimum_rule
from loading_images import load_face_vectors_from_disk, extract_color_channels
from pca import PCA
from plotting import plot_results


def main():
    logging.basicConfig(format='%(levelname)7s: %(message)s', level=logging.INFO)

    no_of_persons = 13  # Number of persons
    samples_person = 10  # Number of samples per person
    samples_training = 9
    image_size = (50, 50)  # All face images will be resized to this

    combining_functions = [
        (combine_majority_vote, "Majority Voting"),
        (combine_minimum_rule, "Minimum Rule"),
        (combine_mean_rule, "Mean Rule")
    ]


    all_image_numbers = generate_all_image_numbers(no_of_persons, samples_person)
    all_face_vectors = load_face_vectors_from_disk(all_image_numbers, image_size, load_channels_bgrhs=True)
    color_channels = extract_color_channels(all_face_vectors)

    test_different_training(color_channels, no_of_persons, samples_person, combining_functions)

    #results = train_and_test(color_channels, no_of_persons, samples_person, samples_training)
    #logging.debug(str(results))


def test_different_training(color_channels, no_of_persons, samples_person, combining_functions):
    plot_number_of_training_samples = []
    x_min = 1
    x_max = samples_person-1
    number_of_diff_trainings = x_max+1 - x_min
    number_of_tests = 10
    number_of_results = 3
    plot_recognition_rate = numpy.empty((number_of_results, number_of_tests*number_of_diff_trainings))
    count = 0
    for test_no in range(number_of_tests):
        sys.stdout.write("\r%d%%" % (test_no * 100 // number_of_tests))
        sys.stdout.flush()
        for samples_training in range(x_min, x_max+1):
            results = train_and_test(color_channels, no_of_persons, samples_person, samples_training, combining_functions)

            plot_number_of_training_samples.append(samples_training)
            plot_recognition_rate[:, count] = results
            count += 1

    print()

    # Plot results:
    plot_results(
        x_axis=plot_number_of_training_samples,
        y_axis=plot_recognition_rate,
        x_min=x_min,
        x_max=x_max,
        labels=[name for func, name in combining_functions]
    )


def train_and_test(color_channels, no_of_persons, samples_person, samples_training, combining_functions):

    # split into training and testing:
    all_testing_idx, all_training_idx = randomly_split_classes(
        no_of_classes=no_of_persons,
        samples_per_class=samples_person,
        training_samples_per_class=samples_training
    )
    classifiers = train_classifiers(all_training_idx, samples_training, color_channels)
    samples_testing = samples_person - samples_training
    predictions = classify_testing_samples(classifiers, all_testing_idx, color_channels,
                                           samples_testing * no_of_persons)

    test_classes = [class_no for class_no in range(no_of_persons) for _ in range(samples_testing)]
    results = []
    for function, function_name in combining_functions:
        combined_predictions = function(predictions, samples_testing * no_of_persons)
        right_classification = numpy.equal(combined_predictions, test_classes)
        prediction_rate = numpy.count_nonzero(right_classification) / right_classification.size
        logging.debug(
            "{}: Prediciton rate:{:.2%} Predictions:{}".format(function_name, prediction_rate, combined_predictions))
        results.append(prediction_rate)

    return results

def show_vectors_as_images(vectors: numpy.ndarray, image_size, wait_time=None):
    for i, vector in enumerate(vectors):
        temp_image = vector[0].reshape(image_size)
        cv2.imshow("channel-{}".format(i), temp_image)
    if wait_time is not None:
        cv2.waitKey(wait_time)


def randomly_split_classes(no_of_classes, samples_per_class, training_samples_per_class):
    testing_samples_per_class = (samples_per_class - training_samples_per_class)
    training_samples_total = no_of_classes * training_samples_per_class
    testing_samples_total = no_of_classes * testing_samples_per_class
    all_training_idx = numpy.empty(training_samples_total, dtype=int)
    all_testing_idx = numpy.empty(testing_samples_total, dtype=int)

    for class_no in range(no_of_classes):
        # For every person, take training and testing samples randomly
        random_permutation = numpy.random.permutation(samples_per_class)
        cls_training_idx, cls_testing_idx = random_permutation[:training_samples_per_class], \
                                            random_permutation[training_samples_per_class:]

        all_training_idx[class_no * training_samples_per_class:(class_no + 1) * training_samples_per_class] = \
            cls_training_idx + samples_per_class * class_no
        all_testing_idx[class_no * testing_samples_per_class:(class_no + 1) * testing_samples_per_class] = \
            cls_testing_idx + samples_per_class * class_no

    return all_testing_idx, all_training_idx


def train_classifiers(all_training_idx, samples_training, color_channels):
    logging.debug("Training classifiers..")
    classifiers = []
    for channel in color_channels:
        classifier = PCA(samples_training)
        classifier.train(channel[all_training_idx])
        classifiers.append(classifier)
    return classifiers


def classify_testing_samples(classifiers, all_testing_idx, color_channels, total_testing_samples):
    logging.debug("Testing classifiers..")
    predictions = numpy.empty((len(classifiers), total_testing_samples, 2), dtype=int)
    for no, (channel, classifier) in enumerate(zip(color_channels, classifiers)):
        assert (isinstance(classifier, PCA))
        predictions[no] = classifier.classify_samples(channel[all_testing_idx])
    return predictions


def generate_all_image_numbers(no_of_persons, samples_person):
    """
    Generates and returns a list of all possible combinations of imagenumbers

    :param no_of_persons: number of persons
    :param samples_person: number of samples used per person
    :return: array of numbers
    """
    return numpy.mgrid[1:samples_person + 1, 1:no_of_persons + 1].T.reshape(-1, 2)[:, ::-1]


if __name__ == '__main__':
    main()