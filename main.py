__author__ = 'Mark Laane'

import cv2
import numpy
import random
import logging
from loading_images import load_face_vectors_from_disk, extract_color_channels
from pca import PCA


no_of_persons = 13  # Number of persons
samples_person = 10  # Number of samples per person
samples_training = 9
image_size = (200, 200)  # All face images will be resized to this





def main():

    logging.basicConfig(format='[%(asctime)s] %(levelname)7s: %(message)s', level=logging.DEBUG)

    all_image_numbers = generate_all_image_numbers(no_of_persons, samples_person)
    all_face_vectors = load_face_vectors_from_disk(all_image_numbers, image_size, load_channels_bgrhs=True)

    color_channels = extract_color_channels(all_face_vectors)

    print(color_channels.shape)
    for i, channel in enumerate(color_channels):
        temp_image = channel[0].reshape(image_size)
        cv2.imshow("channel-{}".format(i), temp_image)
    cv2.waitKey()

    classifier = PCA()
    logging.debug("Training..")

    trainings = [classifier.train(channel) for channel in color_channels]

    for training in trainings:


    #train_average_face = average_sample.copy().reshape(image_size)
    #cv2.imshow('averageFace', train_average_face)
    #cv2.waitKey()


def generate_all_image_numbers(no_of_persons, samples_person):
    """
    Generates and returns a list of all possible combinations of imagenumbers

    :param no_of_persons: number of persons
    :param samples_person: number of samples used per person
    :return: array of numbers
    """
    return numpy.mgrid[1:samples_person+1, 1:no_of_persons+1].T.reshape(-1, 2)[:, ::-1]


if __name__ == '__main__':
    main()