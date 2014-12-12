__author__ = 'Mark'
from scipy import stats


def combine_majority_vote(predictions):
    majority_vote_predictions = (stats.mode(predictions[:, :, 0])[0])[0]
    return majority_vote_predictions.astype(int)


def combine_minimum_rule(predictions):
    total_testing_samples = predictions.shape[1]
    all_minimum_distances = [{} for _ in range(total_testing_samples)]
    for classifier_predictions in predictions:
        for sample_minimum_distances, (class_no, distance) in zip(all_minimum_distances, classifier_predictions):
            try:
                sample_minimum_distances[class_no]
            except KeyError:
                sample_minimum_distances[class_no] = distance
            else:
                if sample_minimum_distances[class_no] < distance:
                    sample_minimum_distances[class_no] = distance
    min_rule_prediction = [min(min_dists, key=min_dists.get) for min_dists in all_minimum_distances]
    return min_rule_prediction


def combine_mean_rule(predictions):
    total_testing_samples = predictions.shape[1]
    all_distances = [{} for _ in range(total_testing_samples)]
    for classifier_predictions in predictions:
        for sample_distances, (class_no, distance) in zip(all_distances, classifier_predictions):
            try:
                sample_distances[class_no]
            except KeyError:
                sample_distances[class_no] = [distance]
            else:
                sample_distances[class_no].append(distance)

    mean_rule_prediction = [min(dists, key=lambda x: sum(dists[x]) / len(dists[x])) for dists in all_distances]
    return mean_rule_prediction