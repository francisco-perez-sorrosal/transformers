import logging
import pickle
from abc import abstractmethod, ABC

import numpy as np
import torch

# from transformers.file_utils import is_tf_available
#
# if is_tf_available():
#     import tensorflow as tf

valid_output_functions = ['softmax', 'sigmoid']

softmax_valid_strategies = ['no_weights', 'per_class']
sigmoid_valid_strategies = ['constant', 'per_example', 'per_class']

logger = logging.getLogger(__name__)


def save_to_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def calculate_class_weights(label_frequency_distribution_dict, total_number_of_documents, mu=1.0, smoothness=0.001):
    """
    Calculates the class weights based on the frequency distribution of documents per class and the number of documents

    :param mu allows to control the global relative weighting between positive and negative examples.
    In the final model, this can impact the overall accuracy of the model, and will also specifically
    affect the precision/recall tradeoff.
    :param smoothness is a hyperparameter to control the smoothness of the weight equalization

    :returns class weights
    """
    alpha = total_number_of_documents * smoothness
    assert alpha >= 0
    max_of_label_frequencies = np.max(list(label_frequency_distribution_dict.values()))
    logger.info("# of examples (train ds): {}".format(total_number_of_documents))
    logger.info("max(label frequencies): {}".format(max_of_label_frequencies))
    logger.info("Mu/Smoothness/Alpha: {}/{}/{}".format(mu, smoothness, alpha))

    class_weights = dict()
    for label in label_frequency_distribution_dict.keys():
        class_weights[label] = mu * ((max_of_label_frequencies + alpha) /
                                     float(label_frequency_distribution_dict[label] + alpha))

    return class_weights


class WeightLossStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def get_weights(self):
        pass


class PerClassWeightLossStrategy(WeightLossStrategy):
    """Strategy that calculates automatically the weights per class based on the frequency distribution
     of documents per class and the number of documents"""

    def __init__(self, label_frequency_distribution, total_number_of_documents, mu, smoothness, **extras):
        logger.info("Total number of documents: {}. Label/class label frequency distribution: {}"
                    .format(total_number_of_documents, label_frequency_distribution))
        class_weights = list(calculate_class_weights(label_frequency_distribution,
                                                     total_number_of_documents,
                                                     mu,
                                                     smoothness).values())
        logger.info("Label/class weights (calculated): {}".format(class_weights))
        # self.weights = tf.constant(class_weights, dtype=tf.float32)
        self.weights = torch.tensor(class_weights, dtype=torch.float32)

    def get_weights(self):
        return self.weights


#######################################################################################################################
# Softmax strategies
#######################################################################################################################

class SoftmaxNoWeightsLossStrategy(WeightLossStrategy):
    """Strategy that allows to pass a specific list of weights per class"""

    def __init__(self, k_hot_labels, **extras):
        # self.label_length = tf.cast(tf.shape(k_hot_labels)[1], dtype=tf.float32)
        # if extras['debug']: self.label_length = tf.Print(self.label_length,
        #                                                  [self.label_length, tf.shape(self.label_length)], 'label lenght=')
        self.label_length = torch.tensor(k_hot_labels.shape[1], dtype=torch.float32).numpy()
        logger.info("No class weights strategy. Using array of ones (# of labels: {})".format(self.label_length))
        # self.weights = tf.ones(tf.cast(self.label_length, dtype=tf.int32), dtype=tf.float32)
        self.weights = torch.ones(self.label_length, dtype=torch.float32)

    def get_weights(self):
        return self.weights


class SoftmaxPerClassWeightLossStrategy(PerClassWeightLossStrategy):
    """Softmax extension of PerClassWeightLossStrategy. Uses all the functionallyty from parent class.
    This class could be avoided. We make it explicit just in case we need to differentiate sigmoid/softmax functionality.
    """


#######################################################################################################################
# Sigmoid strategies
#######################################################################################################################

class SigmoidConstantWeightLossStrategy(WeightLossStrategy):
    """Strategy that allows to pass a specific weight scalar to affect the positive labels"""

    def __init__(self, sigmoid_pos_weight, **extras):
        # self.weights = tf.constant(sigmoid_pos_weight)
        self.weights = torch.tensor(sigmoid_pos_weight, dtype=torch.float32)

    def get_weights(self):
        return self.weights


class SigmoidPerClassWeightLossStrategy(PerClassWeightLossStrategy):
    """Sigmoid extension of PerClassWeightLossStrategy. Uses all the functionallyty from parent class.
    This class could be avoided. We make it explicit just in case we need to differentiate sigmoid/softmax functionality.
    """


class SigmoidPerExampleWeightLossStrategy(WeightLossStrategy):
    """Strategy that automatically calculates the weight per example, based on the ratio:
     negative classes (labels)/possitive classes (labels) in each example"""

    def __init__(self, batch, mu, **extras):
        self.mu = mu
        logger.info("Mu: %d", self.mu)
        self.batch = batch
        self.debug = extras['debug']
        # if self.debug: self.batch = tf.Print(self.batch, [self.batch, tf.shape(self.batch)], 'batch=', summarize=1000)

    def get_weights(self):
        # label_length = tf.cast(tf.shape(self.batch)[1], dtype=tf.float32)
        # # label_length = tf.Print(label_length, [label_length, tf.shape(label_length)], 'label lenght=', summarize=200)
        # label_positives_per_batch = tf.reduce_sum(self.batch, axis=1)
        # if self.debug: label_positives_per_batch = tf.Print(label_positives_per_batch, [label_positives_per_batch, tf.shape(
        #     label_positives_per_batch)], 'label positives=', summarize=200)
        # label_negatives_per_batch = label_length - label_positives_per_batch
        # if self.debug: label_negatives_per_batch = tf.Print(label_negatives_per_batch, [label_negatives_per_batch, tf.shape(
        #     label_negatives_per_batch)], 'label negatives=', summarize=200)
        # weight_per_example = self.mu * tf.div_no_nan(label_negatives_per_batch, label_positives_per_batch)
        # weight_per_example_transposed = tf.reshape(weight_per_example, [-1, 1])
        # if self.debug: weight_per_example_transposed = tf.Print(weight_per_example_transposed,
        #                                                         [weight_per_example_transposed,
        #                                                          tf.shape(weight_per_example_transposed)],
        #                                                         "weight per example tx=", summarize=200)
        # return tf.tile(weight_per_example_transposed, [1, tf.cast(label_length, dtype=tf.int32)])
        return None


class WeightLossStrategyFactory:
    """
    Factory for weight strategies, holding a singleton strategy
    """
    current_strategy = None  # Singleton strategy

    def __init__(self):
        self._builders = {}

    def register_builder(self, output_function, strategy, builder):
        logger.info("Registering weight builder for output function/strategy {}/{}".format(output_function, strategy))
        key = self._compose_key(output_function, strategy)
        self._builders[key] = builder

    def get_weights(self, output_function, strategy, **kwargs):
        logger.info("Getting weights for output function {} and strategy {}".format(output_function, strategy))
        if not WeightLossStrategyFactory.current_strategy:
            key = self._compose_key(output_function, strategy)
            builder = self._builders.get(key)
            if not builder:
                raise ValueError(key)
            logger.info("======================================================")
            logger.info("Creating weight loss strategy {} one time".format(key))
            logger.info("======================================================")
            WeightLossStrategyFactory.current_strategy = builder(**kwargs)
        weights = WeightLossStrategyFactory.current_strategy.get_weights()
        # if kwargs['debug']: weights = tf.Print(weights, [weights, tf.shape(weights)], 'Weights values=', summarize=200)
        logger.info("Weight values ({}): {}".format(weights, weights.shape))
        return weights

    def _compose_key(self, output_function, strategy):
        output_function = output_function.lower()
        strategy = strategy.lower()
        if output_function not in valid_output_functions:
            raise ValueError("output_function should be in {}".format(valid_output_functions))
        if output_function == 'softmax' and strategy not in softmax_valid_strategies:
            raise ValueError("valid strategies for softmax should be in {}".format(softmax_valid_strategies))
        if output_function == 'sigmoid' and strategy not in sigmoid_valid_strategies:
            raise ValueError("valid strategies for sigmoid should be in {}".format(sigmoid_valid_strategies))
        return output_function + "-" + strategy
