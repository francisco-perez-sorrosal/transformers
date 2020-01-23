# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Toxic processors and helpers """

import logging
import os

import pandas as pd

from .utils import DataProcessor, InputExample, InputFeatures
from ...file_utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

toxic_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
toxic_classes_plus_non_toxic_class = ["non_toxic", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def toxic_convert_examples_to_features(examples, tokenizer,
                                       max_length=512,
                                       task=None,
                                       label_list=None,
                                       output_mode=None,
                                       pad_on_left=False,
                                       pad_token=0,
                                       pad_token_segment_id=0,
                                       mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: toxic task, which is only classification
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = toxic_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = toxic_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    logger.info("Using label list %s for task %s" % (label_list, task))
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        label_ids = None
        if output_mode == "classification":
            # print(example.guid)
            # print(example.text_a)
            # print(example.label)
            assert type(example.label) is list, "Examples should provide already binarized labels"
            # logger.info("Managing multi-labeled datasets")
            label_ids = []
            label = ""
            for current_label, is_on  in zip(label_list, example.label):  # support for multiple positive labels needed for sigmoid activation
                # print(is_on)
                if is_on:
                    label_ids.append(label_map[current_label])
                    label = label + " " + current_label
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("Text a: {}\nText b: {}".format(example.text_a, example.text_b))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("Example label/label ids: {}/{}".format(example.label, label_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label,
                          label_ids=label_ids))

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield ({'input_ids': ex.input_ids,
                        'attention_mask': ex.attention_mask,
                        'token_type_ids': ex.token_type_ids},
                       ex.label)

        return tf.data.Dataset.from_generator(gen,
                                              ({'input_ids': tf.int32,
                                                'attention_mask': tf.int32,
                                                'token_type_ids': tf.int32},
                                               tf.int64),
                                              ({'input_ids': tf.TensorShape([None]),
                                                'attention_mask': tf.TensorShape([None]),
                                                'token_type_ids': tf.TensorShape([None])},
                                               tf.TensorShape([])))

    return features


class ToxicClasssificationProcessor(DataProcessor):
    """Processor for the classification data set (Clickbait version)."""

    def __init__(self):
        self.train_label_freq_dist_dict = None

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        train_file = os.path.join(data_dir, "train_100.csv")
        logger.info("{} Loading Train Dataset from {}".format(os.getcwd(), train_file))
        toxic_comments_df = pd.read_csv(train_file)
        df, input_examples = self._create_examples(toxic_comments_df, "train")
        toxic_comments_labels = df[toxic_classes_plus_non_toxic_class]
        self.train_label_freq_dist_dict = dict(zip(toxic_classes_plus_non_toxic_class, toxic_comments_labels.sum(axis=0).values))

        return input_examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        test_file = os.path.join(data_dir, "test_100.csv")
        ground_truth_file = os.path.join(data_dir, "test_labels_100.csv")
        logger.info("Loading Ground Truth File as list from {}".format(ground_truth_file))
        ground_truth_list = [line.rstrip('\n') for line in open(ground_truth_file)]
        logger.info("Loading Dev Dataset from {}".format(test_file))
        test_df = pd.read_csv(test_file)
        ground_truth_df = pd.read_csv(ground_truth_file)
        join_test_df = test_df.join(ground_truth_df.set_index('id'), on='id')
        # join_test_df.drop(join_test_df[join_test_df['toxic'] == -1].index, inplace=True)
        df, input_examples = self._create_examples(join_test_df, "test", ground_truth_list)
        return input_examples

    def get_labels(self):
        """See base class."""
        return toxic_classes_plus_non_toxic_class

    def _create_examples(self, df, set_type, ground_truth_list=None):
        """Creates examples for the training and dev sets."""
        logger.info("Number of examples: {}".format(df.shape[0]))
        df['non_toxic'] = df.apply(lambda row: 1 if sum(row[toxic_classes].values) == 0 else 0, axis=1)
        print(df.head(1))
        df['input_examples'] = df.apply(lambda row: InputExample(guid=row['id'],
                                                                 text_a=row['comment_text'],
                                                                 text_b=None,
                                                                 label=row[toxic_classes_plus_non_toxic_class].values.tolist()), axis=1)
        return (df, df['input_examples'].values)


toxic_tasks_num_labels = {
    "classification": len(toxic_classes_plus_non_toxic_class),
}

toxic_processors = {
    "classification": ToxicClasssificationProcessor,
}

toxic_output_modes = {
    "classification": "classification",
}
