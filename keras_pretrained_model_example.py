"""
Generate attacks on Keras pretrained ImageNet model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import csv
import logging
import os
import pprint
from abc import ABCMeta

import keras
import numpy as np
import tensorflow as tf
import tqdm
from cleverhans.model import Model, NoSuchLayerError
from cleverhans.utils_keras import KerasModelWrapper
from keras.preprocessing import image

from config import attack_name_to_params, attack_name_to_class
from constants import ATTACKS

logger = logging.getLogger(__name__)


# Define VGG16 model that is cleverhans-compliant
class VGG16(Model):
    __metaclass__ = ABCMeta

    def __init__(self):
        Model.__init__(self)
        from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
        self.keras_model = VGG16(weights='imagenet')
        self.model = KerasModelWrapper(self.keras_model)
        self.preprocess_input = preprocess_input
        self.decode_predictions = decode_predictions

    def get_logits(self, x):
        return self.model.get_logits(self.preprocess_input(x))

    def get_probs(self, x):
        return self.model.get_probs(self.preprocess_input(x))

    def get_layer(self, x, layer):
        output = self.model.fprop(self.preprocess_input(x))
        try:
            requested = output[layer]
        except KeyError:
            raise NoSuchLayerError()
        return requested

    def get_layer_names(self):
        """
        :return: Names of all the layers kept by Keras
        """
        layer_names = [x.name for x in self.keras_model.layers]
        return layer_names

    def predict(self, x, preprocess=False):
        if preprocess:
            return self.keras_model.predict(self.preprocess_input(x))
        else:
            return self.keras_model.predict(x)


def load_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    return image.img_to_array(img)


def load_nips_adv_comp_images(input_dir, metadata_file_path, num_classes):
    """
    Retrieve images from the NIPS 17 adversarial image challenge
    Args:
        input_dir:
        metadata_file_path:
        num_classes:

    Returns:

    """
    with open(metadata_file_path) as input_file:
        reader = csv.reader(input_file)
        header_row = next(reader)
        rows = list(reader)

    num_images = len(rows)
    row_idx_image_id = header_row.index('ImageId')
    row_idx_true_label = header_row.index('TrueLabel')
    row_idx_target_label = header_row.index('TargetClass')

    images = np.zeros((num_images, 224, 224, 3))
    labels = np.zeros((num_images, num_classes), dtype=np.int32)
    target_labels = np.zeros((num_images, num_classes), dtype=np.int32)

    for idx in range(num_images):
        row = rows[idx]
        filepath = os.path.join(input_dir, row[row_idx_image_id] + '.png')
        image = load_image(filepath)
        images[idx, :, :, :] = image
        # Make sure to subtract 1 from the labels
        labels[idx, int(row[row_idx_true_label]) - 1] = 1
        target_labels[idx, int(row[row_idx_target_label]) - 1] = 1

    return images, labels, target_labels


def convert_range_to_cwl2_range(x, old_max, old_min, new_max, new_min):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_x = (((x - old_min) * new_range) / old_range) + new_min
    return new_x


# Define attack
def attack_model(attack_name):
    """
    Attack the given model using CWL2

    Args:
        attack_name (str): Name of attack

    """
    # WARNING the the NIPS17 adv competition data has class range of 1 to 1000
    num_classes = 1000

    input_image_dir = 'nips17_adversarial_competition/dataset/images'
    metadata_file_path = 'nips17_adversarial_competition/dataset/dev_dataset.csv'

    # Load data
    images, labels, target_labels = load_nips_adv_comp_images(input_dir=input_image_dir,
                                                              metadata_file_path=metadata_file_path,
                                                              num_classes=num_classes)
    clean_labels_flattened = np.argmax(labels, axis=1)

    old_max = np.max(images)
    old_min = np.min(images)
    logger.info('Max pixel val of clean images: {}'.format(old_max))
    logger.info('Min pixel val of clean images: {}'.format(old_min))
    logger.info('Images shape: {}'.format(images.shape))
    num_images = images.shape[0]

    # Create TF session and set as Keras backend session
    logger.info('Number of images: {}'.format(num_images))
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)
        keras.backend.set_session(sess)
        model = VGG16()

        # Shape of ImageNet data for the keras models is (224, 224, 3)
        x_ph = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='x_input')
        y_target = tf.placeholder(tf.int32, shape=(None, num_classes), name='y_target')

        attack = attack_name_to_class[attack_name](model=model, sess=sess)
        params = attack_name_to_params[attack_name + '_quick']
        logger.info('Running {} attack using params:'.format(attack_name))
        if attack_name == ATTACKS.CARLINI_WAGNER:
            params['clip_min'] = old_min
            params['clip_max'] = old_max
        pprint.pprint(params)
        batch_size = params.get('batch_size', 20)
        x_adv = attack.generate(x_ph, y_target=y_target, **params)

        list_adv_images = []

        if images.shape[0] % batch_size == 0:
            num_batches = int(images.shape[0] / batch_size)
        else:
            num_batches = int(images.shape[0] / batch_size + 1)

        logger.info('Generating attacks for {} batches (batch size: {})'.format(
            num_batches, batch_size))
        clean_predictions = []
        for i in tqdm.tqdm(range(num_batches)):
            # if attack_name == ATTACKS.CARLINI_WAGNER:
            #     feed_dict_i = {
            #         x_ph: images_for_cwl2[i * batch_size:(i + 1) * batch_size],
            #         y_target: target_labels[i * batch_size:(i + 1) * batch_size]}
            # else:
            feed_dict_i = {
                x_ph: images[i * batch_size:(i + 1) * batch_size],
                y_target: target_labels[i * batch_size:(i + 1) * batch_size]}

            clean_prediction_batch = model.predict(
                images[i * batch_size:(i + 1) * batch_size], preprocess=True)
            clean_predictions.extend(clean_prediction_batch)
            adv_img = sess.run(x_adv, feed_dict=feed_dict_i)
            list_adv_images.append(adv_img)

        clean_predictions = np.argmax(clean_predictions, axis=1)
        logger.info('Clean accuracy (top-1) is {:.3f}'.format(
            np.mean(clean_predictions == clean_labels_flattened)))

        logger.info('Finished generating all adv images')
        adv_images = np.concatenate((list_adv_images))
        logger.info('Shape of adversarial images: {}'.format(adv_images.shape))

        logger.info('Getting predictions on adversarial image data')
        predictions = []

        # Run model predictions on adv_images
        for i in tqdm.tqdm(range(num_batches)):
            adv_image_batch = adv_images[i * batch_size:(i + 1) * batch_size]
            predict_batch = model.predict(adv_image_batch, preprocess=True)
            predictions.extend(predict_batch)

    predictions = np.argmax(np.array(predictions), axis=1)
    logger.info('Shape of predictions: {}'.format(predictions.shape))

    target_labels_flattened = np.argmax(target_labels, axis=1)

    logger.info("Success rate (top-1) is: {:.3f}".format(
        np.mean(predictions == target_labels_flattened)))

    logger.info('Accuracy (top-1) is {:.3f}'.format(
        np.mean(predictions == clean_labels_flattened)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, required=True, help='Name of attack')

    args = parser.parse_args()
    args = vars(args)

    attack = args['attack']

    attack_model(attack)
