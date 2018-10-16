"""
Generate Carlini Wagner L2 attack examples across a range of parameters against Inception V3
Examples are generated using data provided in the NIPS 2017 Adversarial Competition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import csv
import itertools
import logging
import os

import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image
from cleverhans.attacks import CarliniWagnerL2, SPSA
from cleverhans.model import Model
from six.moves import xrange
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception

from config import attack_name_to_configurable_params, attack_name_to_params, \
    attack_to_prefix_template, attack_name_prefix
from constants import ATTACKS

logger = logging.getLogger(__name__)

DEFAULT_INCEPTION_PATH = os.path.join(
    ('nips17_adversarial_competition/dev_toolkit/sample_attacks/fgsm/'
     'inception_v3.ckpt'))

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path',
    DEFAULT_INCEPTION_PATH, 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_image_dir',
    os.path.join('nips17_adversarial_competition/dataset/images'),
    'Path to image directory.')

tf.flags.DEFINE_string(
    'metadata_file_path',
    os.path.join('nips17_adversarial_competition/dataset/dev_dataset.csv'),
    'Path to metadata file.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, metadata_file_path, batch_shape, num_classes):
    """Retrieve numpy arrays of images and labels, read from a directory."""
    num_images = batch_shape[0]
    with open(metadata_file_path) as input_file:
        reader = csv.reader(input_file)
        header_row = next(reader)
        rows = list(reader)

    row_idx_image_id = header_row.index('ImageId')
    row_idx_true_label = header_row.index('TrueLabel')
    row_idx_target_label = header_row.index('TargetClass')
    images = np.zeros(batch_shape)
    labels = np.zeros((num_images, num_classes), dtype=np.int32)
    target_classes = np.zeros((num_images, num_classes), dtype=np.int32)
    for idx in xrange(num_images):
        row = rows[idx]
        filepath = os.path.join(input_dir, row[row_idx_image_id] + '.png')

        with tf.gfile.Open(filepath, 'rb') as f:
            image = np.array(
                Image.open(f).convert('RGB')).astype(np.float) / 255.0
        images[idx, :, :, :] = image
        labels[idx, int(row[row_idx_true_label])] = 1
        target_classes[idx, int(row[row_idx_target_label])] = 1

    return images, labels, target_classes


class InceptionModel(Model):
    """Model class for CleverHans library."""

    def __init__(self, nb_classes):
        super(InceptionModel, self).__init__()
        self.nb_classes = nb_classes
        self.num_classes = nb_classes
        self.built = False

    def __call__(self, x_input, return_logits=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            # Inception preprocessing uses [-1, 1]-scaled input.
            x_input = x_input * 2.0 - 1.0
            _, end_points = inception.inception_v3(
                x_input, num_classes=self.nb_classes, is_training=False,
                reuse=reuse)
        self.built = True
        self.logits = end_points['Logits']
        # Strip off the extra reshape op at the output
        self.probs = end_points['Predictions'].op.inputs[0]
        if return_logits:
            return self.logits
        else:
            return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input)


def _top_1_accuracy(logits, labels):
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))


def expand_param_dict(params, configurable_params):
    """
    Create a list of param dictionaries based on the variables to loop over

    Args:
        params (dict): The parameters
        configurable_params (list): list of strings

    Returns:
        (list) dictionaries
    """
    values_to_search_over = []
    for c_param in configurable_params:
        values_to_search_over.append(params[c_param])

    list_param_dict = []
    for comb in itertools.product(*values_to_search_over):
        copy_param = copy.copy(params)
        for param_idx, param in enumerate(comb):
            copy_param[configurable_params[param_idx]] = param
        list_param_dict.append(copy_param)

    logger.info('Returning the following parameters: {}'.format(list_param_dict))

    return list_param_dict


def get_attack_images_filename_prefix(attack_name, params, model, targeted_prefix):
    prefix = attack_name_prefix.format(**{'targeted_prefix': targeted_prefix,
                                          'attack_name': attack_name,
                                          'model': model})
    attack_configs_name = attack_to_prefix_template[attack_name].format(**params)
    attack_images_file_name_prefix = prefix + attack_configs_name

    return attack_images_file_name_prefix

def test_inference():
    """Check model is accurate on unperturbed images."""
    input_dir = FLAGS.input_image_dir
    metadata_file_path = FLAGS.metadata_file_path
    num_images = 16
    batch_shape = (num_images, 299, 299, 3)
    num_classes = 1001
    images, labels, target_labels = load_images(
        input_dir, metadata_file_path, batch_shape, num_classes)

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        y_label = tf.placeholder(tf.int32, shape=(num_images,))
        model = InceptionModel(num_classes)
        logits = model.get_logits(x_input)
        acc = _top_1_accuracy(logits, y_label)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())

        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path,
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            acc_val = sess.run(acc, feed_dict={x_input: images, y_label: np.argmax(labels, axis=1)})
            tf.logging.info('Accuracy: %s', acc_val)
            assert acc_val > 0.8

def iterate_through_cwl2_attacks():
    tf.logging.set_verbosity(tf.logging.INFO)
    input_dir = FLAGS.input_image_dir
    metadata_file_path = FLAGS.metadata_file_path
    num_images = len(os.listdir(input_dir))
    batch_shape = (num_images, 299, 299, 3)
    num_classes = 1001
    batch_size = attack_name_to_params[ATTACKS.CARLINI_WAGNER]['batch_size']
    images, labels, target_classes = load_images(input_dir, metadata_file_path, batch_shape,
                                                 num_classes)

    list_param_dict = expand_param_dict(
        attack_name_to_params[ATTACKS.CARLINI_WAGNER],
        attack_name_to_configurable_params[ATTACKS.CARLINI_WAGNER]
    )

    save_dir = 'saves'
    os.makedirs(save_dir, exist_ok=True)

    for idx, params in enumerate(list_param_dict):
        tf.reset_default_graph()

        logger.info('Running attack with parameters: {}'.format(params))
        logger.info('Current index of parameters: {}/{}'.format(idx, len(list_param_dict)))

        # Get save path
        adv_imgs_save_path = get_attack_images_filename_prefix(
            attack_name=ATTACKS.CARLINI_WAGNER,
            params=params,
            model='inception',
            targeted_prefix='targeted'
        )
        adv_imgs_save_path = os.path.join(save_dir, adv_imgs_save_path)

        # Run inference
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session(graph=graph)
            # Prepare graph
            x_input = tf.placeholder(tf.float32, shape=(batch_size,) + batch_shape[1:])
            y_label = tf.placeholder(tf.int32, shape=(batch_size, num_classes))
            y_target = tf.placeholder(tf.int32, shape=(batch_size, num_classes))
            model = InceptionModel(num_classes)

            cwl2 = True
            if cwl2:
                attack = CarliniWagnerL2(model=model, sess=sess)
                x_adv = attack.generate(x_input, y=y_target, **params)
            else:
                attack = SPSA(model=model)
                x_adv = attack.generate(x_input, y=y_label, epsilon=4./255, num_steps=30,
                  early_stop_loss_threshold=-1., batch_size=32, spsa_iters=16,
                  is_debug=True)

            # logits = model.get_logits(x_adv)
            # acc = _top_1_accuracy(logits, tf.argmax(y_label, axis=1))

            # Run computation
            saver = tf.train.Saver(slim.get_model_variables())
            saver.restore(sess, save_path=FLAGS.checkpoint_path)

            list_adv_images = []

            if num_images % batch_size == 0:
                num_batches = int(num_images / batch_size)
            else:
                num_batches = int(num_images / batch_size + 1)

            for i in tqdm.tqdm(range(num_batches)):
                feed_dict_i = {x_input: images[i*batch_size:(i+1)*batch_size],
                               y_target: target_classes[i*batch_size:(i+1)*batch_size]}
                adv_img = sess.run(x_adv, feed_dict=feed_dict_i)
                list_adv_images.append(adv_img)

            adv_images = np.concatenate((list_adv_images))
            np.save(adv_imgs_save_path, adv_images)

        # tf.reset_default_graph()
        # with tf.Graph().as_default():
        #     # Prepare graph
        #     x_input = tf.placeholder(tf.float32, shape=(1,) + batch_shape[1:])
        #     y_label = tf.placeholder(tf.int32, shape=(1, num_classes))
        #     # y_target = tf.placeholder(tf.int32, shape=(1,))
        #
        #     # logits = model.get_logits(x_adv)
        #     # acc = _top_1_accuracy(logits, y_label)
        #     model = InceptionModel(num_classes)
        #     _ = model.get_logits(x_input)
        #
        #     attack = CarliniWagnerL2(model)
        #     x_adv = attack.generate(x=x_input, y=y_label, **params)
        #     logger.info('x_adv is: {}'.format(x_adv))
        #
        #     # Run computation
        #     saver = tf.train.Saver(slim.get_model_variables())
        #     session_creator = tf.train.ChiefSessionCreator(
        #         scaffold=tf.train.Scaffold(saver=saver),
        #         checkpoint_filename_with_path=FLAGS.checkpoint_path,
        #         master=FLAGS.master)
        #
        #     with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        #
        #         list_adv_images = []
        #
        #         num_correct = 0.
        #         for i in xrange(num_images):
        #             image = images[i]
        #             feed_dict_i = {x_input: np.expand_dims(image, axis=0),
        #                            y_label: np.expand_dims(np.argmax(labels[i]), axis=0)}
        #             adv_image = sess.run([x_adv], feed_dict=feed_dict_i)
        #             list_adv_images.append(adv_image)
        #             # num_correct += acc_val
        #
        #         tf.logging.info('Parameters of attack: {}'.format(params))
        #         # tf.logging.info('Success Rate: %s', num_correct / float(num_images))
        #
        #         # Save the images
        #         adv_images = np.concatenate((list_adv_images))
        #
        #         logger.info('Saving to: {}'.format(adv_imgs_save_path))
        #
        #     # np.save(adv_imgs_save_path, adv_images)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    # test_inference()
    iterate_through_cwl2_attacks()
