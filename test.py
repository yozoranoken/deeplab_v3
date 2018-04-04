#! /usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.ops import control_flow_ops

from metrics import *
import network
from preprocessing.read_data import tf_record_parser, scale_image_with_crop_padding
from preprocessing import training


plt.interactive(False)

# set logging configurations
log = logging.getLogger('test')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('test.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Eval params')

envarg.add_argument(
    '--log-folder',
    type=Path,
    required=True,
    help='Path to Log Folder (contains train and val directories)',
)

envarg.add_argument(
    '--max-iterations',
    type=int,
    help='Maximum iterations to perform.',
)

envarg.add_argument(
    '--data-dir',
    type=Path,
    required=True,
    help='Path to directory containing TFRecords',
)

envarg.add_argument(
    '--use-cpu',
    action='store_true',
    help='Force use of CPU.',
)

input_args = parser.parse_args()

# best: 16645
# model_name = str(input_args.model_id)

# uncomment and set the GPU id if applicable.
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

# log_folder = './tboard_logs'
log_folder_path = input_args.log_folder
train_log_folder_path = log_folder_path / 'train'

# if not os.path.exists(os.path.join(log_folder, model_name, "test")):
#     os.makedirs(os.path.join(log_folder, model_name, "test"))

# with open(log_folder + '/' + model_name + '/train/data.json', 'r') as fp:
with open(str(train_log_folder_path / 'data.json')) as fp:
    args = json.load(fp)

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

args = Dotdict(args)

# 0 = negative
# 1 = positive
# 255 = unknown

class_labels = [v for v in range(args.number_of_classes + 1)]
class_labels[-1] = 255

# LOG_FOLDER = './tboard_logs'
# TEST_DATASET_DIR="./dataset/"
TEST_FILE = 'test.tfrecords'

test_filenames = [str(input_args.data_dir / TEST_FILE)]
test_dataset = tf.data.TFRecordDataset(test_filenames)
test_dataset = test_dataset.map(tf_record_parser)  # Parse the record into tensors.
test_dataset = test_dataset.map(scale_image_with_crop_padding)
test_dataset = test_dataset.shuffle(buffer_size=100)
test_dataset = test_dataset.batch(args.batch_size)

iterator = test_dataset.make_one_shot_iterator()
batch_images_tf, batch_labels_tf, batch_shapes_tf = iterator.get_next()

logits_tf =  network.deeplab_v3(
    batch_images_tf,
    args,
    is_training=False,
    reuse=False,
)

valid_labels_batch_tf, valid_logits_batch_tf = (
    training.get_valid_logits_and_labels(
        annotation_batch_tensor=batch_labels_tf,
        logits_batch_tensor=logits_tf,
        class_labels=class_labels,
    )
)

cross_entropies_tf = tf.nn.softmax_cross_entropy_with_logits(
    logits=valid_logits_batch_tf,
    labels=valid_labels_batch_tf,
)

cross_entropy_mean_tf = tf.reduce_mean(cross_entropies_tf)
tf.summary.scalar('cross_entropy', cross_entropy_mean_tf)

predictions_tf = tf.argmax(logits_tf, axis=3)
probabilities_tf = tf.nn.softmax(logits_tf)

merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver()


config_proto_kwargs = {}

if input_args.use_cpu:
    config_proto_kwargs['device_count'] = {'GPU': 0}

config = tf.ConfigProto(**config_proto_kwargs)


SIDE_2D = 513
IMAGE_DIM_2D = (SIDE_2D, SIDE_2D)
UNKNOWN_LABEL = 255


with tf.Session(config=config) as sess:

    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Restore variables from disk.
    incumbent = tf.train.latest_checkpoint(str(train_log_folder_path))
    saver.restore(sess, incumbent)
    log.info('Restored saved model from %s', incumbent)

    mean_IoU = []
    mean_pixel_acc = []
    mean_freq_weighted_IU = []
    mean_acc = []

    iterations = 0
    max_iterations = input_args.max_iterations or sys.maxsize
    while iterations < max_iterations:
        try:
            (
                batch_images_np,
                batch_predictions_np,
                batch_labels_np,
                batch_shapes_np,
                summary_string,
            ) = sess.run([
                batch_images_tf,
                predictions_tf,
                batch_labels_tf,
                batch_shapes_tf,
                merged_summary_op,
            ])
            heights, widths = batch_shapes_np

            if input_args.max_iterations is not None:
                log.info('Iteration %s of %s done', iterations, max_iterations)
            else:
                log.info('Iteration %s done', iterations)

            # loop through the images in the batch and extract the valid areas
            # from the tensors
            for i in range(batch_predictions_np.shape[0]):

                label_image = batch_labels_np[i]
                pred_image = batch_predictions_np[i]
                input_image = batch_images_np[i]

                indices = np.where(label_image != UNKNOWN_LABEL)
                label_image = label_image[indices]
                pred_image = pred_image[indices]
                input_image = input_image[indices]

                if label_image.shape[0] == SIDE_2D**2:
                    label_image = np.reshape(label_image, IMAGE_DIM_2D)
                    pred_image = np.reshape(pred_image, IMAGE_DIM_2D)
                    input_image = np.reshape(input_image, IMAGE_DIM_2D + (3,))
                else:
                    label_image = np.reshape(
                        label_image, (heights[i], widths[i]))
                    pred_image = np.reshape(
                        pred_image, (heights[i], widths[i]))
                    input_image = np.reshape(
                        input_image, (heights[i], widths[i], 3))

                pix_acc = pixel_accuracy(pred_image, label_image)
                m_acc = mean_accuracy(pred_image, label_image)
                IoU = mean_IU(pred_image, label_image)
                freq_weighted_IU = frequency_weighted_IU(
                    pred_image, label_image)

                mean_pixel_acc.append(pix_acc)
                mean_acc.append(m_acc)
                mean_IoU.append(IoU)
                mean_freq_weighted_IU.append(freq_weighted_IU)

                #f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 8))

                #ax1.imshow(input_image.astype(np.uint8))
                #ax2.imshow(label_image)
                #ax3.imshow(pred_image)
                #plt.show()

                iterations += 1

        except tf.errors.OutOfRangeError:
            break

    log.info('Mean pixel accuracy: -------- %s', np.mean(mean_pixel_acc))
    log.info('Mean accuraccy: ------------- %s', np.mean(mean_acc))
    log.info('Mean IoU: ------------------- %s', np.mean(mean_IoU))
    log.info('Mean frequency weighted IU: - %s',
             np.mean(mean_freq_weighted_IU))
