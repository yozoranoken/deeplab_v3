#! /usr/bin/env python3
import argparse
import json
import logging
import network
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from preprocessing import training
from preprocessing.read_data import tf_record_parser
from preprocessing.read_data import rescale_image_and_annotation_by_factor
from preprocessing.read_data import scale_image_with_crop_padding
from preprocessing.read_data import random_flip_image_and_annotation
from preprocessing.read_data import distort_randomly_image_color


# set logging configurations
log = logging.getLogger('train')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('train.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)


parser = argparse.ArgumentParser()
envarg = parser.add_argument_group('Training params')

envarg.add_argument(
    '--data-dir',
    type=str,
    required=True,
    help='Location of dataset directory.',
)

envarg.add_argument(
    '--log-dir',
    type=Path,
    required=True,
    help='Location of log directory.',
)

envarg.add_argument(
    '--log-folder-name',
    type=str,
    help='Name of the log folder',
    default='TFLogs',
)


envarg.add_argument(
    '--batch-norm-epsilon',
    type=float,
    default=1e-5,
    help='batch norm epsilon argument for batch normalization',
)

envarg.add_argument(
    '--batch-norm-decay',
    type=float,
    default=0.9997,
    help='batch norm decay argument for batch normalization.',
)

envarg.add_argument(
    '--number-of-classes',
    type=int,
    default=2,
    help='Number of classes to be predicted.',
)

envarg.add_argument(
    '--l2-regularizer',
    type=float,
    default=1e-4,
    help='l2 regularizer parameter.',
)

envarg.add_argument(
    '--starting-learning-rate',
    type=float,
    default=1e-5,
    help='initial learning rate.',
)

envarg.add_argument(
    '--multi-grid',
    type=list,
    default=[1,2,4],
    help='Spatial Pyramid Pooling rates',
)

envarg.add_argument(
    '--output-stride',
    type=int,
    default=16,
    help='Spatial Pyramid Pooling rates',
)

envarg.add_argument(
    '--gpu-id',
    type=int,
    default=0,
    help='Id of the GPU to be used',
)

envarg.add_argument(
    '--use-cpu',
    action='store_true',
    help='Force use of CPU.',
)

envarg.add_argument(
    '--resnet-model',
    default='resnet_v2_101',
    choices=[
        'resnet_v2_50',
        'resnet_v2_101',
        'resnet_v2_152',
        'resnet_v2_200',
    ],
    help=('Resnet model to use as feature extractor. ' +
          'Choose one of: resnet_v2_50 or resnet_v2_101'),
)

envarg.add_argument(
    '--current-best-val-loss',
    type=int,
    default=99999,
    help='Best validation loss value.',
)

envarg.add_argument(
    '--accumulated-validation-miou',
    type=int,
    default=0,
    help='Accumulated validation intersection over union.',
)

trainarg = parser.add_argument_group('Training')
trainarg.add_argument(
    '--batch-size',
    type=int,
    default=8,
    help='Batch size for network train.',
)

trainarg.add_argument(
    '--train-steps-before-eval',
    type=int,
    default=100,
    help='Number of training steps to take before evaluation.',
)

trainarg.add_argument(
    '--num-validation-steps',
    type=int,
    default=20,
    help='Number of validation steps.',
)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

BASE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_PATH = BASE_PATH / 'resnet' / 'checkpoints'

TRAIN_DATASET_DIR = args.data_dir
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'


config_proto_kwargs = {}

if args.use_cpu:
    config_proto_kwargs['device_count'] = {'GPU': 0}

config = tf.ConfigProto(**config_proto_kwargs)


training_filenames = [os.path.join(TRAIN_DATASET_DIR,TRAIN_FILE)]
training_dataset = tf.data.TFRecordDataset(training_filenames)
training_dataset = training_dataset.map(tf_record_parser)
training_dataset = training_dataset.map(rescale_image_and_annotation_by_factor)
training_dataset = training_dataset.map(distort_randomly_image_color)
training_dataset = training_dataset.map(scale_image_with_crop_padding)
training_dataset = training_dataset.map(random_flip_image_and_annotation)  # Parse the record into tensors.
training_dataset = training_dataset.repeat()  # number of epochs
training_dataset = training_dataset.shuffle(buffer_size=500)
training_dataset = training_dataset.batch(args.batch_size)

validation_filenames = [os.path.join(TRAIN_DATASET_DIR,VALIDATION_FILE)]
validation_dataset = tf.data.TFRecordDataset(validation_filenames)
validation_dataset = validation_dataset.map(tf_record_parser)  # Parse the record into tensors.
validation_dataset = validation_dataset.map(scale_image_with_crop_padding)
validation_dataset = validation_dataset.shuffle(buffer_size=100)
validation_dataset = validation_dataset.batch(args.batch_size)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(
    handle,
    training_dataset.output_types,
    training_dataset.output_shapes,
)
batch_images_tf, batch_labels_tf, _ = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_initializable_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

is_training_tf = tf.placeholder(tf.bool, shape=[])

logits_tf = tf.cond(
    is_training_tf,
    true_fn=lambda: network.deeplab_v3(
        batch_images_tf,
        args,
        is_training=True,
        reuse=False,
    ),
    false_fn=lambda: network.deeplab_v3(
        batch_images_tf,
        args,
        is_training=False,
        reuse=True,
    )
)

# get valid logits and labels (factor the 255 padded mask out for cross entropy)
valid_labels_batch_tf, valid_logits_batch_tf = training.get_valid_logits_and_labels(
    annotation_batch_tensor=batch_labels_tf,
    logits_batch_tensor=logits_tf,
    class_labels=class_labels)

cross_entropies = tf.nn.softmax_cross_entropy_with_logits(
    logits=valid_logits_batch_tf,
    labels=valid_labels_batch_tf,
)
cross_entropy_tf = tf.reduce_mean(cross_entropies)
predictions_tf = tf.argmax(logits_tf, axis=3)

tf.summary.scalar('cross_entropy', cross_entropy_tf)
# tf.summary.image("prediction", tf.expand_dims(tf.cast(pred, tf.float32),3), 1)
# tf.summary.image("label", tf.expand_dims(tf.cast(batch_labels, tf.float32),3), 1)

with tf.variable_scope('optimizer_vars'):
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=args.starting_learning_rate)
    train_step = slim.learning.create_train_op(
        cross_entropy_tf, optimizer, global_step=global_step)

# Put all summary ops into one op. Produces string when you run it.
process_str_id = str(os.getpid())
merged_summary_op = tf.summary.merge_all()

variables_to_restore = slim.get_variables_to_restore(
    exclude=[
        args.resnet_model + '/logits',
        'optimizer_vars',
        'DeepLab_v3/ASPP_layer',
        'DeepLab_v3/logits'
    ],
)

miou, update_op = tf.contrib.metrics.streaming_mean_iou(
    tf.argmax(valid_logits_batch_tf, axis=1),
    tf.argmax(valid_labels_batch_tf, axis=1),
    num_classes=args.number_of_classes,
)
tf.summary.scalar('miou', miou)

# Add ops to restore all the variables.
restorer = tf.train.Saver(variables_to_restore)

saver = tf.train.Saver()

current_best_val_loss = np.inf

with tf.Session(config=config) as sess:
    # Create the summary writer -- to write all the tboard_log
    # into a specified file. This file can be later read
    # by tensorboard.
    log_dir_path = args.log_dir / args.log_folder_name
    train_dir_path = log_dir_path / 'train'

    train_writer = tf.summary.FileWriter(str(train_dir_path), sess.graph)
    test_writer = tf.summary.FileWriter(str(log_dir_path / 'val'))

    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    if (train_dir_path / 'checkpoint').is_file():
        incumbent = tf.train.latest_checkpoint(str(train_dir_path))
        saver.restore(sess, incumbent)
        log.info('Restored saved model from %s', incumbent)
    else:
        try:
            restorer.restore(
                sess, str(CHECKPOINTS_PATH / (args.resnet_model + '.ckpt')))
            log.info('Model checkpoints for %s restored!', args.resnet_model)
        except FileNotFoundError:
            log.error('Run "./download_resnet.sh" to download desired ' +
                      'resnet model.')

    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    sess.run(training_iterator.initializer)

    validation_running_loss = []

    train_steps_before_eval = args.train_steps_before_eval
    validation_steps = args.num_validation_steps
    while True:
        training_average_loss = 0

        # run this number of batches before validation
        for i in range(train_steps_before_eval):
            _, global_step_np, train_loss, summary_string = sess.run(
                [
                    train_step,
                    global_step, cross_entropy_tf,
                    merged_summary_op,
                ],
                feed_dict={
                    is_training_tf: True,
                    handle: training_handle,
                })

            training_average_loss += train_loss
            if i % 10 == 0:
                train_writer.add_summary(summary_string, global_step_np)

        training_average_loss /= train_steps_before_eval

        # at the end of each train interval, run validation
        sess.run(validation_iterator.initializer)

        validation_average_loss = 0
        validation_average_miou = 0
        for i in range(validation_steps):
            val_loss, summary_string, _= sess.run(
                [
                    cross_entropy_tf,
                    merged_summary_op,
                    update_op
                ],
                feed_dict={
                    handle: validation_handle,
                    is_training_tf: False,
                },
            )


            validation_average_loss+=val_loss
            validation_average_miou+=sess.run(miou)

        validation_average_loss /= validation_steps
        validation_average_miou /= validation_steps

        # keep running average of the miou and validation loss
        validation_running_loss.append(validation_average_loss)

        validation_global_loss = np.mean(validation_running_loss)

        if validation_global_loss < current_best_val_loss:
            # Save the variables to disk.
            save_path = saver.save(
                sess,
                str(log_dir_path / 'train' / 'model.ckpt'),
            )
            log.info('Saved model checkpoints')
            log.info('Best Average Loss: %s', validation_global_loss)
            current_best_val_loss = validation_global_loss

            # update metadata and save it
            args.current_best_val_loss = str(current_best_val_loss)

            with open(str(log_dir_path / 'train' / 'data.json'), 'w') as fp:
                args_dict = {k: str(v) for k, v in args.__dict__.items()}
                json.dump(args_dict, fp, sort_keys=True, indent=4)

        log.info(
            '\n' +
            'Global Step: ----------------- %s\n' +
            'Average Train Loss: ---------- %s\n' +
            'Global Validation Ave. Loss: - %s\n' +
            'MIoU: ------------------------ %s',
            global_step_np,
            training_average_loss,
            validation_global_loss,
            validation_average_miou,
        )

        test_writer.add_summary(summary_string, global_step_np)

    train_writer.close()
