# 04/10
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import ceil, floor
import argparse
from math import pi
from PIL import Image
import os
import cv2


IMAGE_SIZE = 224
IMAGES = 4


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path_', type=str,
                        default='/home/enningxie/Documents/DataSets/crop_img')
    parser.add_argument('--test_path', type=str,
                        default='/home/enningxie/Documents/DataSets/butterfly_/convert_')
    parser.add_argument('--test_result', type=str,
                        default='/home/enningxie/Documents/DataSets/butterfly_/test_path')
    parser.add_argument('--test_convert', type=str,
                        default='/home/enningxie/Documents/DataSets/butterfly_/test_convert')
    parser.add_argument('--file_path', type=str, default='/var/Data/xz/butterfly/process_data')
    parser.add_argument('--process_data', type=str, default='/var/Data/xz/butterfly/data_augmentation')
    return parser.parse_args()


# return [img_file_paths]
def img_file_path(path):
    img_pathes = []
    for path_ in os.listdir(path):
        img_pathes.append(os.path.join(path, path_))
    return img_pathes


# save img
def img_save_op(img_data, img_label, path, step):
    for index, img in enumerate(img_data):
        mpimg.imsave(os.path.join(path, img_label[index] + '_' + str(step) + '_' + str(index) + '.png'), img)
    print('done.')


# save img
def img_save_op_(img_data, path, step):
    for index, img in enumerate(img_data):
        mpimg.imsave(os.path.join(path, str(step) + '_' + str(index) + '.png'), img)
    print('done.')


def convert_jpg_to_png(path, to_path):
    for filename in os.listdir(path):
        if os.path.splitext(filename)[-1] == '.jpg':
            # print(filename)
            img = cv2.imread(os.path.join(path, filename))
            # img = Image.open(path + '\\' + filename)
            # # print(filename.replace(".jpg", ".png"))
            newfilename = filename.replace(".jpg", ".png")
            # img.save(to_path + '\\' + newfilename)
            # # cv2.imshow("Image",img)
            # # cv2.waitKey(0)
            cv2.imwrite(os.path.join(to_path, newfilename), img)
    print('convert done.')


# resize the images
def tf_resize_images(X_img_file_paths):
    X_data = []
    X_label = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE),
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Each image is resized individually as different image may be of different size.
        for index, file_path in enumerate(X_img_file_paths):
            img = mpimg.imread(file_path)[:, :, :3]  # Do not read alpha channel.
            resized_img = sess.run(tf_img, feed_dict={X: img})
            X_data.append(resized_img)
            X_label.append(file_path.split('/')[-1][:11])

    X_data = np.array(X_data, dtype=np.float32)  # Convert to numpy
    X_label = np.array(X_label, dtype=np.str)
    return X_data, X_label


# scaling
def central_scale_images(X_imgs, X_label, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype=np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
    box_ind = np.zeros((len(scales)), dtype=np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype=np.int32)

    X_scale_data = []
    X_label_ = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE, IMAGE_SIZE, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index, img_data in enumerate(X_imgs):
            batch_img = np.expand_dims(img_data, axis=0)
            scaled_imgs = sess.run(tf_img, feed_dict={X: batch_img})
            X_scale_data.extend(scaled_imgs)
            for _ in range(len(scales)):
                X_label_.append(X_label[index])


    X_scale_data = np.array(X_scale_data, dtype=np.float32)
    X_label_ = np.array(X_label_, dtype=np.str)
    return X_scale_data, X_label_


# Translation
def get_translate_parameters(index):
    if index == 0:  # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1:  # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2:  # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE))
    else:  # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE

    return offset, size, w_start, w_end, h_start, h_end


def translate_images(X_imgs, X_label):
    offsets = np.zeros((len(X_imgs), 2), dtype=np.float32)
    n_translations = 4
    X_translated_arr = []
    X_label_ = []

    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, 3),
                                    dtype=np.float32)
            X_translated.fill(1.0)  # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)

            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], \
            w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
            X_label_.extend(X_label)

    X_translated_arr = np.array(X_translated_arr, dtype=np.float32)
    X_label_ = np.array(X_label_, dtype=np.str)
    return X_translated_arr, X_label_


# Rotation at 90 degrees
def rotate_images(X_imgs, X_label):
    X_rotate = []
    X_label_ = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k=k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, img in enumerate(X_imgs):
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict={X: img, k: i + 1})
                X_rotate.append(rotated_img)
                X_label_.append(X_label[index])

    X_rotate = np.array(X_rotate, dtype=np.float32)
    X_label_ = np.array(X_label_, dtype=np.str)
    return X_rotate, X_label_


# Rotation at finer angles
def rotate_images_(X_imgs, X_label, start_angle, end_angle, n_images):
    X_rotate = []
    X_label_ = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    radian = tf.placeholder(tf.float32, shape=(len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict={X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)
            X_label_.extend(X_label)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    X_label_ = np.array(X_label_, dtype=np.str)
    return X_rotate, X_label_


# flipping
def flip_images(X_imgs, X_label):
    X_flip = []
    X_label_ = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, img in enumerate(X_imgs):
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict={X: img})
            X_flip.extend(flipped_imgs)
            for _ in range(3):
                X_label_.append(X_label[index])
    X_flip = np.array(X_flip, dtype=np.float32)
    X_label_ = np.array(X_label_, dtype=np.str)
    return X_flip, X_label_


# Adding Salt and Pepper noise
def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy


# Lighting condition by adding Gaussian noise
def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    return gaussian_noise_imgs


# Perspective transform
def get_mask_coord(imshape):
    vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]),
                          (0.43 * imshape[1], 0.32 * imshape[0]),
                          (0.56 * imshape[1], 0.32 * imshape[0]),
                          (0.85 * imshape[1], 0.99 * imshape[0])]], dtype=np.int32)
    return vertices


def get_perspective_matrices(X_img):
    offset = 15
    img_size = (X_img.shape[1], X_img.shape[0])

    # Estimate the coordinates of object of interest inside the image.
    src = np.float32(get_mask_coord(X_img.shape))
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]]])

    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    return perspective_matrix


def perspective_transform(X_img):
    # Doing only for one type of example
    perspective_matrix = get_perspective_matrices(X_img)
    warped_img = cv2.warpPerspective(X_img, perspective_matrix,
                                     (X_img.shape[1], X_img.shape[0]),
                                     flags=cv2.INTER_LINEAR)
    return warped_img


if __name__ == '__main__':
    FLAGS = arg_parse()
    origin_path = FLAGS.file_path
    to_path = FLAGS.process_data
    # step -1
    # convert_jpg_to_png(FLAGS.test_path, FLAGS.test_convert)
    path = img_file_path(origin_path)
    # print(path[0].split('/')[-1][:11])
    # step 0: resize
    X_imgs, X_label0 = tf_resize_images(path)
    img_save_op(X_imgs, X_label0, to_path, 0)
    # step 1: scale
    # Produce each image at scaling of 90%, 75% and 60% of original image.
    scaled_imgs, X_label1 = central_scale_images(X_imgs, X_label0, [0.90, 0.75, 0.60])
    img_save_op(scaled_imgs, X_label1, to_path, 1)
    # # step 2: Translate
    translated_imgs, X_label2 = translate_images(X_imgs, X_label0)
    img_save_op(translated_imgs, X_label2, to_path, 2)
    # step 3: Rotation at 90 degrees
    rotated_imgs, X_label3 = rotate_images(X_imgs, X_label0)
    img_save_op(rotated_imgs, X_label2, to_path, 3)
    # step 4: Rotation at finer angles
    # Start rotation at -90 degrees, end at 90 degrees and produce totally 14 images
    rotated_imgs_, X_label4 = rotate_images_(X_imgs, X_label0, -90, 90, 14)
    img_save_op(rotated_imgs_, X_label4, to_path, 4)
    # step 5: flipping
    flipped_images, X_label5 = flip_images(X_imgs, X_label0)
    img_save_op(flipped_images, X_label5, to_path, 5)
    # step 6: Adding Salt and Pepper noise
    salt_pepper_noise_imgs = add_salt_pepper_noise(X_imgs)
    img_save_op(salt_pepper_noise_imgs, X_label0, to_path, 6)
    # step 7: Lighting condition by adding Gaussian noise
    gaussian_noise_imgs = add_gaussian_noise(X_imgs)
    img_save_op(gaussian_noise_imgs, X_label0, to_path, 7)
    # # step 8:
    # # perspective_img = perspective_transform(X_imgs[0])
    # # img_save_op(perspective_img, FLAGS.process_data, 8)





