# tools
import xml.etree.ElementTree as ET
import argparse
import os
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np


# parse
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_path',
                        default='/home/enningxie/Documents/DataSets/butterfly_/butt_train/train_set/Annotations')
    parser.add_argument('--new_annotations_path',
                        default='/home/enningxie/Documents/DataSets/butterfly_/butt_train/train_set/annotations')
    parser.add_argument('--class_path',
                        default='/var/Data/xz/butterfly/classes_new_03.csv')
    parser.add_argument('--image_path',
                        default='/home/enningxie/Documents/DataSets/butterfly_/butt_train/train_set/JPEGImages')
    parser.add_argument('--image_for_classify',
                        default='/var/Data/xz/butterfly/crop_img')
    return parser.parse_args()


class Some_path(object):
    def __init__(self, args):
        self.annotations_path = args.annotations_path
        self.new_annotations_path = args.new_annotations_path
        self.image_path = args.image_path
        self.image_for_classify = args.image_for_classify
        self.classes_pd = pd.read_csv(args.class_path)

args = parse()
s = Some_path(args)


# read .xml
# return img_xxx & xx蝶
def rename_files(path, new_path, image_path, pd_data):
    classes_dict = {}
    for xml in os.listdir(path):
        tmp_path = os.path.join(path, xml)
        tmp_xml = ET.parse(tmp_path)
        root = tmp_xml.getroot()
        # root[1].text: filename, root[6][0].text: class_name
        img_name = root[1].text
        class_name = root[6][0].text
        ids = get_ids(pd_data, class_name)

        count = classes_dict.setdefault(ids, 0)
        count += 1
        classes_dict[ids] = count
        ids_ = ids + '_' + str(count)
        new_image_name = ids_ + '.jpg'
        root[1].text = new_image_name
        # 02
        for i in tmp_xml.findall('./object'):
            i[0].text = ids
        make_dir(new_path)
        # rename image
        os.rename(os.path.join(image_path, img_name), os.path.join(image_path, new_image_name))
        # rename xml
        tmp_xml.write(os.path.join(new_path, ids_ + '.xml'), encoding='utf-8')


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


# png convert to jpg
def png_to_jpg(source_file, to_file):
    tmp_img = Image.open(source_file)
    rgb_img = tmp_img.convert('RGB')
    rgb_img.save(to_file)


# given classes return ids
def get_ids(pd_data, class_name):
    return pd_data[pd_data.classes == class_name]['new_ids'].values[0]


def test_write_xml(path, args):
    tmp_xml = ET.parse(path)
    root = tmp_xml.getroot()
    print(root[1].text)
    root[1].text = 'lol'
    tmp_xml.write(os.path.join(args.annotations_path, 'test.xml'), encoding='utf-8')
    print('done.')


def test_png_to_jpg(path):
    tmp_img = Image.open(path)
    rgb_img = tmp_img.convert('RGB')
    rgb_img.save('/home/enningxie/Documents/DataSets/butterfly/test.jpg')
    print('done.')


def test_xml(path):
    test_xml_ = ET.parse(path)
    root = test_xml_.getroot()
    print(root[1].text)
    for i in test_xml_.findall('./object'):
        print(i[0].text)


# process image for classify
def _create_feature_label(path):
    classes_data = s.classes_pd
    fnames = []
    labels = []
    for image_name in os.listdir(path):
        class_ = image_name.split('_')[0]
        labels.append(int(classes_data[classes_data['new_ids'] == class_]['classes_num']))
        fnames.append(os.path.join(path, image_name))
    fnames = np.asarray(fnames)
    labels = np.asarray(labels)
    order = np.arange(len(labels))
    np.random.shuffle(order)
    fnames = fnames[order]
    labels = labels[order]
    return fnames, labels


def _parse_cell_image(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [500, 500])
    return image_resized, label


def _create_dataset(data, label, batch_size):
    filenames = tf.constant(data)
    labels = tf.constant(label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_cell_image)
    dataset = dataset.batch(batch_size)
    return dataset


def process_data(batch_size):
    train_data = _create_feature_label(s.image_for_classify)[0][:-20]
    train_data_labels = _create_feature_label(s.image_for_classify)[1][:-20]

    test_data = _create_feature_label(s.image_for_classify)[0][-20:]
    test_data_labels = _create_feature_label(s.image_for_classify)[1][-20:]
    # print(len(test_data[0]))
    train_dataset = _create_dataset(train_data, train_data_labels, batch_size)
    test_dataset = _create_dataset(test_data, test_data_labels, batch_size)

    return train_dataset, test_dataset


IMAGE_SIZE = 500


def tf_resize_images(X_img_file_paths):
    X_data = []
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

    X_data = np.array(X_data, dtype=np.float32)  # Convert to numpy
    return X_data

if __name__ == '__main__':
    pass

