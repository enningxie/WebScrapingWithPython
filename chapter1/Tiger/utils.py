# tools
import xml.etree.ElementTree as ET
import argparse
import os
import pandas as pd
from PIL import Image
import numpy as np


# parse
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_path',
                        default='/home/enningxie/Documents/DataSets/butterfly_/butt_train/train_set/Annotations')
    parser.add_argument('--new_annotations_path',
                        default='/home/enningxie/Documents/DataSets/butterfly_/butt_train/train_set/annotations')
    parser.add_argument('--class_path',
                        default='/home/enningxie/Documents/DataSets/butterfly_/butt_train/class_new.csv')
    parser.add_argument('--image_path',
                        default='/home/enningxie/Documents/DataSets/butterfly_/butt_train/train_set/JPEGImages')
    return parser.parse_args()


class Some_path(object):
    def __init__(self, args):
        self.annotations_path = args.annotations_path
        self.new_annotations_path = args.new_annotations_path
        self.image_path = args.image_path
        self.classes_pd = pd.read_csv(args.class_path)


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



if __name__ == '__main__':
    args = parse()
    sp = Some_path(args)
    rename_files(sp.annotations_path, sp.new_annotations_path, sp.image_path, sp.classes_pd)
    # test_xml('/home/enningxie/Documents/test.xml')