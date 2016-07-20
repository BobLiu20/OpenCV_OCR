#!/usr/bin/env python2.7
# coding=utf8

import numpy as np
import cv2
import cv2.cv as cv

import os
import copy

from opencv_ocr import opencv_ocr

if __name__ == '__main__' and __package__ is None:
    _opencv_ocr = opencv_ocr()
    # 使用SVM算法
    _svm_model = _opencv_ocr.svm_init('digits_svm.dat')
    # 从文件夹img_data中读取所有图片
    work_path = '%s/img_data/data4/' % os.getcwd()
    images_path = os.listdir(work_path)
    # 指定单张测试
    # images_path = ['10.jpg']
    # 循环从列表中取原始图片进行识别
    for index, _path in enumerate(images_path):
        # 读取图片
        if _path[_path.rfind('.'):] != '.jpg':
            continue
        image = cv2.imread(work_path + _path)
        if image.shape[0] > image.shape[1]:
            image = np.rot90(image, 1)
        # 字符串精确定位，包括歪斜修正
        img_binary = _opencv_ocr._character_location(image)
        # 分割每一个字符
        img_binary_copy = img_binary.copy()
        contours, hierarchy = cv2.findContours(
            img_binary_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 得到每一个字符的坐标
        _box_shape = []
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            _box_shape.append([x, x+w, y, y+h])
        _box_shape = sorted(_box_shape, key=lambda _box:_box[0])

        # 所有可能字符的list
        _image_list = [img_binary[_box[2]:_box[3],_box[0]:_box[1]] for _box in _box_shape]
        # Dynamic size
        _image_h_list = [_box[3]-_box[2] for _box in _box_shape]
        _median_h = int(np.median(_image_h_list))
        if _median_h < 6:
            _median_h = 6
        # 过滤掉一些不是字符的小图，并且将一些依旧未分割的连接字符分割
        _opencv_ocr._correct_char_image(_image_list, (_median_h-5, _median_h+5))
        # 将正确的字符图片列表丢到SVM模型里进行识别
        _string = _opencv_ocr._svm_classify_string(_svm_model, _image_list)
        print 'classify %s is :'%_path, _string



        # 输出单个字符图片用于训练
        for __idx, (__img, _ch) in enumerate(zip(_image_list, _string)):
             cv2.imwrite('%s/char_good/ch%d_%d_%s.jpg'%(os.getcwd(), index, __idx, _ch), __img)

        # 输出原图的二值化图像
        # if len(_string) != 0:
        #     for _idx, __img in enumerate(_image_list):
        #         cv2.imshow('img%d'%_idx, __img)
        #     cv2.imshow('imagefull', img_binary)
        #     cv2.waitKey()
