#!/usr/bin/env python2.7
# coding=utf8

import numpy as np
from numpy.linalg import norm
import cv2
import cv2.cv as cv
import math
import os
import copy

CHAR_SIZE = 20

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()

class opencv_ocr(object):
    """docstring for opencv_ocr"""
    def __init__(self):
        super(opencv_ocr, self).__init__()

    def svm_init(self, classifier_fn):
        if not os.path.exists(classifier_fn):
            print '"%s" not found, run digits.py first' % classifier_fn
            return None
        model = SVM()
        model.load(classifier_fn)
        return model

    def _svm_classify_one_char(self, model, img):
        _h, _w = img.shape[:2]
        x, y, w, h = 0, 0, _w, _h
        #if not (16 <= h <= 64 and w <= 1.2 * h):
        #    print 'the size of image is incorrect.'

        m = img != 0
        if not 0.1 < m.mean() < 0.9:
            print 'incorrect mean.', m.mean()

        s = 1.5 * float(h) / CHAR_SIZE
        m = cv2.moments(img)
        c1 = np.float32([m['m10'], m['m01']]) / m['m00']
        c0 = np.float32([CHAR_SIZE / 2, CHAR_SIZE / 2])
        t = c1 - s * c0
        A = np.zeros((2, 3), np.float32)
        A[:, :2] = np.eye(2) * s
        A[:, 2] = t
        bin_norm = cv2.warpAffine(
            img, A, (CHAR_SIZE, CHAR_SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        bin_norm = self.deskew(bin_norm)
        if x + w + CHAR_SIZE < img.shape[1] and y + CHAR_SIZE < img.shape[0]:
            img[y:, x + w:][:CHAR_SIZE, :CHAR_SIZE] = bin_norm[..., np.newaxis]
        sample = self.preprocess_hog([bin_norm])
        digit = model.predict(sample)[0]
        return chr(digit)

    def _svm_classify_string(self, model, img_list):
        _str = ''
        for img in img_list:
            _str = _str + self._svm_classify_one_char(model, img)
        return _str

    def _split_special_char(self, image):
        _h, _w = image.shape[:2]
        _count_nonzero = []
        for _col in image.T:
            _count_nonzero.append(np.count_nonzero(_col))
        _min = min(_count_nonzero[_w / 4: _w *3 / 4])
        _index = _count_nonzero.index(_min, _w / 4, _w *3 / 4)
        _img_list = []
        _img_list.append(image[0:_h, _index:_w])
        _img_list.append(image[0:_h, 0:_index])
        return _img_list

    def _split_special_char_fine(self, image):
        _h, _w = image.shape[:2]
        _count_nonzero = []
        for _col in image.T:
            _count_nonzero.append(np.count_nonzero(_col))
        _min = min(_count_nonzero[_w / 4: _w *3 / 4])
        if _min < int(_h * 0.1):
            _index = _count_nonzero.index(_min, _w / 4, _w *3 / 4)
            _img_list = []
            _img_list.append(image[0:_h, _index:_w])
            _img_list.append(image[0:_h, 0:_index])
            return _img_list
        else:
            return None

    def _split_add_iter(self, _idx, _index, split_list, image_list, size_limit):
        for _split in split_list:
            _h1, _w1 = _split.shape[:2]
            if _h1 > size_limit[1] or _w1 > 1.2 * _h1:
                __split = self._split_special_char(_split)
                if len(__split) > 1 and __split[0] != [] and __split[1] != []:
                    _idx = self._split_add_iter(_idx, _index, __split, image_list, size_limit)
                continue
            elif _h1 < size_limit[0]:
                continue
            elif not (10 < _split.mean() < 240):
                continue
            image_list.insert(_index, _split)
            _idx += 1
        return _idx

    def _correct_char_image(self, image_list, size_limit):
        _idx = 0
        for _index, _img in enumerate(copy.copy(image_list)):
            _h, _w = _img.shape[:2]
            if not (10 < _img.mean() < 240):
                # print 'incorrect mean. remove it'
                del image_list[_idx]
                _idx -= 1
            # 粗分割
            elif not (size_limit[0] <= _h <= size_limit[1] and _w <= 1.2 * _h):
                del image_list[_idx]
                _idx -= 1
                _list_img = self._split_special_char(_img)
                # print 'incorrect size. try to split it to %d'%len(_list_img)
                _idx = self._split_add_iter(_idx, _index, _list_img, image_list, size_limit)
            # 细分割判断
            elif (size_limit[0] <= _h <= size_limit[1] and (1.0 * _h) <= _w <= (1.2 * _h)):
                _list_img = self._split_special_char_fine(_img)
                if _list_img:
                    del image_list[_idx]
                    _idx -= 1
                    for __img in _list_img:
                        image_list.insert(_idx, __img)
                        _idx += 1
            _idx += 1

    def _character_location(self, image, Precise=True):
        '''
        字符区域精确定位
        :param image:
        :return:
        '''
        if len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image.copy()

        h, w = img_gray.shape[:2]

        _, img_binary = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        if Precise:
            # 腐蚀
            kernel_1 = np.ones((3, 3), np.uint8)
            img_erosion = cv2.erode(img_binary, kernel_1, iterations=1)
            # 膨胀
            kernel_2 = np.ones((7, 7), np.uint8)
            img_dilate = cv2.dilate(img_erosion, kernel_2, iterations=2)
            # 提取轮廓
            contours, hierarchy = cv2.findContours(
                img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # 获取最大轮廓，这里的轮廓是不规则图形
            contour = max(contours, key=cv2.contourArea)
            # 获取包含轮廓的最大矩形，这个矩形有旋转角度的
            rect = cv2.minAreaRect(contour)
            # 获取有可能倾斜的矩形的四个角坐标
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = sorted(
                list(cv2.cv.BoxPoints(rect)))
            (x1, y1), (x2, y2) = ((x1, y1), (x2, y2)
                                  ) if y1 > y2 else ((x2, y2), (x1, y1))
            (x3, y3), (x4, y4) = ((x3, y3), (x4, y4)
                                  ) if y3 < y4 else ((x4, y4), (x3, y3))
            # 获取矩形的长和宽
            rect_w, rect_h = int(np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
                                 ), int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
            # 通过仿射变换得到矩阵
            src = np.float32([[x1, y1], [x2, y2], [x3, y3]])
            dst = np.float32([[0, rect_h], [0, 0], [rect_w, 0]])
            mat = cv2.getAffineTransform(src, dst)
            # 通过此函数，可以将文字区域修正为水平方向
            img_rot = cv2.warpAffine(img_binary, mat, (w, h))
            # 再次二值化，必须确保是黑底白字
            _, img_rot = cv2.threshold(
                img_rot, 0, 255, cv2.THRESH_OTSU)
            # 截取最小矩形区域
            img_roi = img_rot[0:rect_h, 0:rect_w]
        else:
            img_roi = img_binary

        return img_roi

    def deskew(self, img):
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*CHAR_SIZE*skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (CHAR_SIZE, CHAR_SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img

    def preprocess_hog(self, digits):
        samples = []
        for img in digits:
            # 计算两个方向的梯度值
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            # 计算幅度值和梯度的方向
            mag, ang = cv2.cartToPolar(gx, gy)
            # 将360度（2*PI）分割成16个bin
            bin_n = 16
            bin = np.int32(bin_n*ang/(2*np.pi))
            bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
            mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
            hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
            hist = np.hstack(hists)

            # transform to Hellinger kernel
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= norm(hist) + eps

            samples.append(hist)
        return np.float32(samples)
