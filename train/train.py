#!/usr/bin/env python
# coding=utf8

import numpy as np
import cv2
from numpy.linalg import norm
import os

CHAR_SIZE = 20

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model = cv2.KNearest()
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()

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


def __scall_and_border_image(src_image, target_size):
    _h, _w = src_image.shape[:2]
    _ratio_target = float(target_size[0]) / float(target_size[1])
    _ratio_image = float(_w) / float(_h)
    _max = max(_ratio_image, _ratio_target)
    if _max == _ratio_image:
        __w = target_size[0]
        __h = int(target_size[0] / _ratio_image)
    else:
        __w = int(target_size[1] * _ratio_image)
        __h = target_size[1]
    _src_image = cv2.resize(src_image, (__w, __h), interpolation=cv2.INTER_LINEAR)

    _left = (target_size[0] - __w) / 2
    _right = target_size[0] - __w - _left
    _top = (target_size[1] - __h) / 2
    _bottom = target_size[1] - __h - _top
    _src_image = cv2.copyMakeBorder(_src_image, _top, _bottom, _left, _right, cv2.BORDER_CONSTANT, value = 0)

    return _src_image, (_left, _top, float(__w)/float(_w), float(__h)/float(_h))

def load_chars(work_path):
    images_path = os.listdir(work_path)
    digits, labels = [], []
    for index, _path in enumerate(images_path):
        if _path[_path.rfind('.'):] != '.jpg':
            continue
        ch_img = cv2.imread(work_path + _path)
        _h, _w = ch_img.shape[:2]
        ch_img, _ = __scall_and_border_image(ch_img, (CHAR_SIZE, CHAR_SIZE))
        ch_img = cv2.cvtColor(ch_img, cv2.COLOR_BGR2GRAY)
        digits.append(ch_img)
        labels.append(ord(_path.split('.')[-2].split('_')[-1]))
    return digits, labels

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*CHAR_SIZE*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (CHAR_SIZE, CHAR_SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
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

def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print 'error: %.2f %%' % (err*100)

if __name__ == '__main__':
    # 读取训练用的字符图片，这里以名字末尾字符作为标签，进行有监督学习
    digits, labels = load_chars('%s/char_good/'%os.getcwd())
    print 'preprocessing...digits=%d,labels=%d'%(len(digits),len(labels))
    # shuffle digits 打乱所有字符
    # rand = np.random.RandomState(321)
    # shuffle = rand.permutation(len(digits))
    # digits, labels = digits[shuffle], labels[shuffle]

    digits2 = map(deskew, digits)
    samples = preprocess_hog(digits2)

    train_n = int(0.9*len(samples))
    digits_train, digits_test = np.split(digits2, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    print 'training KNearest...'
    model = KNearest(k=4)
    model.train(samples_train, labels_train)
    evaluate_model(model, digits_test, samples_test, labels_test)
    # model.save('digits_knearest.dat')

    print 'training SVM...'
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    evaluate_model(model, digits_test, samples_test, labels_test)
    print 'saving SVM as "digits_svm.dat"...'
    model.save('digits_svm.dat')
