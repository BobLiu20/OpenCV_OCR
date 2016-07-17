#decoding:utf-8
import numpy as np
import cv2
import os

CHAR_SIZE = 20

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

def load_base(work_path):
    images_path = os.listdir(work_path)
    digits, labels = [], []
    for index, _path in enumerate(images_path):
        if _path[_path.rfind('.'):] != '.jpg':
            continue
        ch_img = cv2.imread(work_path + '/' + _path)
        _h, _w = ch_img.shape[:2]
        ch_img, _ = __scall_and_border_image(ch_img, (CHAR_SIZE, CHAR_SIZE))
        ch_img = cv2.cvtColor(ch_img, cv2.COLOR_BGR2GRAY)
        digits.append(ch_img)
        labels.append(ord(_path.split('.')[-2].split('_')[-1]))
    return digits, labels

class LetterStatModel(object):
    class_n = 26
    train_ratio = 0.5

    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)
    
    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape#获取特征维数和特征个数
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples
    
    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*self.class_n, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        new_responses[resp_idx] = 1
        return new_responses

class RTrees(LetterStatModel):
    def __init__(self):
        self.model = cv2.RTrees()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL], np.uint8)
        #CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));
        params = dict(max_depth=10 )
        self.model.train(samples, cv2.CV_ROW_SAMPLE, responses, varType = var_types, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )
        

class KNearest(LetterStatModel):
    def __init__(self):
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, k = 10)
        return results.ravel()


class Boost(LetterStatModel):
    def __init__(self):
        self.model = cv2.Boost()
    
    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL, cv2.CV_VAR_CATEGORICAL], np.uint8)
        #CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 )
        params = dict(max_depth=5) #, use_surrogates=False)
        self.model.train(new_samples, cv2.CV_ROW_SAMPLE, new_responses, varType = var_types, params=params)

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        pred = np.array( [self.model.predict(s, returnSum = True) for s in new_samples] )
        pred = pred.reshape(-1, self.class_n).argmax(1)
        return pred


class SVM(LetterStatModel):
    train_ratio = 0.1
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        params = dict( kernel_type = cv2.SVM_LINEAR, 
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )


class MLP(LetterStatModel):
    def __init__(self):
        self.model = cv2.ANN_MLP()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.class_n)

        layer_sizes = np.int32([var_n, 100, 100, self.class_n])
        self.model.create(layer_sizes)
        
        # CvANN_MLP_TrainParams::BACKPROP,0.001
        params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 300, 0.01),
                       train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, 
                       bp_dw_scale = 0.001,
                       bp_moment_scale = 0.0 )
        self.model.train(samples, np.float32(new_responses), None, params = params)

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)


if __name__ == '__main__':
    import getopt
    import sys

    models = [RTrees, KNearest, Boost, SVM, MLP] # NBayes
    models = dict( [(cls.__name__.lower(), cls) for cls in models] )#将名字之母字母转为小写

    print 'USAGE: letter_recog.py [--model <model>] [--data <data fn>] [--load <model fn>] [--save <model fn>]'
    print 'Models: ', ', '.join(models)
    print

    args, dummy = getopt.getopt(sys.argv[1:], '', ['model=', 'data=', 'load=', 'save='])
    args = dict(args)
    args.setdefault('--model', 'boost')
    args.setdefault('--data', 'char_good')

    print 'loading data %s ...' % args['--data']
    samples, responses = load_base(args['--data'])
    Model = models[args['--model']]
    model = Model()

    train_n = int(len(samples)*model.train_ratio)#获取训练数据的数目
    if '--load' in args:
        fn = args['--load']
        print 'loading model from %s ...' % fn
        model.load(fn)
    else:
        print 'training %s ...' % Model.__name__
        model.train(samples[:train_n], responses[:train_n])

    print 'testing...'
    train_rate = np.mean(model.predict(samples[:train_n]) == responses[:train_n])#前一半进行训练，并得到训练准确率
    test_rate  = np.mean(model.predict(samples[train_n:]) == responses[train_n:])#后一半进行测试，并得到测试准确率

    print 'train rate: %f  test rate: %f' % (train_rate*100, test_rate*100)

    if '--save' in args:
        fn = args['--save']
        print 'saving model to %s ...' % fn
        model.save(fn)
    cv2.destroyAllWindows()             
