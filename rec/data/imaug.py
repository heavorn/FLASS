import six
import math
import cv2
import numpy as np
import numbers
import random
import copy
from PIL import Image
from paddle.vision.transforms import Compose, ColorJitter
from rec.utils.logging import get_logger



def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data

def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops

class DecodeImage(object):
    """ decode image """

    def __init__(self,
                 img_mode='RGB',
                 channel_first=False,
                 ignore_orientation=False,
                 **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation

    def __call__(self, data):
        img = data['image']
        if six.PY2:
            assert type(img) is str and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype='uint8')
        if self.ignore_orientation:
            img = cv2.imdecode(img, cv2.IMREAD_IGNORE_ORIENTATION |
                               cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list
    

class BaseRecLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 lower=False):

        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower

        if character_dict_path is None:
            logger = get_logger()
            logger.warning(
                "The character_dict_path is None, model can only recognize number and lower letters"
            )
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data['label_ace'] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class SVTRRecAug(object):
    def __init__(self,
                 aug_type=0,
                 geometry_p=0.5,
                 deterioration_p=0.25,
                 colorjitter_p=0.25,
                 **kwargs):
        self.transforms = Compose([
            SVTRGeometry(
                aug_type=aug_type,
                degrees=45,
                translate=(0.0, 0.0),
                scale=(0.5, 2.),
                shear=(45, 15),
                distortion=0.5,
                p=geometry_p), SVTRDeterioration(
                    var=20, degrees=6, factor=4, p=deterioration_p),
            CVColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
                p=colorjitter_p)
        ])

    def __call__(self, data):
        img = data['image']
        img = self.transforms(img)
        data['image'] = img
        return data



class SVTRRecResizeImg(object):
    def __init__(self, image_shape, padding=True, **kwargs):
        self.image_shape = image_shape
        self.padding = padding

    def __call__(self, data):
        img = data['image']

        norm_img, valid_ratio = resize_norm_img(img, self.image_shape,
                                                self.padding)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data



def resize_norm_img(img,
                    image_shape,
                    padding=True,
                    interpolation=cv2.INTER_LINEAR):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    if not padding:
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=interpolation)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio



def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def hsv_aug(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


def jitter(img):
    """
    jitter
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=0.1):
    """
    Gasuss noise
    """

    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def get_crop(image):
    """
    random crop
    """
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img


def rad(x):
    """
    rad
    """
    return x * np.pi / 180


def get_warpR(config):
    """
    get_warpR
    """
    anglex, angley, anglez, fov, w, h, r = \
        config.anglex, config.angley, config.anglez, config.fov, config.w, config.h, config.r
    if w > 69 and w < 112:
        anglex = anglex * 1.5

    z = np.sqrt(w**2 + h**2) / 2 / np.tan(rad(fov / 2))
    # Homogeneous coordinate transformation matrix
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0], [
                       0,
                       -np.sin(rad(anglex)),
                       np.cos(rad(anglex)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0], [
                       -np.sin(rad(angley)),
                       0,
                       np.cos(rad(angley)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    r = rx.dot(ry).dot(rz)
    # generate 4 points
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
    list_dst = np.array([dst1, dst2, dst3, dst4])
    org = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
    dst = np.zeros((4, 2), np.float32)
    # Project onto the image plane
    dst[:, 0] = list_dst[:, 0] * z / (z - list_dst[:, 2]) + pcenter[0]
    dst[:, 1] = list_dst[:, 1] * z / (z - list_dst[:, 2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)

    dst1, dst2, dst3, dst4 = dst
    r1 = int(min(dst1[1], dst2[1]))
    r2 = int(max(dst3[1], dst4[1]))
    c1 = int(min(dst1[0], dst3[0]))
    c2 = int(max(dst2[0], dst4[0]))

    try:
        ratio = min(1.0 * h / (r2 - r1), 1.0 * w / (c2 - c1))

        dx = -c1
        dy = -r1
        T1 = np.float32([[1., 0, dx], [0, 1., dy], [0, 0, 1.0 / ratio]])
        ret = T1.dot(warpR)
    except:
        ratio = 1.0
        T1 = np.float32([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        ret = T1
    return ret, (-r1, -c1), ratio, dst


def get_warpAffine(config):
    """
    get_warpAffine
    """
    anglez = config.anglez
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0]], np.float32)
    return rz


def sample_asym(magnitude, size=None):
    return np.random.beta(1, 4, size) * magnitude


def sample_sym(magnitude, size=None):
    return (np.random.beta(4, 4, size=size) - 0.5) * 2 * magnitude


def sample_uniform(low, high, size=None):
    return np.random.uniform(low, high, size=size)


def get_interpolation(type='random'):
    if type == 'random':
        choice = [
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA
        ]
        interpolation = choice[random.randint(0, len(choice) - 1)]
    elif type == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif type == 'linear':
        interpolation = cv2.INTER_LINEAR
    elif type == 'cubic':
        interpolation = cv2.INTER_CUBIC
    elif type == 'area':
        interpolation = cv2.INTER_AREA
    else:
        raise TypeError(
            'Interpolation types only nearest, linear, cubic, area are supported!'
        )
    return interpolation


class CVRandomRotation(object):
    def __init__(self, degrees=15):
        assert isinstance(degrees,
                          numbers.Number), "degree should be a single number."
        assert degrees >= 0, "degree must be positive."
        self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        return sample_sym(degrees)

    def __call__(self, img):
        angle = self.get_params(self.degrees)
        src_h, src_w = img.shape[:2]
        M = cv2.getRotationMatrix2D(
            center=(src_w / 2, src_h / 2), angle=angle, scale=1.0)
        abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
        dst_w = int(src_h * abs_sin + src_w * abs_cos)
        dst_h = int(src_h * abs_cos + src_w * abs_sin)
        M[0, 2] += (dst_w - src_w) / 2
        M[1, 2] += (dst_h - src_h) / 2

        flags = get_interpolation()
        return cv2.warpAffine(
            img,
            M, (dst_w, dst_h),
            flags=flags,
            borderMode=cv2.BORDER_REPLICATE)


class CVRandomAffine(object):
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        assert isinstance(degrees,
                          numbers.Number), "degree should be a single number."
        assert degrees >= 0, "degree must be positive."
        self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError(
                        "translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError(
                        "If shear is a single number, it must be positive.")
                self.shear = [shear]
            else:
                assert isinstance(shear, (tuple, list)) and (len(shear) == 2), \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

    def _get_inverse_affine_matrix(self, center, angle, translate, scale,
                                   shear):
        # https://github.com/pytorch/vision/blob/v0.4.0/torchvision/transforms/functional.py#L717
        from numpy import sin, cos, tan

        if isinstance(shear, numbers.Number):
            shear = [shear, 0]

        if not isinstance(shear, (tuple, list)) and len(shear) == 2:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))

        rot = math.radians(angle)
        sx, sy = [math.radians(s) for s in shear]

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = cos(rot - sy) / cos(sy)
        b = -cos(rot - sy) * tan(sx) / cos(sy) - sin(rot)
        c = sin(rot - sy) / cos(sy)
        d = -sin(rot - sy) * tan(sx) / cos(sy) + cos(rot)

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        M = [d, -b, 0, -c, a, 0]
        M = [x / scale for x in M]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
        M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        M[2] += cx
        M[5] += cy
        return M

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, height):
        angle = sample_sym(degrees)
        if translate is not None:
            max_dx = translate[0] * height
            max_dy = translate[1] * height
            translations = (np.round(sample_sym(max_dx)),
                            np.round(sample_sym(max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = sample_uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 1:
                shear = [sample_sym(shears[0]), 0.]
            elif len(shears) == 2:
                shear = [sample_sym(shears[0]), sample_sym(shears[1])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        src_h, src_w = img.shape[:2]
        angle, translate, scale, shear = self.get_params(
            self.degrees, self.translate, self.scale, self.shear, src_h)

        M = self._get_inverse_affine_matrix((src_w / 2, src_h / 2), angle,
                                            (0, 0), scale, shear)
        M = np.array(M).reshape(2, 3)

        startpoints = [(0, 0), (src_w - 1, 0), (src_w - 1, src_h - 1),
                       (0, src_h - 1)]
        project = lambda x, y, a, b, c: int(a * x + b * y + c)
        endpoints = [(project(x, y, *M[0]), project(x, y, *M[1]))
                     for x, y in startpoints]

        rect = cv2.minAreaRect(np.array(endpoints))
        bbox = cv2.boxPoints(rect).astype(dtype=np.int32)
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()

        dst_w = int(max_x - min_x)
        dst_h = int(max_y - min_y)
        M[0, 2] += (dst_w - src_w) / 2
        M[1, 2] += (dst_h - src_h) / 2

        # add translate
        dst_w += int(abs(translate[0]))
        dst_h += int(abs(translate[1]))
        if translate[0] < 0: M[0, 2] += abs(translate[0])
        if translate[1] < 0: M[1, 2] += abs(translate[1])

        flags = get_interpolation()
        return cv2.warpAffine(
            img,
            M, (dst_w, dst_h),
            flags=flags,
            borderMode=cv2.BORDER_REPLICATE)


class CVRandomPerspective(object):
    def __init__(self, distortion=0.5):
        self.distortion = distortion

    def get_params(self, width, height, distortion):
        offset_h = sample_asym(
            distortion * height / 2, size=4).astype(dtype=np.int32)
        offset_w = sample_asym(
            distortion * width / 2, size=4).astype(dtype=np.int32)
        topleft = (offset_w[0], offset_h[0])
        topright = (width - 1 - offset_w[1], offset_h[1])
        botright = (width - 1 - offset_w[2], height - 1 - offset_h[2])
        botleft = (offset_w[3], height - 1 - offset_h[3])

        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1),
                       (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return np.array(
            startpoints, dtype=np.float32), np.array(
                endpoints, dtype=np.float32)

    def __call__(self, img):
        height, width = img.shape[:2]
        startpoints, endpoints = self.get_params(width, height, self.distortion)
        M = cv2.getPerspectiveTransform(startpoints, endpoints)

        # TODO: more robust way to crop image
        rect = cv2.minAreaRect(endpoints)
        bbox = cv2.boxPoints(rect).astype(dtype=np.int32)
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()
        min_x, min_y = max(min_x, 0), max(min_y, 0)

        flags = get_interpolation()
        img = cv2.warpPerspective(
            img,
            M, (max_x, max_y),
            flags=flags,
            borderMode=cv2.BORDER_REPLICATE)
        img = img[min_y:, min_x:]
        return img


class CVRescale(object):
    def __init__(self, factor=4, base_size=(128, 512)):
        """ Define image scales using gaussian pyramid and rescale image to target scale.
        
        Args:
            factor: the decayed factor from base size, factor=4 keeps target scale by default.
            base_size: base size the build the bottom layer of pyramid
        """
        if isinstance(factor, numbers.Number):
            self.factor = round(sample_uniform(0, factor))
        elif isinstance(factor, (tuple, list)) and len(factor) == 2:
            self.factor = round(sample_uniform(factor[0], factor[1]))
        else:
            raise Exception('factor must be number or list with length 2')
        # assert factor is valid
        self.base_h, self.base_w = base_size[:2]

    def __call__(self, img):
        if self.factor == 0: return img
        src_h, src_w = img.shape[:2]
        cur_w, cur_h = self.base_w, self.base_h
        scale_img = cv2.resize(
            img, (cur_w, cur_h), interpolation=get_interpolation())
        for _ in range(self.factor):
            scale_img = cv2.pyrDown(scale_img)
        scale_img = cv2.resize(
            scale_img, (src_w, src_h), interpolation=get_interpolation())
        return scale_img


class CVGaussianNoise(object):
    def __init__(self, mean=0, var=20):
        self.mean = mean
        if isinstance(var, numbers.Number):
            self.var = max(int(sample_asym(var)), 1)
        elif isinstance(var, (tuple, list)) and len(var) == 2:
            self.var = int(sample_uniform(var[0], var[1]))
        else:
            raise Exception('degree must be number or list with length 2')

    def __call__(self, img):
        noise = np.random.normal(self.mean, self.var**0.5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img


class CVMotionBlur(object):
    def __init__(self, degrees=12, angle=90):
        if isinstance(degrees, numbers.Number):
            self.degree = max(int(sample_asym(degrees)), 1)
        elif isinstance(degrees, (tuple, list)) and len(degrees) == 2:
            self.degree = int(sample_uniform(degrees[0], degrees[1]))
        else:
            raise Exception('degree must be number or list with length 2')
        self.angle = sample_uniform(-angle, angle)

    def __call__(self, img):
        M = cv2.getRotationMatrix2D((self.degree // 2, self.degree // 2),
                                    self.angle, 1)
        motion_blur_kernel = np.zeros((self.degree, self.degree))
        motion_blur_kernel[self.degree // 2, :] = 1
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M,
                                            (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        img = cv2.filter2D(img, -1, motion_blur_kernel)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class CVGeometry(object):
    def __init__(self,
                 degrees=15,
                 translate=(0.3, 0.3),
                 scale=(0.5, 2.),
                 shear=(45, 15),
                 distortion=0.5,
                 p=0.5):
        self.p = p
        type_p = random.random()
        if type_p < 0.33:
            self.transforms = CVRandomRotation(degrees=degrees)
        elif type_p < 0.66:
            self.transforms = CVRandomAffine(
                degrees=degrees, translate=translate, scale=scale, shear=shear)
        else:
            self.transforms = CVRandomPerspective(distortion=distortion)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transforms(img)
        else:
            return img


class CVDeterioration(object):
    def __init__(self, var, degrees, factor, p=0.5):
        self.p = p
        transforms = []
        if var is not None:
            transforms.append(CVGaussianNoise(var=var))
        if degrees is not None:
            transforms.append(CVMotionBlur(degrees=degrees))
        if factor is not None:
            transforms.append(CVRescale(factor=factor))

        random.shuffle(transforms)
        transforms = Compose(transforms)
        self.transforms = transforms

    def __call__(self, img):
        if random.random() < self.p:

            return self.transforms(img)
        else:
            return img


class CVColorJitter(object):
    def __init__(self,
                 brightness=0.5,
                 contrast=0.5,
                 saturation=0.5,
                 hue=0.1,
                 p=0.5):
        self.p = p
        self.transforms = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)

    def __call__(self, img):
        if random.random() < self.p: return self.transforms(img)
        else: return img


class SVTRDeterioration(object):
    def __init__(self, var, degrees, factor, p=0.5):
        self.p = p
        transforms = []
        if var is not None:
            transforms.append(CVGaussianNoise(var=var))
        if degrees is not None:
            transforms.append(CVMotionBlur(degrees=degrees))
        if factor is not None:
            transforms.append(CVRescale(factor=factor))
        self.transforms = transforms

    def __call__(self, img):
        if random.random() < self.p:
            random.shuffle(self.transforms)
            transforms = Compose(self.transforms)
            return transforms(img)
        else:
            return img


class SVTRGeometry(object):
    def __init__(self,
                 aug_type=0,
                 degrees=15,
                 translate=(0.3, 0.3),
                 scale=(0.5, 2.),
                 shear=(45, 15),
                 distortion=0.5,
                 p=0.5):
        self.aug_type = aug_type
        self.p = p
        self.transforms = []
        self.transforms.append(CVRandomRotation(degrees=degrees))
        self.transforms.append(
            CVRandomAffine(
                degrees=degrees, translate=translate, scale=scale, shear=shear))
        self.transforms.append(CVRandomPerspective(distortion=distortion))

    def __call__(self, img):
        if random.random() < self.p:
            if self.aug_type:
                random.shuffle(self.transforms)
                transforms = Compose(self.transforms[:random.randint(1, 3)])
                img = transforms(img)
            else:
                img = self.transforms[random.randint(0, 2)](img)
            return img
        else:
            return img
        


