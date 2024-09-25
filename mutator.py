import numpy as np
import cv2

def image_translation(img, params):
    rows, cols, ch = img.shape
    M = np.float32([[1, 0, params], [0, 1, params]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_scale(img, params):
    rows, cols, ch = img.shape
    res = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
    res = res.reshape((res.shape[0],res.shape[1],ch))
    y, x, z = res.shape
    if params > 1:  # need to crop
        startx = x // 2 - cols // 2
        starty = y // 2 - rows // 2
        return res[starty:starty + rows, startx:startx + cols]
    elif params < 1:  # need to pad
        sty = int((rows - y) / 2)
        stx = int((cols - x) / 2)
#             print((sty, rows - y - sty), (stx, cols - x - stx),np.array([(sty, rows - y - sty), (stx, cols - x - stx), (0, 0)]).dtype.kind)
        return np.pad(res, [(sty, rows - y - sty), (stx, cols - x - stx), (0, 0)], mode='constant',
                        constant_values=0)
    return res

def image_shear(img, params):
    rows, cols, ch = img.shape
    factor = params * (-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def image_rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_AREA)
    return dst

def image_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha

    return new_img

def image_brightness(img, params):
    # Attention: Only works for 1 channel image.
    beta = params
    sec_matrix = np.zeros(img.shape)
    sec_matrix += beta
    if beta >= 0:
        mask = (img >= beta)
    else:
        mask = (img <= 255 + beta)
    new_img = img - beta * mask # new_img = img*alpha + beta
    return new_img

def image_blur(img, params):
    img = img.transpose([1, 2, 0])
    img = img.astype('uint8')
    # print("blur")
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    # if params == 9:
    #     blur = cv2.blur(img, (6, 6))
    if params == 9:
        blur = cv2.bilateralFilter(img, 6, 50, 50)
        # blur = cv2.bilateralFilter(img, 9, 75, 75)
    blur = np.expand_dims(blur, 0)
    #blur = blur.transpose([2, 0, 1])
    return blur

def image_pixel_change(img, params):
    # random change 1 - 5 pixels from 0 -255
    img_shape = img.shape
    img1d = np.ravel(img)
    arr = np.random.randint(0, len(img1d), params)
    for i in arr:
        img1d[i] = np.random.randint(0, 256)
    new_img = img1d.reshape(img_shape)
    return new_img

def image_noise(img, params):
    if params == 1:  # Gaussian-distributed additive noise.
        row, col, ch = img.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5

        gauss = np.random.normal(mean, sigma, (row, col, ch))*255
        mask = np.random.uniform(0,1,(row, col, ch))
        noisy = np.clip(img + mask*gauss,0,255)
        return noisy.astype(np.uint8)
    elif params == 2:  # Replaces random pixels with 0 or 1.
        s_vs_p = 0.5
        amount = 0.005
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt))
                    for i in img.shape]
        out[tuple(coords)] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper))
                    for i in img.shape]
        out[tuple(coords)] = 0
        return out
    elif params == 3:  # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
        row, col, ch = img.shape
        gauss = np.random.randn(row, col, ch)
        mask = np.random.uniform(0,1,(row, col, ch))
        gauss *= mask
        noisy = np.clip(img + img * gauss,0,255)
        return noisy.astype(np.uint8)
        
class Mutators():
    def __init__(self):
        self.transformations = [
            image_translation, image_scale, image_shear, image_rotation,
            image_contrast, image_brightness, image_blur, image_pixel_change, image_noise
            ]
        self.classA = [7, 8]  # pixel value transformation
        self.classB = [0, 1, 2, 3, 4, 5, 6] # Affine transformatio
        params = []
        params.append(list(range(-3, 3)))  # image_translation
        params.append(list(map(lambda x: x * 0.1, list(range(7, 12)))))  # image_scale
        params.append(list(map(lambda x: x * 0.1, list(range(-6, 6)))))  # image_shear
        params.append(list(range(-50, 50)))  # image_rotation
        params.append(list(map(lambda x: x * 0.1, list(range(5, 13)))))  # image_contrast
        params.append(list(range(-20, 20)))  # image_brightness
        params.append(list(range(1, 10)))  # image_blur
        params.append(list(range(1, 10)))  # image_pixel_change
        params.append(list(range(1, 4)))  # image_noise
        self.params = params