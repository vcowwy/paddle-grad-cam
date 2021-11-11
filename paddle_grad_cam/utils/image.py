import cv2
import numpy as np
from PIL import Image

import paddle
import paddle.vision.transforms as T


def preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    preprocessing = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return paddle.to_tensor(preprocessing(img.copy())).unsqueeze(0)


def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-05)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def show_cam_on_image(img, mask, use_rgb: bool=False, colormap: int=cv2.COLORMAP_JET):

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception('The input image should np.float32 in the range [0, 1]')

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
