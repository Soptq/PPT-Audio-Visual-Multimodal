from detection import detect_skimage
from PIL import Image
from torchvision import transforms

from crnn.crnn import crnnOcr
import numpy as np

import segmentation
import cv2
from skimage import io



def image_process(image, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
  img = Image.fromarray(image).convert('RGB')

  if keep_ratio:
    w, h = img.size
    ratio = w / float(h)
    imgW = int(np.floor(ratio * imgH))
    imgW = max(imgH * min_ratio, imgW)

  img = img.resize((imgW, imgH), Image.BILINEAR)
  img = transforms.ToTensor()(img)
  img.sub_(0.5).div_(0.5)

  return img


def ocr_opencv(img):
    """传入一个opencv格式的图片 进行ocr"""
    image = img[:, :, ::-1]
    return ocr_core(image)


def ocr_pillow(img):
    """传入一个pillow格式的图片 进行ocr"""
    image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    image = image[:, :, ::-1]
    return ocr_core(image)


def ocr_skimage(img):
    """传入一个skimage格式的图片 进行ocr"""
    return ocr_core(img)


def ocr_path(path):
    """传入一个图像地址 进行ocr"""
    image = io.imread(path)
    return ocr_core(image)


def ocr_core(image_src):
    """
    调用 ocr 识别图像
    :param image_path: skimage 处理过的图像
    :return: [[识别字符, bounding box], ...]
    """
    rectification = detect_skimage(image_src)
    result = []
    index = 0
    for ele in rectification:
        index += 1
        img = ele[0]

        minlength = img.shape[0] if img.shape[0] < img.shape[1] else img.shape[1]
        if minlength < 1280:
            factor = 1280 // minlength + 1
            img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if th[0][0] == 0:
            th = cv2.bitwise_not(th)
        img = Image.fromarray(th).convert('L')
        res = crnnOcr(img)
        res = segmentation.correct(res).strip()
        result.append([res, ele[1]])
    return result



if __name__ == "__main__":
    result = ocr_path("./test/sun#2.png")
    print(result)