import time, os, math
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from torchsummary import summary

from craft_pytorch.craft import CRAFT
from craft_pytorch import imgproc
from craft_pytorch import file_utils
from craft_pytorch import craft_utils

from collections import OrderedDict

CRAFT_MODEL_DIR = "./craft_pytorch/craft_mlt_25k.pth"
CRAFT_REFINER_MODEL_DIR = "./craft_pytorch/craft_refiner_CTW1500.pth"


class craft_net_with_refiner():
    def __init__(self, trained_model=CRAFT_MODEL_DIR, text_threshold=0.7, low_text=0.4,
                 link_threshold=0.4, cuda=True, canvas_size=1280, mag_ratio=1.5, poly=False,
                 show_time=False, refine=True, refiner_model=CRAFT_REFINER_MODEL_DIR):
        """
        :param image: image to predict
        :param trained_model: pretrained model location
        :param text_threshold: text confidence threshold
        :param low_text: text low-bound score
        :param link_threshold: link confidencee threshold
        :param cuda: use cuda for inference
        :param canvas_size: image size for inference
        :param mag_ratio: image magnification ratio
        :param poly: enable polygon type
        :param show_time: show process time
        :param refine: enable link refiner
        :param refiner_model: pretrained refiner model location
        """
        self.trained_model = trained_model
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.cuda = cuda
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.poly = poly
        self.show_time = show_time
        self.refine = refine
        self.refiner_model = refiner_model

        self.net = CRAFT()
        print('Loading weights from checkpoint (' + self.trained_model + ')')
        if self.cuda:
            self.net.load_state_dict(self.copy_statedict(torch.load(self.trained_model)))
        else:
            self.net.load_state_dict(self.copy_statedict(torch.load(self.trained_model, map_location='cpu')))

        if self.cuda:
            self.net = self.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if self.refine:
            from craft_pytorch.refinenet import RefineNet
            self.refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.refiner_model + ')')
            if self.cuda:
                self.refine_net.load_state_dict(self.copy_statedict(torch.load(self.refiner_model)))
                self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(self.copy_statedict(torch.load(self.refiner_model, map_location='cpu')))

            self.refine_net.eval()
            self.poly = True

        self.t = time.time()

    def copy_statedict(self, state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict


    def craft_image(self, image):
        # summary(self.net, (image.shape[2], image.shape[0], image.shape[1]))
        self.t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size,
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()


        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        self.t0 = time.time() - self.t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold,
                                               self.low_text, self.poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if self.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(self.t0, t1))

        return boxes, polys, ret_score_text


def get_result_of_location(boxes, image):
    """
    :param boxes: boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
    :return:
        output: [[image, location]]
    """
    output = [] # [[image, location]]
    image = np.array(image)
    for i, box in enumerate(boxes):
        # print("Text {}".format(i))

        polys = np.array(box).astype(np.int32).reshape((-1))
        polys = polys.reshape(-1, 2) # a group of [[x, y]]
        w, w_list, h, s_len = get_rectificated_size(polys)

        pts_tar = []
        pts_tar.append([0, 0])
        for j in range(0, (s_len // 2) - 1):
            pts_tar.append([pts_tar[j][0] + int(w_list[j]), 0])
        pts_tar.append([int(w), int(h)])
        for j in range(s_len // 2, s_len - 1):
            pts_tar.append([pts_tar[j][0] - int(w_list[j - 1]), int(h)])

        pts_tar[s_len - 1][0] = 0
        dst = np.zeros((int(h), int(w), 3), np.uint8)
        for j in range(0, (s_len // 2) - 1):
            pts1 = []
            pts2 = []
            # print("{}, {}".format(i, j))
            pts1.append(polys[j])
            pts1.append(polys[j + 1])
            pts1.append(polys[s_len - j - 2])
            pts1.append(polys[s_len - j - 1])
            pts2.append(pts_tar[j])
            pts2.append(pts_tar[j + 1])
            pts2.append(pts_tar[s_len - j - 2])
            pts2.append(pts_tar[s_len - j - 1])
            # for k in range(4):
            #     print(k, pts1[k], pts2[k])

            # M, _ = cv2.findHomography(np.float32(pts1), np.float32(pts2), 0)
            M = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
            wrap = cv2.warpPerspective(image, M, (int(w), int(h)))
            dst = cv2.bitwise_or(dst, crop(wrap, pts2))

        output.append([dst, polys])
        # plt.subplot(221), plt.imshow(image), plt.title('input')
        # plt.subplot(222), plt.imshow(dst), plt.title('output')
    return output



def crop(img, pts):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    roi_corner = np.int32(pts).reshape(-1, 2)
    cv2.drawContours(mask, [roi_corner], -1, (255, 255, 255), -1, cv2.LINE_AA)
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    return masked_image


def rectificat(image):
    pass


def get_rectificated_size(polys):
    s_len = len(polys)
    h1 = get_distance(polys[0], polys[-1])
    h2 = get_distance(polys[(s_len // 2) - 1], polys[s_len // 2])
    height = h1 if h1 >= h2 else h2
    width_u = 0
    width_d = 0
    width_list_u = []
    width_list_d = []

    for i in range(0, (s_len // 2) - 1):
        width_u += get_distance(polys[i], polys[i + 1])
        width_list_u.append(get_distance(polys[i], polys[i + 1]))
    for i in range((s_len // 2), s_len - 1):
        width_d += get_distance(polys[i], polys[i + 1])
        width_list_d.append(get_distance(polys[i], polys[i + 1]))

    if width_u >= width_d:
        # down -> up
        width = width_u
        for i in range((s_len // 2) - 1):
            width_list_d[i] = (width_u * width_list_d[i]) / width_d
    else:
        # up -> down
        width = width_d
        for i in range((s_len // 2) - 1):
            width_list_u[i] = (width_d * width_list_u[i]) / width_u
    return width, width_list_u + width_list_d, height, s_len


def get_distance(group1, group2):
    return math.sqrt((group1[0] - group2[0]) ** 2 + (group1[1] - group2[1]) ** 2)


def sort_box(box):
    """
    对box进行排序
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box


def detect(image_path, adjust=False):
    craft_net = craft_net_with_refiner(cuda=False, low_text=0.285, link_threshold=0.85, text_threshold=0.5)
    image = imgproc.loadImage(image_path)
    bboxes, polys, score_text = craft_net.craft_image(image)
    rectification = get_result_of_location(polys, image[:, :, ::-1])
    # .array(Image.open(image_path).convert('RGB')), polys, adjust
    return rectification


def detect_skimage(img, adjust=False):
    craft_net = craft_net_with_refiner(cuda=False, low_text=0.285, link_threshold=0.85, text_threshold=0.5)
    image = imgproc.loadImage_img(img)
    bboxes, polys, score_text = craft_net.craft_image(image)
    rectification = get_result_of_location(polys, image[:, :, ::-1])
    return rectification




if __name__ == "__main__":
    # detect("./craft_pytorc1507.05717h/test/star.jpg")
    craft_net = craft_net_with_refiner(cuda=False, low_text=0.285, link_threshold=0.85, text_threshold=0.5)
    image_path = "./craft_pytorch/test/star.jpg"
    image = imgproc.loadImage(image_path)

    bboxes, polys, score_text = craft_net.craft_image(image)
    get_result_of_location(polys, image[:, :, ::-1])

    # save score text
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    # mask_file = "./res_" + filename + '_mask.jpg'
    # cv2.imwrite(mask_file, score_text)

    post_img = file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname="./")
    plt.subplot(223), plt.imshow(score_text), plt.title('affine')
    plt.subplot(224), plt.imshow(post_img), plt.title('region')
    plt.show()
