import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import math
import mmcv

msra_root_dir = './data/MSRA-TD500/'
msra_train_data_dir = msra_root_dir + 'train/'
msra_train_gt_dir = msra_root_dir + 'train/'
msra_test_data_dir = msra_root_dir + 'test/'
msra_test_gt_dir = msra_root_dir + 'test/'

hust_root_dir = './data/HUST-TR400/'
hust_train_data_dir = hust_root_dir
hust_train_gt_dir = hust_root_dir


def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception as e:
        print(img_path)
        raise
    return img


def get_ann(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')

        gt = line.split(' ')

        w_ = np.float(gt[4])
        h_ = np.float(gt[5])
        x1 = np.float(gt[2]) + w_ / 2.0
        y1 = np.float(gt[3]) + h_ / 2.0
        theta = np.float(gt[6]) / math.pi * 180

        bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
        bbox = bbox.reshape(-1) / ([w * 1.0, h * 1.0] * 4)

        bboxes.append(bbox)
        words.append('???')
    return np.array(bboxes), words


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def scale_aligned_short(img, short_size=736):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=736):
    h, w = img.shape[0:2]

    scale = np.random.choice(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
    scale = (scale * short_size) / min(h, w)

    aspect = np.random.choice(np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
    h_scale = scale * math.sqrt(aspect)
    w_scale = scale / math.sqrt(aspect)

    img = scale_aligned(img, h_scale, w_scale)
    return img


def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


def jaccard(As, Bs):

    A = As.shape[0]
    B = Bs.shape[0]

    dis = np.sqrt(np.sum((As[:, np.newaxis, :].repeat(B, axis=1)
                        - Bs[np.newaxis, :, :].repeat(A, axis=0)) ** 2, axis=-1))

    ind = np.argmin(dis, axis=-1)

    return ind


class CT_MSRA(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=736,
                 kernel_scale=0.5,
                 read_type='pil',
                 report_speed=False):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.read_type = read_type

        if split == 'train':
            data_dirs = [msra_train_data_dir, hust_train_data_dir]
            gt_dirs = [msra_train_gt_dir, hust_train_gt_dir]
        elif split == 'test':
            data_dirs = [msra_test_data_dir]
            gt_dirs = [msra_test_gt_dir]
        else:
            raise ValueError('Error: split must be train or test!')

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = [img_name for img_name in mmcv.utils.scandir(data_dir) if img_name.endswith('.JPG')]
            img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir) if img_name.endswith('.jpg')])

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = img_name.split('.')[0] + '.gt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

        if report_speed:
            target_size = 3000
            data_size = len(self.img_paths)
            extend_scale = (target_size + data_size - 1) // data_size
            self.img_paths = (self.img_paths * extend_scale)[:target_size]
            self.gt_paths = (self.gt_paths * extend_scale)[:target_size]

    def __len__(self):
        return len(self.img_paths)

    def vis(self, name, img, gt_instance, training_mask, gt_kernel, gt_kernel_instance, gt_kernel_inner, gt_distance):

        img = img[:, :, [2, 1, 0]].copy()
        cv2.imwrite(name + '_img.jpg', img)
        max_instance = np.max(gt_instance)

        color_list = [(0, 0, 0)]
        for _ in range(50):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color_list.append((r, g, b))

        vis_instance = np.zeros_like(img)
        for i in range(max_instance + 1):
            pixels = np.where(gt_instance == i)
            vis_instance[pixels] = color_list[i]
        vis_instance = cv2.addWeighted(vis_instance, 0.7, img, 0.3, 0)
        cv2.imwrite(name + '_instance.jpg', vis_instance)

        training_mask = training_mask[:, :, np.newaxis].repeat(3, axis=-1) * 255
        training_mask[:, :, -1] = 0
        vis_instance = cv2.addWeighted(training_mask, 0.7, img, 0.3, 0)
        cv2.imwrite(name + '_trainingMask.jpg', vis_instance)

        cv2.imwrite(name + '_kernel.jpg', gt_kernel * 255)

        vis_instance = np.zeros_like(img)
        for i in range(max_instance + 1):
            pixels = np.where(gt_kernel_instance == i)
            vis_instance[pixels] = color_list[i]
        vis_instance = cv2.addWeighted(vis_instance, 0.7, img, 0.3, 0)
        cv2.imwrite(name + '_kernelInstance.jpg', vis_instance)

        vis_instance = np.zeros_like(img)
        for i in range(max_instance + 1):
            pixels = np.where(gt_kernel_inner == i)
            vis_instance[pixels] = color_list[i]
        vis_instance = cv2.addWeighted(vis_instance, 0.9, img, 0.1, 0)
        cv2.imwrite(name + '_kernelInner.jpg', vis_instance)

        # gt_distance
        vis_instance = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                tj = int(j + gt_distance[0][i][j] * 10)
                ti = int(i + gt_distance[1][i][j] * 10)
                vis_instance[i][j] = color_list[gt_kernel_instance[ti][tj]]
        vis_instance = cv2.addWeighted(vis_instance, 0.7, img, 0.3, 0)
        cv2.imwrite(name + '_distance.jpg', vis_instance)

    def prepare_train_data(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann(img, gt_path)

        if self.is_transform:
            img = random_scale(img, self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        training_mask_distance = np.ones(img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4),
                                (bboxes.shape[0], -1, 2)).astype('int32')

            bbox_areas = [-plg.Polygon(box.reshape(-1, 2)).area() for box in bboxes]
            bbox_inds = np.argsort(bbox_areas)
            tmp_bboxes = []
            for ind in bbox_inds:
                tmp_bboxes.append(bboxes[ind])
            bboxes = tmp_bboxes

            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask_distance, [bboxes[i]], -1, 0, -1)

        gt_kernel_instance = np.zeros(img.shape[0:2], dtype='uint8')
        kernel_bboxes = shrink(bboxes, self.kernel_scale)
        for i in range(len(bboxes)):
            cv2.drawContours(gt_kernel_instance, [kernel_bboxes[i]], -1, i + 1, -1)
            if words[i] != '###':
                cv2.drawContours(training_mask, [kernel_bboxes[i]], -1, 1, -1)
        gt_kernel = gt_kernel_instance.copy()
        gt_kernel[gt_kernel > 0] = 1

        max_instance = np.max(gt_kernel_instance)
        gt_kernel_inner = np.zeros_like(gt_kernel_instance)
        erode_kernel = np.ones((3, 3), np.uint8)
        for i in range(1, max_instance + 1):
            if np.sum(gt_kernel_instance == i) == 0:
                continue
            tmp1 = np.zeros(gt_kernel_instance.shape, dtype='uint8')
            tmp1[gt_kernel_instance == i] = 1
            tmp1 = cv2.erode(tmp1, erode_kernel, iterations=1)
            tmp2 = tmp1.copy()
            tmp2 = cv2.erode(tmp2, erode_kernel, iterations=1)
            gt_kernel_inner[(tmp1 - tmp2) == 1] = i

        if self.is_transform:
            imgs = [img, gt_instance, training_mask, gt_kernel_instance, gt_kernel, gt_kernel_inner, training_mask_distance]

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernel_instance, gt_kernel, gt_kernel_inner, training_mask_distance = \
                imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], imgs[5], imgs[6]

        max_instance = np.max(gt_instance)
        gt_distance = np.zeros((2, *img.shape[0:2]), dtype=np.float32)
        for i in range(1, max_instance + 1):
            ind = (gt_kernel_inner == i)
            if np.sum(ind) == 0:
                training_mask[gt_instance == i] = 0
                training_mask_distance[gt_instance == i] = 0
                continue
            kpoints = np.array(np.where(ind)).transpose((1, 0))[:, ::-1].astype('float32')

            ind = (gt_instance == i) * (gt_kernel_instance != i)
            if np.sum(ind) == 0:
                continue
            pixels = np.where(ind)
            points = np.array(pixels).transpose((1, 0))[:, ::-1].astype('float32')

            bbox_ind = jaccard(points, kpoints)
            offset_gt = kpoints[bbox_ind] - points
            gt_distance[:, pixels[0], pixels[1]] = offset_gt.T * 0.1

        # img_name = img_path.split('/')[-1].split('.')[0]
        # self.vis(img_name, img, gt_instance, training_mask, gt_kernel, gt_kernel_instance, gt_kernel_inner, gt_distance)

        img = Image.fromarray(img)
        img = img.convert('RGB')

        if self.is_transform:
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_kernel = torch.from_numpy(gt_kernel).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_kernel_instance = torch.from_numpy(gt_kernel_instance).long()
        training_mask_distance = torch.from_numpy(training_mask_distance).long()
        gt_distance = torch.from_numpy(gt_distance).float()

        data = dict(
            imgs=img,
            gt_kernels=gt_kernel,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_kernel_instances=gt_kernel_instance,
            training_mask_distances=training_mask_distance,
            gt_distances=gt_distance
        )

        return data

    def prepare_test_data(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path, self.read_type)
        img_meta = dict(
            ori_img=img,
            org_img_size=np.array(img.shape[:2])
        )

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(
            img_size=np.array(img.shape[:2]),
            imgs=img,
            img_name=img_path.split('/')[-1].split('.')[0]
        ))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        data = dict(
            imgs=img,
            img_metas=img_meta
        )

        return data

    def __getitem__(self, index):
        if self.split == 'train':
            return self.prepare_train_data(index)
        elif self.split == 'test':
            return self.prepare_test_data(index)