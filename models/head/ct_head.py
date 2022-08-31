import torch
import torch.nn as nn
import math
import numpy as np
import cv2
import time
from ..loss import build_loss, ohem_batch, iou
from IPython import embed


class CT_Head(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_classes,
                 loss_kernel,
                 loss_loc):
        super(CT_Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0)

        self.kernel_loss = build_loss(loss_kernel)
        self.loc_loss = build_loss(loss_loc)

        self.coord = np.zeros((2, 300, 300), dtype=np.int32)
        for i in range(300):
            for j in range(300):
                self.coord[0, i, j] = j
                self.coord[1, i, j] = i

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)

        return out

    def get_results(self, out, img_meta, cfg):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        score = torch.sigmoid(out[:, 0, :, :])
        kernel = out[:, 0, :, :] > 0.2
        loc = out[:, 1:, :, :].float()

        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernel = kernel.data.cpu().numpy()[0].astype(np.uint8)
        loc = loc.data.cpu().numpy()[0].astype(np.float32)

        label_num, label_kernel = cv2.connectedComponents(kernel, connectivity=4)
        for i in range(1, label_num):
            ind = (label_kernel == i)
            if ind.sum() < 10:
                label_kernel[ind] = 0

        label = np.zeros_like(label_kernel)
        h, w = label_kernel.shape
        pixels = self.coord[:, :h, :w].reshape(2, -1)
        points = pixels.transpose([1, 0]).astype(np.float32)

        off_points = (points + 10. / 4. * loc[:, pixels[1], pixels[0]].T).astype(np.int32)
        off_points[:, 0] = np.clip(off_points[:, 0], 0, label.shape[1] - 1)
        off_points[:, 1] = np.clip(off_points[:, 1], 0, label.shape[0] - 1)
        label[pixels[1], pixels[0]] = label_kernel[off_points[:, 1], off_points[:, 0]]
        label[label_kernel > 0] = label_kernel[label_kernel > 0]

        score_pocket = [0.0]
        for i in range(1, label_num):
            # ind = ((label_kernel == i) & (label == i))
            ind = (label_kernel == i)
            if ind.sum() == 0:
                score_pocket.append(0.0)
                continue
            score_i = np.mean(score[ind])
            score_pocket.append(score_i)

        # image size
        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]

        label_num = np.max(label) + 1
        label = cv2.resize(label, (int(np.float32(img_size[1])), int(np.float32(img_size[0]))), interpolation=cv2.INTER_NEAREST)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_pa_time=time.time() - start
            ))

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        bboxes = []
        scores = []
        for i in range(1, label_num):
            ind = (label == i)
            points = np.array(np.where(ind)).transpose((1, 0))

            if points.shape[0] < cfg.test_cfg.min_area:
                continue

            score_i = score_pocket[i]
            if score_i < cfg.test_cfg.min_score:
                continue

            if cfg.test_cfg.bbox_type == 'rect':
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
                z = bbox.mean(0)
                bbox = z + (bbox - z) * 0.85
            elif cfg.test_cfg.bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                try:
                	_, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                except BaseException:
                	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # contour = contours[0]
                # epsilon = 0.01 * cv2.arcLength(contour, True)
                # approx = cv2.approxPolyDP(contour, epsilon, True)
                # bbox = approx * scale
                bbox = contours[0] * scale

            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        outputs.update(dict(
            bboxes=bboxes,
            scores=scores
        ))

        # self.vis(img_meta, score.copy(), kernel.copy(), label_kernel.copy(), label.copy(), outputs, loc.copy())
        # embed()

        return outputs

    def vis(self, img_meta, score, kernel, label_kernel, label, outputs, loc):
        color_list = []
        import random
        for _ in range(100):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color_list.append((r, g, b))

        img_name = 'vis/' + img_meta['img_name'][0]
        img = img_meta['imgs'].numpy()[0]
        # img
        cv2.imwrite(img_name + '.jpg', img)
        # score map
        pmin = np.min(score)
        pmax = np.max(score)
        score = ((score - pmin) / (pmax - pmin + 0.000001)) * 255
        score = score.astype(np.uint8)
        vis_score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
        cv2.imwrite(img_name + '_score.jpg', vis_score)
        # kernel map
        cv2.imwrite(img_name + '_kernel.jpg', kernel * 255)
        # label kernel map
        vis_label_kernel = np.zeros((*kernel.shape, 3), np.uint8)
        label_num = np.max(label_kernel)
        for i in range(1, label_num + 1):
            ind = (label_kernel == i)
            vis_label_kernel[ind] = color_list[i]
        cv2.imwrite(img_name + '_label_kernel.jpg', vis_label_kernel)
        # label text map
        vis_text = np.zeros((*label.shape, 3), np.uint8)
        for i in range(1, label_num + 1):
            ind = (label == i)
            vis_text[ind] = color_list[i]
        cv2.imwrite(img_name + '_text.jpg', vis_text)
        # distance map
        vis_distance = np.sum(np.abs(loc), 0)
        pmin = np.min(vis_distance)
        pmax = np.max(vis_distance)
        vis_distance = ((vis_distance - pmin) / (pmax - pmin + 0.000001)) * 255
        vis_distance = vis_distance.astype(np.uint8)
        vis_distance = cv2.applyColorMap(vis_distance, cv2.COLORMAP_JET)
        cv2.imwrite(img_name + '_distance.jpg', vis_distance)
        # result map
        oimg = img_meta['ori_img'].numpy()[0]
        vis_res = oimg.copy()
        for bbox, score in zip(outputs['bboxes'], outputs['scores']):
            cv2.drawContours(vis_res, [bbox.reshape(-1, 2)], -1, (255, 255, 0), 2)
            cv2.putText(vis_res, "{:.2f}".format(score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imwrite(img_name+'_result.jpg', vis_res)

    def loss(self, out, gt_kernels, training_masks, gt_instances, gt_kernel_instances, training_mask_distances, gt_distances):
        # output
        kernels = out[:, 0, :, :]
        distances = out[:, 1:, :, :]

        # kernel loss
        selected_masks = ohem_batch(kernels, gt_kernels, training_masks)
        loss_kernel = self.kernel_loss(kernels, gt_kernels, selected_masks, reduce=False)
        iou_kernel = iou((kernels > 0).long(), gt_kernels, training_masks, reduce=False)
        losses = dict(
            loss_kernels=loss_kernel,
            iou_kernel=iou_kernel
        )

        # loc loss
        loss_loc, iou_text = self.loc_loss(distances, gt_instances, gt_kernel_instances, training_mask_distances, gt_distances, reduce=False)
        losses.update(dict(
            loss_loc=loss_loc,
            iou_text=iou_text
        ))

        return losses
