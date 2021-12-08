import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

        self.coord = nn.Parameter(torch.zeros([640, 640, 2]).long(), requires_grad=False)
        for i in range(640):
            for j in range(640):
                self.coord[i, j, 0] = j
                self.coord[i, j, 1] = i
        self.coord.data = self.coord.view(-1, 2) # (h*w, 2)

    def forward_single(self, input, target, mask, beta=1.0, eps=1e-6):
        batch_size = input.size(0)

        diff = torch.abs(input - target) * mask.unsqueeze(1)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                           diff - 0.5 * beta)
        loss = loss.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()
        loss = torch.sum(loss, dim=-1)
        loss = loss / (mask.sum(dim=-1) + eps)

        return loss

    def select_single(self, distance, gt_instance, gt_kernel_instance, training_mask):

        with torch.no_grad():
            off_points = (self.coord.float() + 10 * distance[:, self.coord[:, 1], self.coord[:, 0]].transpose(1, 0)).long() # (h*w, 2)
            off_points = torch.clamp(off_points, 0, distance.size(-1) - 1)
            selected_mask = (gt_instance[self.coord[:, 1], self.coord[:, 0]] != gt_kernel_instance[off_points[:, 1], off_points[:, 0]])
            selected_mask = selected_mask.contiguous().view(1, -1, distance.shape[-1]).long()
            selected_training_mask = selected_mask * training_mask

            return selected_training_mask

    def forward(self, distances, gt_instances, gt_kernel_instances, training_masks, gt_distances, reduce=True):

        selected_training_masks = []
        for i in range(distances.size(0)):
            selected_training_masks.append(
                self.select_single(distances[i, :, :, :], gt_instances[i, :, :],
                                    gt_kernel_instances[i, :, :], training_masks[i, :, :])
            )
        selected_training_masks = torch.cat(selected_training_masks, 0).float()

        loss = self.forward_single(distances, gt_distances, selected_training_masks, self.beta)
        loss = self.loss_weight * loss

        with torch.no_grad():
            batch_size = distances.size(0)
            false_num = selected_training_masks.contiguous().view(batch_size, -1)
            false_num = false_num.sum(dim=-1)
            total_num = training_masks.contiguous().view(batch_size, -1).float()
            total_num = total_num.sum(dim=-1)
            iou_text = (total_num - false_num) / (total_num + 1e-6)

        if reduce:
            loss = torch.mean(loss)

        return loss, iou_text
