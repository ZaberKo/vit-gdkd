import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller
from .utils import kl_div

"""
    This is the improved version: mdistiller's DKDMod
"""


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def get_top1_masks(logits, target):
    # NOTE: masks are calculated in cuda

    # top1 mask
    max_indices = logits.argmax(dim=1, keepdim=True)
    mask_u1 = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, max_indices, 1)

    # other mask
    mask_u2 = torch.ones_like(logits, dtype=torch.bool).scatter_(1, max_indices, 0)

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def dkd_loss_fn(
    student_logits,
    teacher_logits,
    target,
    temperature,
    mask_magnitude=1000,
    kl_type="forward",
    strategy="target",
):
    if strategy == "target":
        gt_mask = _get_gt_mask(teacher_logits, target)
        other_mask = _get_other_mask(teacher_logits, target)
    elif strategy == "top1":
        gt_mask, other_mask = get_top1_masks(teacher_logits, target)
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))

    soft_student_logits = student_logits / temperature
    soft_teacher_logits = teacher_logits / temperature

    p_student = F.softmax(soft_student_logits, dim=1)
    p_teacher = F.softmax(soft_teacher_logits, dim=1)
    p0_student = cat_mask(p_student, gt_mask, other_mask)
    p0_teacher = cat_mask(p_teacher, gt_mask, other_mask)

    # tckd_loss = (
    #     F.binary_cross_entropy(pred_student, pred_teacher, reduction="mean")
    #     * (temperature**2)
    # )
    log_p0_student = torch.log(p0_student)
    tckd_loss = F.kl_div(log_p0_student, p0_teacher, reduction="batchmean") * (
        temperature**2
    )

    log_p2_student = F.log_softmax(
        soft_student_logits - mask_magnitude * gt_mask, dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_teacher_logits - mask_magnitude * gt_mask, dim=1
    )

    nckd_loss = kl_div(log_p2_student, log_p2_teacher, temperature, kl_type=kl_type)

    return tckd_loss, nckd_loss


class DKD(Distiller):
    """DKD with some new losses"""

    def __call__(self, image, target, **kwargs):
        student_logits = self.student(image)
        with torch.inference_mode():
            teacher_logits = self.teacher(image)

        # losses
        ce_loss = self.target_criterion(student_logits, target)

        if kwargs["epoch"] < self.cfg.skip_epochs:
            loss = ce_loss
            dkd_loss = torch.zeros_like(loss)
            tckd_loss = torch.zeros_like(loss)
            nckd_loss = torch.zeros_like(loss)
        else:
            tckd_loss, nckd_loss = dkd_loss_fn(
                student_logits,
                teacher_logits,
                target,
                temperature=self.cfg.T,
            )
            dkd_loss = (
                self.cfg.loss_weights.alpha * tckd_loss
                + self.cfg.loss_weights.beta * nckd_loss
            )

            if self.cfg.warmup_epochs <= 0:
                ratio = 1.0
            else:
                ratio = min(
                    (kwargs["epoch"] - self.cfg.skip_epochs) / self.cfg.warmup_epochs,
                    1.0,
                )

            dkd_loss = ratio * dkd_loss

            loss = self.cfg.loss_weights.ce * ce_loss + dkd_loss

        info_dict = {
            "ce_loss": ce_loss.detach(),
            "dkd_loss": dkd_loss.detach(),
            "tckd_loss": tckd_loss.detach(),
            "nckd_loss": nckd_loss.detach(),
            "student_logits": student_logits.detach(),
        }
        return loss, info_dict
