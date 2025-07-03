import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller
from .utils import kl_div

MASK_MAGNITUDE = 1000.0


def get_masks(logits, k=5):
    ranks = torch.topk(logits, k, dim=-1, largest=True, sorted=True).indices
    top1_ranks = ranks[:, :1]
    top2_k_ranks = ranks[:, 1:]

    # top1 mask
    mask_u1 = torch.zeros_like(
        logits, dtype=torch.bool).scatter_(1, top1_ranks, 1)
    # top2-k mask
    mask_u2 = torch.zeros_like(
        logits, dtype=torch.bool).scatter_(1, top2_k_ranks, 1)
    # other mask
    mask_u3 = torch.logical_not(
        torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ranks, 1)
    )

    return mask_u1, mask_u2, mask_u3


def cat_mask(t, mask1, mask2, mask3):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    t3 = (t * mask3).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2, t3], dim=1)  # [B, 3]
    return rt


def gdkd3_loss_fn(
    student_logits,
    teacher_logits,
    k,
    temperature,
):
    mask_u1, mask_u2, mask_u3 = get_masks(teacher_logits, k)

    soft_student_logits = student_logits / temperature
    soft_logits_teacher = teacher_logits / temperature

    p_student = F.softmax(soft_student_logits, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # Notation: high_loss: level 0 loss; low_loss: level 1 loss
    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2, mask_u3)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2, mask_u3)

    log_p0_student = torch.log(p0_student)
    high_loss = F.kl_div(log_p0_student, p0_teacher, reduction="batchmean") * (
        temperature**2
    )

    # top2-k loss
    log_p1_student = F.log_softmax(
        soft_student_logits - MASK_MAGNITUDE * mask_u2.logical_not(), dim=1
    )
    log_p1_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * mask_u2.logical_not(), dim=1
    )

    low_top_loss = kl_div(log_p1_student, log_p1_teacher, temperature)

    # other classes loss
    log_p2_student = F.log_softmax(
        soft_student_logits - MASK_MAGNITUDE * mask_u3.logical_not(), dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * mask_u3.logical_not(), dim=1
    )

    low_other_loss = kl_div(log_p2_student, log_p2_teacher, temperature)

    return high_loss, low_top_loss, low_other_loss


class GDKD3(Distiller):
    def __call__(self, image, target, **kwargs):
        student_logits = self.student(image)
        with torch.inference_mode():
            teacher_logits = self.teacher(image)

        # losses
        ce_loss = self.target_criterion(student_logits, target)

        if kwargs["epoch"] < self.cfg.skip_epochs:
            loss = ce_loss
            high_loss = torch.zeros_like(loss)
            low_top_loss = torch.zeros_like(loss)
            low_other_loss = torch.zeros_like(loss)
            gdkd_loss = torch.zeros_like(loss)
        else:
            high_loss, low_top_loss, low_other_loss = gdkd3_loss_fn(
                student_logits, teacher_logits, k=self.cfg.k, temperature=self.cfg.T
            )

            gdkd_loss = (
                self.cfg.loss_weights.w0 * high_loss
                + self.cfg.loss_weights.w1 * low_top_loss
                + self.cfg.loss_weights.w2 * low_other_loss
            )

            if self.cfg.warmup_epochs <= 0:
                ratio = 1.0
            else:
                ratio = min(
                    (kwargs["epoch"] - self.cfg.skip_epochs) / self.cfg.warmup_epochs,
                    1.0,
                )

            gdkd_loss = ratio * gdkd_loss

            loss = self.cfg.loss_weights.ce * ce_loss + gdkd_loss

        info_dict = {
            "ce_loss": ce_loss.detach(),
            "gdkd_loss": gdkd_loss.detach(),
            "high_loss": high_loss.detach(),
            "low_top_loss": low_top_loss.detach(),
            "low_other_loss": low_other_loss.detach(),
            "student_logits": student_logits.detach(),
        }
        return loss, info_dict
