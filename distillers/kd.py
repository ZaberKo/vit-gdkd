import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller


def kd_loss_fn(student_logits, teacher_logits, temperature):
    log_pred_student = F.log_softmax(student_logits / temperature, dim=1)
    pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def forward_train(self, image, target, **kwargs):
        student_logits = self.student(image)
        with torch.inference_mode():
            teacher_logits = self.teacher(image)
        # losses
        ce_loss = self.target_criterion(student_logits, target)
        kd_loss = kd_loss_fn(student_logits, teacher_logits, self.cfg.T)

        loss = self.cfg.loss_weights.ce * ce_loss + self.cfg.loss_weights.kd * kd_loss

        info_dict = {
            "ce_loss": ce_loss,
            "kd_loss": kd_loss,
            "student_logits": student_logits,
        }
        return loss, info_dict
