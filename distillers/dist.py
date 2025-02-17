import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(
        a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps
    )


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


def dist_loss_fn(student_logits, teacher_logits, T):
    y_s = F.softmax(student_logits / T, dim=1)
    y_t = F.softmax(teacher_logits / T, dim=1)
    inter_loss = inter_class_relation(y_s, y_t) * (T**2)
    intra_loss = intra_class_relation(y_s, y_t) * (T**2)

    return inter_loss, intra_loss


class DIST(Distiller):
    def __call__(self, image, target, **kwargs):
        student_logits = self.student(image)
        with torch.inference_mode():
            teacher_logits = self.teacher(image)

        # losses
        ce_loss = self.target_criterion(student_logits, target)

        if kwargs["epoch"] < self.cfg.skip_epochs:
            loss = ce_loss
            inter_loss = torch.zeros_like(loss)
            intra_loss = torch.zeros_like(loss)
            dist_loss = torch.zeros_like(loss)
        else:
            inter_loss, intra_loss = dist_loss_fn(
                student_logits,
                teacher_logits,
                T=self.cfg.T,
            )

            dist_loss = (
                self.cfg.loss_weights.beta * inter_loss
                + self.cfg.loss_weights.gamma * intra_loss
            )

            loss = self.cfg.loss_weights.ce * ce_loss + dist_loss

        info_dict = {
            "ce_loss": ce_loss.detach(),
            "dist_loss": dist_loss.detach(),
            "inter_loss": inter_loss.detach(),
            "intra_loss": intra_loss.detach(),
            "student_logits": student_logits.detach(),
        }
        return loss, info_dict
