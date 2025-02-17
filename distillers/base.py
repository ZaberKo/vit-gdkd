import torch.nn as nn


class Distiller:
    def __init__(self, student, teacher, target_criterion, cfg):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.target_criterion = target_criterion  # criterion for standard training
        self.cfg = cfg

    def train(self, mode=True):
        self.student.train(mode)
        self.teacher.eval()
        return self

    def eval(self):
        self.train(False)

    def __call__(self, image, target, **kwargs):
        raise NotImplementedError
