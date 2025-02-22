import datetime
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
import numpy as np

import torch
import torch.utils.data
import torchvision
import torchvision.transforms
from torch import nn
from torch.utils.data.dataloader import default_collate
import utils
from transforms import get_mixup_cutmix
from dataset import load_data
from wandb_utils import setup_wandb, wandb_record

"""
    Compute logits of a model
"""

os.environ["TORCH_HOME"] = "./torch_home"


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        correct_flags_list = []
        for k in topk:
            correct_flags_list.append(correct[:k].any(dim=0))
    return correct_flags_list


def get_logits(model, data_loader, device, args, header=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    logits_dict = defaultdict(list)

    for i, (_image, _target) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        start_time = time.time()
        image, target = _image.to(device), _target.to(device)
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(image)

        acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
        batch_size = image.shape[0]
        num_classes = logits.shape[-1]
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        logits = logits.to("cpu")
        for j in range(num_classes):
            logits_dict[j].append(logits[_target == j])

    metric_logger.synchronize_between_processes()

    final_logits_dict = {}
    for i in range(num_classes):
        final_logits_dict[f"class{i}"] = torch.concat(logits_dict[i]).numpy()

    return final_logits_dict


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    utils.print_args(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args
    )

    num_classes = len(dataset.classes)
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        num_classes=num_classes,
        use_v2=args.use_v2,
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    data_loader = data_loader_test if args.valset else data_loader_train

    print("Creating model")
    model = torchvision.models.get_model(
        args.model, weights=args.weights, num_classes=num_classes
    )
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    logits_dict = get_logits(
        model,
        data_loader,
        device=device,
        args=args,
        header=args.model,
    )

    path = os.path.join(args.output_dir, args.save_name)
    np.savez_compressed(path, **logits_dict)


def get_args_parser(add_help=True):
    parser = utils.get_default_args_parser()

    #  ===== new =======
    parser.add_argument("--save-name", type=str)
    parser.add_argument(
        "--valset",
        action="store_true",
        help="whether to use trainset or valset for logits",
    )

    return parser


def post_setup_args(args):
    args.ra_sampler = False
    args.output_dir = os.path.join(args.output_dir, "logits")
    args.weights = args.weights if args.weights is not None else "DEFAULT"

    if args.valset:
        suffix = "val"
    else:
        suffix = "train"
    if args.save_name is None:
        args.save_name = f"{args.model}_{suffix}.npz"


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    post_setup_args(args)

    main(args)

# Example:
