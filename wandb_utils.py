import os
import copy
import torch.distributed
import wandb
from pathlib import Path
import torch

from utils import MetricLogger, is_main_process


def setup_wandb(args):
    if is_main_process():
        # os.environ["WANDB_DATA_DIR"] = Path(args.output_dir)/"wandb_cache"
        # os.environ["WANDB_CACHE_DIR"] = Path(args.output_dir)/"wandb_cache"

        distiller_name = getattr(args, "distiller", "vanilla")
        if distiller_name == "vanilla":
            exp_name = f"{args.model}-vanilla"
            tags = [args.model]
        else:
            exp_name = f"{args.model}-{args.teacher}-{distiller_name}"
            tags = [args.model, args.teacher]

        config = copy.deepcopy(vars(args))
        del config["tags"]

        if len(args.tags) > 0:
            exp_name = f"{exp_name}|{args.tags}"
            tags += args.tags.split(",")

        wandb_kwargs = dict(
            project="torch-distil",
            name=exp_name,
            group=exp_name,
            config=config,
            tags=tags,
            dir=Path(args.output_dir).absolute(),
        )

        if args.resume:
            assert args.wandb_resume_id is not None, "wandb_id must be provided when resuming"

            wandb_kwargs.update(
                id=args.wandb_resume_id,
                resume="must",
            )

        wandb.init(**wandb_kwargs)


def wandb_record(metrics: MetricLogger, step=None, group=None | str):
    """
    Note: We assume that metrics have been synced
    """
    if is_main_process():
        if group is None:
            metrics_dict = {k: v.global_avg for k, v in metrics.meters.items()}
        else:
            metrics_dict = {
                f"{group}/{k}": v.global_avg for k, v in metrics.meters.items()
            }

        wandb.log(metrics_dict, step=step)
