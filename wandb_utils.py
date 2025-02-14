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
        exp_name = f"{args.model}-{args.teacher}-{distiller_name}"

        config = copy.deepcopy(vars(args))
        del config["wandb_tags"]

        tags = [args.model, args.teacher]
        
        if len(args.wandb_tags) > 0:
            exp_name = f"{exp_name}|{args.wandb_tags}"
            tags += args.wandb_tags.split(",")

        wandb.init(
            project="torch-distil",
            name=exp_name,
            group=exp_name,
            config=config,
            tags=tags,
            dir=Path(args.output_dir).absolute(),
        )


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
