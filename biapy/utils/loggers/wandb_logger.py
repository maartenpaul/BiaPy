"""Weights & Biases logging backend for BiaPy."""

import os
from typing import Any, Dict, List, Optional

from biapy.utils.loggers.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    """
    Weights & Biases experiment tracking backend.

    Lazily imports ``wandb`` so that it is only required when this backend
    is selected.

    Parameters
    ----------
    project : str
        wandb project name.
    entity : str, optional
        wandb entity (team or username).
    name : str, optional
        Display name for this run.
    tags : list of str, optional
        Tags to attach to the run.
    save_dir : str, optional
        Local directory for wandb files.
    group : str, optional
        wandb run group.
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        save_dir: Optional[str] = None,
        group: Optional[str] = None,
    ):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is required for the 'wandb' logger. "
                "Install it with: pip install wandb"
            )
        self._wandb = wandb
        self.step = 0

        init_kwargs: Dict[str, Any] = {
            "project": project,
            "name": name,
        }
        if entity:
            init_kwargs["entity"] = entity
        if tags:
            init_kwargs["tags"] = tags
        if group:
            init_kwargs["group"] = group
        if save_dir:
            init_kwargs["dir"] = save_dir

        self._run = wandb.init(**init_kwargs)

    def set_step(self, step: Optional[int] = None) -> None:
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head: str = "scalar", step: Optional[int] = None, **kwargs) -> None:
        import torch

        log_dict: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            log_dict[f"{head}/{k}"] = v
        if log_dict:
            self._wandb.log(log_dict, step=self.step if step is None else step)

    def flush(self) -> None:
        # wandb handles flushing automatically
        pass

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        self._wandb.config.update(params, allow_val_change=True)

    def log_artifact(self, local_path: str, name: Optional[str] = None) -> None:
        artifact_name = name or os.path.basename(local_path)
        # Sanitize artifact name: wandb only allows alphanumeric, dashes, underscores, dots
        artifact_name = artifact_name.replace("/", "_").replace("\\", "_")
        artifact = self._wandb.Artifact(artifact_name, type="model")
        artifact.add_file(local_path)
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        self._wandb.finish()
