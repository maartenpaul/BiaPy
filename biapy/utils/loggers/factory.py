"""Factory for building the experiment tracking logger from config."""

from typing import Optional

from yacs.config import CfgNode as CN

from biapy.utils.loggers.base_logger import BaseLogger, CompositeLogger


def build_logger(cfg: CN, job_identifier: str) -> Optional[CompositeLogger]:
    """
    Build a :class:`CompositeLogger` based on the ``LOG`` section of the BiaPy config.

    Parameters
    ----------
    cfg : CN
        BiaPy configuration.
    job_identifier : str
        Name of the current job/run.

    Returns
    -------
    CompositeLogger or None
        A composite logger wrapping all selected backends, or ``None`` if
        ``LOG.LOGGER`` is ``"NONE"``.
    """
    loggers: list[BaseLogger] = []
    logger_type = cfg.LOG.LOGGER.upper()

    if logger_type in ("TENSORBOARD", "ALL"):
        from biapy.utils.loggers.tensorboard_logger import TensorboardLogger

        loggers.append(TensorboardLogger(log_dir=cfg.LOG.TENSORBOARD_LOG_DIR))

    if logger_type in ("WANDB", "ALL"):
        from biapy.utils.loggers.wandb_logger import WandbLogger

        loggers.append(
            WandbLogger(
                project=cfg.LOG.WANDB.PROJECT if cfg.LOG.WANDB.PROJECT else job_identifier,
                entity=cfg.LOG.WANDB.ENTITY if cfg.LOG.WANDB.ENTITY else None,
                name=job_identifier,
                tags=list(cfg.LOG.WANDB.TAGS) if cfg.LOG.WANDB.TAGS else None,
                save_dir=cfg.LOG.WANDB.SAVE_DIR if cfg.LOG.WANDB.SAVE_DIR else None,
                group=cfg.LOG.WANDB.GROUP if cfg.LOG.WANDB.GROUP else None,
            )
        )

    if logger_type in ("MLFLOW", "ALL"):
        from biapy.utils.loggers.mlflow_logger import MLflowLogger

        loggers.append(
            MLflowLogger(
                tracking_uri=cfg.LOG.MLFLOW.TRACKING_URI if cfg.LOG.MLFLOW.TRACKING_URI else None,
                experiment_name=cfg.LOG.MLFLOW.EXPERIMENT_NAME if cfg.LOG.MLFLOW.EXPERIMENT_NAME else job_identifier,
                run_name=job_identifier,
            )
        )

    if not loggers:
        return None

    return CompositeLogger(loggers)
