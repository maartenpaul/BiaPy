"""TensorBoard logging backend for BiaPy."""

import torch
from typing import Optional
from tensorboardX import SummaryWriter

from biapy.utils.loggers.base_logger import BaseLogger


class TensorboardLogger(BaseLogger):
    """
    Wrapper around ``tensorboardX.SummaryWriter`` implementing :class:`BaseLogger`.

    Parameters
    ----------
    log_dir : str
        The directory where TensorBoard log files will be saved.
    """

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step: Optional[int] = None) -> None:
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head: str = "scalar", step: Optional[int] = None, **kwargs) -> None:
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self) -> None:
        self.writer.flush()

    def finish(self) -> None:
        self.writer.close()
