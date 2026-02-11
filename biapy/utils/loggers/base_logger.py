"""
Abstract base logger and composite logger for BiaPy experiment tracking.

Provides a unified interface for logging metrics, hyperparameters, and artifacts
to different backends (TensorBoard, wandb, MLflow).
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLogger(ABC):
    """
    Abstract interface that every logging backend must implement.

    The interface mirrors the existing TensorboardLogger so that all current
    call sites (train_one_epoch, base_workflow) work without modification.
    """

    @abstractmethod
    def set_step(self, step: Optional[int] = None) -> None:
        """
        Set or increment the global step counter.

        Parameters
        ----------
        step : int, optional
            The specific step number to set. If None, increments the current step.
        """
        ...

    @abstractmethod
    def update(self, head: str = "scalar", step: Optional[int] = None, **kwargs) -> None:
        """
        Log one or more scalar metrics.

        Parameters
        ----------
        head : str
            Category prefix (e.g. "loss", "opt", "perf").
        step : int, optional
            Override for the global step.
        **kwargs
            Metric name/value pairs.
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Ensure all pending data is written."""
        ...

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters / config at the start of a run.

        Parameters
        ----------
        params : dict
            Dictionary of hyperparameter name/value pairs.
        """
        pass

    def log_artifact(self, local_path: str, name: Optional[str] = None) -> None:
        """
        Log a file artifact (checkpoint, chart image).

        Parameters
        ----------
        local_path : str
            Path to the file to log.
        name : str, optional
            Name for the artifact. If None, uses the file basename.
        """
        pass

    def finish(self) -> None:
        """Clean up / finalize the run."""
        pass


class CompositeLogger(BaseLogger):
    """
    Dispatches every call to a list of backend loggers.

    This is the object that base_workflow.py holds as ``self.log_writer``.
    It forwards set_step/update/flush/log_hyperparameters/log_artifact/finish
    to every registered backend.

    Parameters
    ----------
    loggers : list of BaseLogger
        Backend logger instances to dispatch calls to.
    """

    def __init__(self, loggers: List[BaseLogger]):
        self._loggers = loggers
        self.step = 0

    def set_step(self, step: Optional[int] = None) -> None:
        if step is not None:
            self.step = step
        else:
            self.step += 1
        for logger in self._loggers:
            logger.set_step(step)

    def update(self, head: str = "scalar", step: Optional[int] = None, **kwargs) -> None:
        for logger in self._loggers:
            logger.update(head=head, step=step, **kwargs)

    def flush(self) -> None:
        for logger in self._loggers:
            logger.flush()

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        for logger in self._loggers:
            logger.log_hyperparameters(params)

    def log_artifact(self, local_path: str, name: Optional[str] = None) -> None:
        for logger in self._loggers:
            try:
                logger.log_artifact(local_path, name)
            except Exception as e:
                print(f"Warning: Failed to log artifact to {type(logger).__name__}: {e}")

    def finish(self) -> None:
        for logger in self._loggers:
            logger.finish()
