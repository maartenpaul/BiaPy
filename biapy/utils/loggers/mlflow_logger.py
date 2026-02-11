"""MLflow logging backend for BiaPy."""

from typing import Any, Dict, Optional

from biapy.utils.loggers.base_logger import BaseLogger


class MLflowLogger(BaseLogger):
    """
    MLflow experiment tracking backend.

    Lazily imports ``mlflow`` so that it is only required when this backend
    is selected.

    Parameters
    ----------
    tracking_uri : str, optional
        MLflow tracking server URI (e.g. ``"http://localhost:5000"``).
        If None, uses the default local file store.
    experiment_name : str, optional
        MLflow experiment name. Created if it does not exist.
    run_name : str, optional
        Display name for this run.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "mlflow is required for the 'mlflow' logger. "
                "Install it with: pip install mlflow"
            )
        self._mlflow = mlflow
        self.step = 0

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        self._run = mlflow.start_run(run_name=run_name)

    def set_step(self, step: Optional[int] = None) -> None:
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head: str = "scalar", step: Optional[int] = None, **kwargs) -> None:
        import torch

        metrics: Dict[str, float] = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            metrics[f"{head}/{k}"] = v
        if metrics:
            self._mlflow.log_metrics(metrics, step=self.step if step is None else step)

    def flush(self) -> None:
        # MLflow logs are sent immediately; nothing to flush.
        pass

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        flat = self._flatten_dict(params)
        # MLflow has a 500-param limit and 6000-char value limit.
        # Batch in groups of 100 and truncate values.
        items = list(flat.items())
        for i in range(0, len(items), 100):
            batch = {k: str(v)[:6000] for k, v in items[i : i + 100]}
            self._mlflow.log_params(batch)

    def log_artifact(self, local_path: str, name: Optional[str] = None) -> None:
        self._mlflow.log_artifact(local_path)

    def finish(self) -> None:
        self._mlflow.end_run()

    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """
        Flatten a nested dict into dot-separated keys.

        Parameters
        ----------
        d : dict
            Nested dictionary to flatten.
        parent_key : str
            Prefix for keys (used in recursion).
        sep : str
            Separator between levels.

        Returns
        -------
        dict
            Flattened dictionary.
        """
        items: list = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowLogger._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
