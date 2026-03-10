"""
Simplified inference API for BiaPy.

Provides a ``predict()`` function and CLI entry point that only require a
checkpoint path and an input image directory — no YAML configuration file
needed.  The full configuration is extracted from the checkpoint (BiaPy
v3.5.1+ checkpoints embed it) and a temporary config is generated
automatically.
"""

import os
import sys
import argparse
import tempfile
from functools import partial
from typing import Optional, Tuple

import torch
import yaml
from yacs.config import CfgNode as CN

from biapy.engine.check_configuration import convert_old_model_cfg_to_current_version


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfgnode_to_dict(node):
    """Recursively convert a YACS CfgNode to a plain dict."""
    if isinstance(node, CN):
        return {k: _cfgnode_to_dict(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_cfgnode_to_dict(v) for v in node]
    if isinstance(node, tuple):
        return list(node)
    return node


def extract_config_from_checkpoint(
    checkpoint_path: str,
) -> Tuple[CN, Optional[str]]:
    """
    Load a BiaPy checkpoint and return the embedded configuration.

    Parameters
    ----------
    checkpoint_path : str
        Path to a ``.pth`` checkpoint file.

    Returns
    -------
    cfg : yacs.config.CfgNode
        The configuration stored in the checkpoint.
    biapy_version : str or None
        BiaPy version that produced the checkpoint, if available.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_path* does not exist.
    ValueError
        If the checkpoint does not contain an embedded configuration
        (pre-v3.5.1 or non-BiaPy checkpoints).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if checkpoint_path.endswith(".safetensors"):
        raise ValueError(
            "safetensors checkpoints do not store configuration. "
            "Please use a .pth checkpoint from BiaPy v3.5.1+ or provide "
            "a full YAML configuration file via the standard BiaPy interface."
        )

    # Register the same safe globals that BiaPy uses when loading checkpoints
    torch.serialization.add_safe_globals([CN])
    torch.serialization.add_safe_globals([set])
    torch.serialization.add_safe_globals([partial])
    torch.serialization.add_safe_globals([torch.nn.modules.normalization.LayerNorm])

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "cfg" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain an embedded configuration. "
            "This simplified API requires BiaPy v3.5.1+ checkpoints. "
            "Please provide a full YAML configuration file and use the "
            "standard BiaPy interface instead:\n\n"
            "  biapy --config your_config.yaml --result_dir results\n"
        )

    return (
        checkpoint["cfg"],
        checkpoint.get("biapy_version"),
    )


def _build_inference_config(
    checkpoint_cfg: CN,
    checkpoint_path: str,
    input_path: str,
    patch_size: Optional[tuple] = None,
) -> dict:
    """
    Build a YAML-serialisable config dict for inference-only execution.

    Takes the configuration extracted from a checkpoint, applies version
    migration, and overrides the keys necessary for a pure-inference run.
    """
    # Migrate old config keys to current version
    cfg_dict = _cfgnode_to_dict(checkpoint_cfg)
    cfg_dict = convert_old_model_cfg_to_current_version(cfg_dict)

    # --- Enforce inference-only settings ---
    cfg_dict.setdefault("TRAIN", {})
    cfg_dict["TRAIN"]["ENABLE"] = False

    cfg_dict.setdefault("TEST", {})
    cfg_dict["TEST"]["ENABLE"] = True

    cfg_dict.setdefault("DATA", {})
    cfg_dict["DATA"].setdefault("TEST", {})
    cfg_dict["DATA"]["TEST"]["PATH"] = os.path.abspath(input_path)
    cfg_dict["DATA"]["TEST"]["LOAD_GT"] = False

    cfg_dict.setdefault("MODEL", {})
    cfg_dict["MODEL"]["LOAD_CHECKPOINT"] = True
    # Config is already applied — don't re-extract from checkpoint
    cfg_dict["MODEL"]["LOAD_MODEL_FROM_CHECKPOINT"] = False

    cfg_dict.setdefault("PATHS", {})
    cfg_dict["PATHS"]["CHECKPOINT_FILE"] = os.path.abspath(checkpoint_path)

    if patch_size is not None:
        cfg_dict["DATA"]["PATCH_SIZE"] = list(patch_size)

    return cfg_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    model: str,
    input: str,
    output: str = "./biapy_predictions",
    gpu: Optional[str] = None,
    patch_size: Optional[tuple] = None,
    name: str = "biapy_predict",
    run_id: int = 1,
) -> str:
    """
    Run inference with a trained BiaPy model.

    This is the simplified entry point: just provide a model checkpoint and
    an input directory.  The full configuration is read from the checkpoint
    (BiaPy v3.5.1+).

    Parameters
    ----------
    model : str
        Path to a ``.pth`` checkpoint file that contains an embedded
        BiaPy configuration.
    input : str
        Path to a directory of input images.
    output : str, optional
        Directory where results will be written.  Defaults to
        ``"./biapy_predictions"``.
    gpu : str or None, optional
        GPU device id (e.g. ``"0"``).  ``None`` uses the CPU.
    patch_size : tuple or None, optional
        Override the patch size stored in the checkpoint.  Must include the
        channel dimension, e.g. ``(256, 256, 1)`` for 2-D data.
    name : str, optional
        Job name used in the output directory structure.
    run_id : int, optional
        Run number (for repeated runs with the same name).

    Returns
    -------
    str
        Absolute path to the output directory.

    Raises
    ------
    FileNotFoundError
        If *model* or *input* do not exist.
    ValueError
        If the checkpoint does not contain an embedded configuration.

    Examples
    --------
    >>> from biapy import predict
    >>> predict(model="my_model.pth", input="images/", output="results/")
    """
    model = str(model)
    input_path = str(input)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # 1. Extract config from checkpoint
    checkpoint_cfg, _ = extract_config_from_checkpoint(model)

    # 2. Build inference config
    cfg_dict = _build_inference_config(
        checkpoint_cfg,
        checkpoint_path=model,
        input_path=input_path,
        patch_size=patch_size,
    )

    # 3. Write temporary YAML and run through standard BiaPy
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="biapy_predict_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)

        from biapy._biapy import BiaPy

        biapy = BiaPy(
            config=tmp_path,
            result_dir=os.path.abspath(output),
            name=name,
            run_id=run_id,
            gpu=gpu or "",
        )
        biapy.run_job()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return os.path.abspath(output)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def predict_from_cli(argv=None):
    """Parse CLI arguments for ``biapy predict`` and run inference."""
    parser = argparse.ArgumentParser(
        prog="biapy predict",
        description="Run inference with a trained BiaPy model. "
        "Only a checkpoint and input images are required.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a .pth checkpoint file (BiaPy v3.5.1+).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to directory containing input images.",
    )
    parser.add_argument(
        "--output",
        default="./biapy_predictions",
        help="Output directory (default: ./biapy_predictions).",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU id (e.g. '0'). Omit for CPU.",
    )
    parser.add_argument(
        "--patch-size",
        default=None,
        help="Override patch size, e.g. '256,256,1'.",
    )
    parser.add_argument(
        "--name",
        default="biapy_predict",
        help="Job name (default: biapy_predict).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help="Run number (default: 1).",
    )
    args = parser.parse_args(argv)

    patch_size = None
    if args.patch_size:
        patch_size = tuple(int(x) for x in args.patch_size.split(","))

    output_dir = predict(
        model=args.model,
        input=args.input,
        output=args.output,
        gpu=args.gpu,
        patch_size=patch_size,
        name=args.name,
        run_id=args.run_id,
    )
    print(f"\nResults saved to: {output_dir}")
