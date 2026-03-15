"""
Simplified inference API for BiaPy.

Provides a ``predict()`` function and CLI entry point that only require a
checkpoint path and an input image directory — no YAML configuration file
needed.  The full configuration is extracted from the checkpoint (BiaPy
v3.5.1+ checkpoints embed it) and a temporary config is generated
automatically.
"""

import glob
import os
import argparse
import shutil
import tempfile
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
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


def _collect_image_paths(input_path: str) -> List[str]:
    """
    Resolve *input_path* to a list of absolute image file paths.

    *input_path* may be:
    - a directory  → all files directly inside it
    - a single file path  → that one file
    - a glob pattern (contains ``*`` or ``?``) → matching files
    """
    if os.path.isdir(input_path):
        return []  # directory mode — no symlinking needed
    if os.path.isfile(input_path):
        return [os.path.abspath(input_path)]
    # Try as a glob pattern
    matches = sorted(glob.glob(input_path))
    files = [os.path.abspath(m) for m in matches if os.path.isfile(m)]
    if not files:
        raise FileNotFoundError(
            f"No images found for input: {input_path!r}. "
            "Provide a directory, a file path, or a glob pattern (e.g. 'images/*.tif')."
        )
    return files


@contextmanager
def _input_as_directory(input_path: str):
    """
    Context manager that yields a directory path suitable for BiaPy.

    If *input_path* is already a directory, yields it directly.
    Otherwise (single file or glob), creates a temporary directory
    containing symlinks to the resolved files, and cleans it up on exit.
    """
    files = _collect_image_paths(input_path)
    if not files:
        # Already a directory
        yield os.path.abspath(input_path)
        return

    tmp_dir = tempfile.mkdtemp(prefix="biapy_input_")
    try:
        for f in files:
            link = os.path.join(tmp_dir, os.path.basename(f))
            # Handle duplicate basenames by appending a suffix
            if os.path.exists(link):
                stem, ext = os.path.splitext(os.path.basename(f))
                counter = 1
                while os.path.exists(link):
                    link = os.path.join(tmp_dir, f"{stem}_{counter}{ext}")
                    counter += 1
            os.symlink(f, link)
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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

    # Inference-only overrides
    overrides = {
        "TRAIN": {"ENABLE": False},
        "TEST": {"ENABLE": True},
        "DATA": {
            "TEST": {
                "PATH": os.path.abspath(input_path),
                "LOAD_GT": False,
            },
        },
        "MODEL": {
            "LOAD_CHECKPOINT": True,
            # Config is already applied — don't re-extract from checkpoint
            "LOAD_MODEL_FROM_CHECKPOINT": False,
        },
        "PATHS": {"CHECKPOINT_FILE": os.path.abspath(checkpoint_path)},
    }
    if patch_size is not None:
        overrides["DATA"]["PATCH_SIZE"] = list(patch_size)

    # Deep-merge overrides into the checkpoint config
    for section, values in overrides.items():
        cfg_dict.setdefault(section, {})
        for key, val in values.items():
            if isinstance(val, dict):
                cfg_dict[section].setdefault(key, {})
                cfg_dict[section][key].update(val)
            else:
                cfg_dict[section][key] = val

    return cfg_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    model: str,
    input_path: str,
    output: str = "./biapy_predictions",
    gpu: Optional[str] = None,
    patch_size: Optional[tuple] = None,
    name: str = "biapy_predict",
    run_id: int = 1,
) -> str:
    """
    Run inference with a trained BiaPy model.

    This is the simplified entry point: just provide a model checkpoint and
    input images.  The full configuration is read from the checkpoint
    (BiaPy v3.5.1+).

    Parameters
    ----------
    model : str
        Path to a ``.pth`` checkpoint file that contains an embedded
        BiaPy configuration.
    input_path : str
        Input images. Can be:

        - a **directory** of images (e.g. ``"images/"``)
        - a **single file** (e.g. ``"images/cell.tif"``)
        - a **glob pattern** (e.g. ``"images/*.tif"``,
          ``"data/**/*.png"``)
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
        If *model* or *input_path* do not exist / match no files.
    ValueError
        If the checkpoint does not contain an embedded configuration.

    Examples
    --------
    >>> from biapy import predict
    >>> predict(model="my_model.pth", input_path="images/", output="results/")
    >>> predict(model="my_model.pth", input_path="images/cell.tif")
    >>> predict(model="my_model.pth", input_path="images/*.tif")
    """
    model = str(model)
    input_path = str(input_path)

    # 1. Extract config from checkpoint
    checkpoint_cfg, _ = extract_config_from_checkpoint(model)

    # 2. Resolve input to a directory (symlinks single files / glob matches)
    with _input_as_directory(input_path) as resolved_dir:
        # 3. Build inference config
        cfg_dict = _build_inference_config(
            checkpoint_cfg,
            checkpoint_path=model,
            input_path=resolved_dir,
            patch_size=patch_size,
        )

        # 4. Write temporary YAML and run through standard BiaPy
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
            Path(tmp_path).unlink(missing_ok=True)

    return os.path.abspath(output)


def predict_from_array(
    model: str,
    images: Union[NDArray, Sequence[NDArray]],
    *,
    image_names: Optional[Sequence[str]] = None,
    output: Optional[str] = None,
    gpu: Optional[str] = None,
    patch_size: Optional[tuple] = None,
    name: str = "biapy_predict",
    run_id: int = 1,
) -> Union[NDArray, List[NDArray]]:
    """
    Run inference on in-memory numpy arrays.

    Writes arrays to temporary TIFFs, runs BiaPy prediction, and reads the
    results back as numpy arrays.  This is the recommended entry point when
    images are already loaded in memory (e.g. from a Fractal task or another
    pipeline).

    Parameters
    ----------
    model : str
        Path to a ``.pth`` checkpoint (BiaPy v3.5.1+).
    images : ndarray or sequence of ndarray
        One image or a list of images.  Each array should have shape
        ``(Y, X)`` , ``(Y, X, C)``, ``(Z, Y, X)``, or ``(Z, Y, X, C)``.
    image_names : sequence of str, optional
        Base filenames for the temporary TIFFs (without extension).  If
        *None*, images are named ``image_000``, ``image_001``, etc.
    output : str or None, optional
        Directory for BiaPy output.  If *None* a temporary directory is
        used and cleaned up after the results are read back.
    gpu : str or None, optional
        GPU device id (e.g. ``"0"``).  *None* uses the CPU.
    patch_size : tuple or None, optional
        Override patch size from the checkpoint.
    name : str, optional
        Job name (default ``"biapy_predict"``).
    run_id : int, optional
        Run number.

    Returns
    -------
    ndarray or list of ndarray
        If a single array was passed, returns a single result array.
        If a list was passed, returns a list of result arrays in the same
        order.

    Examples
    --------
    >>> import numpy as np
    >>> from biapy import predict_from_array
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> result = predict_from_array(model="model.pth", images=img)
    """
    import tifffile

    single = isinstance(images, np.ndarray)
    if single:
        images = [images]

    n = len(images)
    if image_names is None:
        image_names = [f"image_{i:03d}" for i in range(n)]
    elif len(image_names) != n:
        raise ValueError(
            f"image_names length ({len(image_names)}) does not match "
            f"number of images ({n})."
        )

    # Determine whether we manage the output dir ourselves
    manage_output = output is None
    if manage_output:
        output_dir = tempfile.mkdtemp(prefix="biapy_out_")
    else:
        output_dir = os.path.abspath(output)

    tmp_input = tempfile.mkdtemp(prefix="biapy_in_")
    try:
        # Write arrays as TIFFs
        for arr, img_name in zip(images, image_names):
            stem = os.path.splitext(img_name)[0]
            tifffile.imwrite(os.path.join(tmp_input, f"{stem}.tif"), arr)

        # Run prediction via the existing path-based API
        predict(
            model=model,
            input_path=tmp_input,
            output=output_dir,
            gpu=gpu,
            patch_size=patch_size,
            name=name,
            run_id=run_id,
        )

        # Read results back
        job_id = f"{name}_{run_id}"
        per_image_dir = os.path.join(
            output_dir, name, "results", job_id, "per_image"
        )

        results: List[NDArray] = []
        for img_name in image_names:
            stem = os.path.splitext(img_name)[0]
            out_path = os.path.join(per_image_dir, f"{stem}.tif")
            try:
                results.append(tifffile.imread(out_path))
            except FileNotFoundError:
                # Fall back: look for any matching file
                matches = glob.glob(os.path.join(per_image_dir, f"{stem}.*"))
                if not matches:
                    raise FileNotFoundError(
                        f"No output found for {img_name!r} in {per_image_dir}"
                    )
                results.append(tifffile.imread(matches[0]))
    finally:
        shutil.rmtree(tmp_input, ignore_errors=True)
        if manage_output:
            shutil.rmtree(output_dir, ignore_errors=True)

    return results[0] if single else results


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
        help="Input images: a directory, a single file, or a glob pattern "
        "(e.g. 'images/*.tif').",
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
        input_path=args.input,
        output=args.output,
        gpu=args.gpu,
        patch_size=patch_size,
        name=args.name,
        run_id=args.run_id,
    )
    print(f"\nResults saved to: {output_dir}")
