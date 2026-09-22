import math
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import signal

from utils.constants import ACC_COLS, LABEL_COLS


def get_patch_label(label_indices):
    """Return multiclass majority label for a patch (0,1,2,3)."""
    return int(np.bincount(label_indices).argmax())


def get_patch_label_binary(label_indices):
    """Return binary majority label for a patch (0=no FoG, 1=FoG)."""
    label = get_patch_label(label_indices)
    return 1 if label > 0 else 0


def get_patch_label_any_fog(label_indices):
    """Return 1 if any FOG frame in patch, else 0."""
    return int((label_indices > 0).any())


def compute_patch_metadata(label_indices, valid_mask=None):
    """
    Compute metadata for a patch with multiclass labels.

    Args:
        label_indices: Array of class indices for patch
        valid_mask: Optional boolean array marking valid timesteps

    Returns:
        dict with keys:
            - class_label: Multiclass label (0,1,2,3)
            - fog_percentage: Percentage of FoG frames (0.0-1.0)
            - transition_type: 'onset', 'offset', 'none', or 'mixed'
            - purity: Percentage of most dominant class
            - validity: Percentage of valid timesteps (0.0-1.0)
    """
    unique, counts = np.unique(label_indices, return_counts=True)
    percentages = counts / len(label_indices)

    # Calculate fog percentage (any non-zero class)
    fog_mask = unique > 0
    fog_percentage = percentages[fog_mask].sum() if fog_mask.any() else 0.0

    # Determine patch class label
    class_label = get_patch_label(label_indices)

    # Detect transition patterns
    transition_type = _detect_transition_type(label_indices)

    # Calculate purity (percentage of most dominant class)
    purity = float(percentages.max())

    # Calculate validity (percentage of valid timesteps)
    if valid_mask is not None:
        validity = float(valid_mask.mean())
    else:
        validity = 1.0  # Assume fully valid if no mask

    return {
        "class_label": class_label,
        "fog_percentage": float(fog_percentage),
        "transition_type": transition_type,
        "purity": purity,
        "validity": validity,
    }


def compute_patch_metadata_binary(label_indices, valid_mask=None):
    """
    Compute metadata for a patch with binary labels.

    Args:
        label_indices: Array of class indices for patch
        valid_mask: Optional boolean array marking valid timesteps

    Returns:
        dict with keys:
            - class_label: Binary label (0=no FoG, 1=FoG)
            - fog_percentage: Percentage of FoG frames (0.0-1.0)
            - transition_type: 'onset', 'offset', 'none', or 'mixed'
            - purity: Percentage of most dominant class
            - validity: Percentage of valid timesteps (0.0-1.0)
    """
    unique, counts = np.unique(label_indices, return_counts=True)
    percentages = counts / len(label_indices)

    # Calculate fog percentage (any non-zero class)
    fog_mask = unique > 0
    fog_percentage = percentages[fog_mask].sum() if fog_mask.any() else 0.0

    # Determine patch class label (binary)
    class_label = get_patch_label_binary(label_indices)

    # Detect transition patterns
    transition_type = _detect_transition_type(label_indices)

    # Calculate purity (percentage of most dominant class)
    purity = float(percentages.max())

    # Calculate validity (percentage of valid timesteps)
    if valid_mask is not None:
        validity = float(valid_mask.mean())
    else:
        validity = 1.0  # Assume fully valid if no mask

    return {
        "class_label": class_label,
        "fog_percentage": float(fog_percentage),
        "transition_type": transition_type,
        "purity": purity,
        "validity": validity,
    }


def _detect_transition_type(label_indices):
    """
    Detect if patch contains FoG transitions.

    Returns:
        'onset': 0->1 transition (FoG start)
        'offset': 1->0 transition (FoG end)
        'mixed': Multiple transitions
        'none': No transitions
    """
    # Find where labels change
    changes = np.diff(label_indices != 0)  # Boolean changes in FoG state

    if not changes.any():
        return "none"

    # Count transitions
    num_changes = changes.sum()

    if num_changes > 2:
        return "mixed"

    # Check type of transition
    first_change_idx = np.where(changes)[0][0]

    # Check if it's onset (0->1) or offset (1->0)
    if label_indices[first_change_idx] == 0 and label_indices[first_change_idx + 1] > 0:
        return "onset"
    elif (
        label_indices[first_change_idx] > 0 and label_indices[first_change_idx + 1] == 0
    ):
        return "offset"
    else:
        return "mixed"


def metadata2dict(metadata_df_path: str) -> Dict[str, List[str]]:
    important_cols = ["Id", "Subject"]
    metadata = pd.read_csv(metadata_df_path)
    metadata = metadata[important_cols]
    metadata.set_index("Id", inplace=True)
    metadata_dict = metadata.to_dict()
    return metadata_dict


def _ohe_to_indices(ohe_array: np.ndarray) -> np.ndarray:
    """Convert OHE array to class indices. Background (all zeros) -> 0, else argmax + 1."""
    is_background = np.all(ohe_array == 0, axis=-1)
    indices = np.argmax(ohe_array, axis=-1) + 1
    indices[is_background] = 0
    return indices.astype(np.int32)


def get_blocks(
    session_data: pd.DataFrame,
    block_len: int,
    stride_len: int,
    fog_stride_len: int | None = None,
):
    """
    Get blocks from a session data frame.

    Handles both labeled and unlabeled data. Labels are returned as class indices (not OHE).

    When fog_stride_len is provided and labels are available, uses adaptive stride:
    patches that contain any FOG frame advance by fog_stride_len (denser), others by
    stride_len. fog_stride_len has no effect on unlabeled sessions.

    Args:
        session_data: pd.DataFrame
        block_len: int
        stride_len: int
        fog_stride_len: optional denser stride for FOG-containing patches
    Returns:
        session_blocks: dict with keys 'accs', 'labels' (if available as indices), 'valid'
    """
    session_accs = session_data[ACC_COLS].values

    # Check if label columns exist (labeled vs unlabeled data)
    has_labels = all(col in session_data.columns for col in LABEL_COLS)
    if has_labels:
        # Convert OHE labels to indices immediately
        session_labels_ohe = session_data[LABEL_COLS].values
        session_labels = _ohe_to_indices(session_labels_ohe)
    else:
        session_labels = None

    # Extract metadata masks if available
    if "Valid" in session_data.columns:
        valid = session_data["Valid"].values
    else:
        valid = np.ones(len(session_data), dtype=bool)

    seq_length = session_data.shape[0]
    session_blocks = {"accs": [], "labels": [], "valid": [], "start_frames": []}

    use_adaptive = fog_stride_len is not None and has_labels

    if use_adaptive:
        # Adaptive stride: denser sampling over FOG regions
        current_start = 0
        while current_start < seq_length:
            end_idx = min(current_start + block_len, seq_length)
            start_idx = current_start

            if end_idx - start_idx < block_len:
                acc_block = np.zeros((block_len, session_accs.shape[1]), dtype=np.float32)
                acc_block[: end_idx - start_idx] = session_accs[start_idx:end_idx]
                label_block = np.zeros(block_len, dtype=np.int32)
                label_block[: end_idx - start_idx] = session_labels[start_idx:end_idx]
                valid_block = np.zeros(block_len, dtype=bool)
                valid_block[: end_idx - start_idx] = valid[start_idx:end_idx]
            else:
                acc_block = session_accs[start_idx:end_idx]
                label_block = session_labels[start_idx:end_idx].copy()
                valid_block = valid[start_idx:end_idx]

            session_blocks["accs"].append(acc_block)
            session_blocks["labels"].append(label_block)
            session_blocks["valid"].append(valid_block)
            session_blocks["start_frames"].append(start_idx)

            # Advance by fog_stride_len if patch has any FOG, else stride_len
            has_fog = (label_block > 0).any()
            current_start += fog_stride_len if has_fog else stride_len
    else:
        # Fixed stride (original behaviour)
        num_blocks = max(1, math.ceil((seq_length - block_len) / stride_len) + 1)
        for i in range(num_blocks):
            start_idx = i * stride_len
            end_idx = min(start_idx + block_len, seq_length)

            if end_idx - start_idx < block_len:
                acc_block = np.zeros((block_len, session_accs.shape[1]), dtype=np.float32)
                acc_block[: end_idx - start_idx] = session_accs[start_idx:end_idx]

                if has_labels:
                    label_block = np.zeros(block_len, dtype=np.int32)
                    label_block[: end_idx - start_idx] = session_labels[start_idx:end_idx]
                else:
                    label_block = None

                valid_block = np.zeros(block_len, dtype=bool)
                valid_block[: end_idx - start_idx] = valid[start_idx:end_idx]
            else:
                acc_block = session_accs[start_idx:end_idx]
                label_block = (
                    session_labels[start_idx:end_idx].copy() if has_labels else None
                )
                valid_block = valid[start_idx:end_idx]

            session_blocks["accs"].append(acc_block)
            if has_labels:
                session_blocks["labels"].append(label_block)
            session_blocks["valid"].append(valid_block)
            session_blocks["start_frames"].append(start_idx)

    return session_blocks
