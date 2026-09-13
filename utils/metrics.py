from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, Union

import logging
import numpy as np
import pandas as pd
import pingouin as pg
import torch

logger = logging.getLogger(__name__)


def map_sessions_to_patients(
    session_data_dict, sess2patient, stage: Literal["val", "test"], keys: List[str]
):
    patient_data_dict = defaultdict(lambda: {k: [] for k in keys})

    for sid in session_data_dict.keys():
        # Extract session ID from {patient_id}_{session_id} format if needed
        session_id = sid.split('_')[-1] if '_' in sid else sid
        
        if session_id not in sess2patient:
            logger.info(f"Session {session_id} (from {sid}) not found in patient data mapping")
            continue

        patient_id = sess2patient[session_id]
        for k in keys:
            patient_data_dict[patient_id][k].append(session_data_dict[sid][k])

    return patient_data_dict


def map_sessions_to_ptype(
    session_data_dict, sid2type, stage: Literal["val", "test"], keys: List[str]
):
    ptype_data_dict = defaultdict(lambda: {k: [] for k in keys})
    for session_id in session_data_dict.keys():
        # Get protocol type directly from sid2type mapping
        ptype = sid2type.get(session_id)

        if ptype is None:
            raise ValueError(f"[INFO] Session {session_id} not found in protocol type mapping")

        for k in keys:
            ptype_data_dict[ptype][k].append(session_data_dict[session_id][k])

    return ptype_data_dict


def map_sessions_to_subgroups(
    session_data_dict: Dict,
    session_to_group_mapping: Dict[str, str],
    keys: List[str],
    group_column_name: str = "subgroup"
) -> Dict:
    """
    Generic function to group session data by any metadata column.

    Args:
        session_data_dict: Dictionary of session data {session_id: {key: values}}
        session_to_group_mapping: Mapping from session_id to group value
        keys: List of data keys to aggregate (e.g., ["labels", "probas", "logits"])
        group_column_name: Name of the grouping column (for logging)

    Returns:
        Dictionary of grouped data {group_value: {key: [session_values]}}
    """
    subgroup_data_dict = defaultdict(lambda: {k: [] for k in keys})

    for session_id in session_data_dict.keys():
        # Get group value from mapping
        group_value = session_to_group_mapping.get(session_id)

        if group_value is None:
            logger.info(f"Session {session_id} not found in {group_column_name} mapping, skipping")
            continue

        # Aggregate data for this group
        for k in keys:
            subgroup_data_dict[group_value][k].append(session_data_dict[session_id][k])

    return subgroup_data_dict


def calculate_icc(
    predictions: List[float], ground_truth: List[float], form: str = "1-1"
) -> Tuple[float, float, float]:
    """
    Calculate intraclass correlation coefficient (ICC) between model predictions
    and ground truth measurements.

    Args:
        predictions: List of predictions from model
        ground_truth: List of gold standard measurements from expert review
        form: ICC form, default '1-1' for single measurement, absolute agreement
              Alternatively, '3-1' for fixed raters, consistency

    Returns:
        icc: ICC value (ranges from 0 to 1, where 1 is perfect correlation)
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
    """
    # Prepare data format for pingouin's ICC calculation
    data = []

    # Stack predictions and ground truth into required format
    for i in range(len(predictions)):
        # Add model prediction
        data.append({"patient": i, "rater": "model", "score": predictions[i]})
        # Add expert/ground truth
        data.append({"patient": i, "rater": "expert", "score": ground_truth[i]})

    # Calculate ICC using pingouin
    data = pd.DataFrame(data)
    icc_results = pg.intraclass_corr(
        data=data, targets="patient", raters="rater", ratings="score", nan_policy="omit"
    )

    # Extract results for the specified form
    result = icc_results.loc[icc_results["Type"] == form]

    if len(result) == 0:
        return np.nan, np.nan, np.nan

    icc = result["ICC"].values[0]
    ci_lower = result["CI95%"].values[0][0]
    ci_upper = result["CI95%"].values[0][1]

    return icc, ci_lower, ci_upper
