"""
Clinical metrics management for FoG detection evaluation.

Provides episode-level detection metrics, patient-specific analysis,
and clinically-relevant evaluation measures that bridge ML metrics
to real-world clinical utility.
"""

import logging
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_curve, auc

logger = logging.getLogger(__name__)


class EpisodeDetector:
    """
    Converts patch-level predictions to episode-level detections.
    
    Handles temporal smoothing, minimum episode duration constraints,
    and overlap calculation between predicted and ground truth episodes.
    """
    
    def __init__(self, 
                 min_episode_duration: float = 2.0,
                 confidence_threshold: float = 0.5,
                 temporal_smoothing: bool = True,
                 smoothing_window: int = 5):
        """
        Initialize episode detector.
        
        Args:
            min_episode_duration: Minimum episode duration in seconds
            confidence_threshold: Threshold for positive predictions
            temporal_smoothing: Apply temporal smoothing to predictions
            smoothing_window: Window size for smoothing (patches)
        """
        self.min_episode_duration = min_episode_duration
        self.confidence_threshold = confidence_threshold
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window = smoothing_window
        
    def detect_episodes(self, 
                       probas: np.ndarray, 
                       timestamps: Optional[np.ndarray] = None,
                       patch_duration: float = 1.0) -> List[Tuple[int, int, float]]:
        """
        Detect FoG episodes from patch-level probabilities.
        
        Args:
            probas: Patch-level probabilities (N_patches, N_classes)
            timestamps: Optional timestamps for each patch
            patch_duration: Duration of each patch in seconds
            
        Returns:
            List of (start_idx, end_idx, confidence) tuples for detected episodes
        """
        if probas.ndim == 1:
            probas = probas.reshape(-1, 1)
            
        # Get maximum probability across FoG classes (excluding background)
        if probas.shape[1] > 1:
            fog_probas = np.max(probas[:, 1:], axis=1)  # Exclude background class
        else:
            fog_probas = probas[:, 0]
            
        # Apply temporal smoothing if requested
        if self.temporal_smoothing and len(fog_probas) > self.smoothing_window:
            fog_probas = self._smooth_predictions(fog_probas)
            
        # Find regions above threshold
        above_threshold = fog_probas >= self.confidence_threshold
        
        # Find episode boundaries
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(above_threshold)]])
            
        # Filter episodes by minimum duration
        episodes = []
        min_duration_patches = max(1, int(self.min_episode_duration / patch_duration))
        
        for start, end in zip(starts, ends):
            if end - start >= min_duration_patches:
                episode_confidence = np.mean(fog_probas[start:end])
                episodes.append((start, end, episode_confidence))
                
        return episodes
        
    def _smooth_predictions(self, probas: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to predictions."""
        # Simple moving average smoothing
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(probas, size=self.smoothing_window, mode='nearest')
        
    def calculate_episode_overlap(self, 
                                 pred_episodes: List[Tuple[int, int, float]], 
                                 true_episodes: List[Tuple[int, int]]) -> Dict:
        """
        Calculate overlap metrics between predicted and true episodes.
        
        Args:
            pred_episodes: List of (start, end, confidence) for predictions
            true_episodes: List of (start, end) for ground truth
            
        Returns:
            Dict with overlap metrics
        """
        if not pred_episodes and not true_episodes:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0, 
                   "n_predicted": 0, "n_true": 0, "n_matched_pred": 0, "n_matched_true": 0}
        if not pred_episodes:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0,
                   "n_predicted": 0, "n_true": len(true_episodes), "n_matched_pred": 0, "n_matched_true": 0}
        if not true_episodes:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0,
                   "n_predicted": len(pred_episodes), "n_true": 0, "n_matched_pred": 0, "n_matched_true": 0}
            
        # Calculate IoU for each predicted episode
        matched_predictions = 0
        total_overlap = 0.0
        
        for pred_start, pred_end, _ in pred_episodes:
            best_iou = 0.0
            for true_start, true_end in true_episodes:
                # Calculate IoU
                intersection = max(0, min(pred_end, true_end) - max(pred_start, true_start))
                pred_duration = pred_end - pred_start
                true_duration = true_end - true_start
                union = pred_duration + true_duration - intersection
                iou = intersection / union if union > 0 else 0.0
                best_iou = max(best_iou, iou)
                
            total_overlap += best_iou
            if best_iou > 0.1:  # Consider matched if IoU > 0.1
                matched_predictions += 1
                
        # Calculate recall (how many true episodes were detected)
        matched_true_episodes = 0
        for true_start, true_end in true_episodes:
            for pred_start, pred_end, _ in pred_episodes:
                intersection = max(0, min(pred_end, true_end) - max(pred_start, true_start))
                pred_duration = pred_end - pred_start
                true_duration = true_end - true_start
                union = pred_duration + true_duration - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou > 0.1:
                    matched_true_episodes += 1
                    break
                    
        precision = matched_predictions / len(pred_episodes) if pred_episodes else 0.0
        recall = matched_true_episodes / len(true_episodes) if true_episodes else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = total_overlap / len(pred_episodes) if pred_episodes else 0.0
        
        return {
            "precision": precision,
            "recall": recall, 
            "f1": f1,
            "iou": mean_iou,
            "n_predicted": len(pred_episodes),
            "n_true": len(true_episodes),
            "n_matched_pred": matched_predictions,
            "n_matched_true": matched_true_episodes
        }


class ClinicalMetricsManager:
    """
    Comprehensive clinical metrics manager for FoG detection evaluation.
    
    Provides patient-level analysis, episode detection metrics, time-to-detection
    calculations, and clinical significance testing.
    """
    
    def __init__(self, 
                 num_classes: int,
                 patch_duration: float = 1.0,
                 episode_detector_config: Optional[Dict] = None,
                 clinical_thresholds: Optional[Dict] = None):
        """
        Initialize clinical metrics manager.
        
        Args:
            num_classes: Total number of classes (including background)
            patch_duration: Duration of each patch in seconds
            episode_detector_config: Configuration for episode detector
            clinical_thresholds: Custom clinical thresholds for acceptability assessment
        """
        self.num_classes = num_classes
        self.num_event_classes = num_classes - 1  # Event classes exclude background
        self.patch_duration = patch_duration
        
        # Setup episode detector
        detector_config = episode_detector_config or {}
        self.episode_detector = EpisodeDetector(**detector_config)
        
        # Clinical thresholds for different metrics
        default_thresholds = {
            "sensitivity": 0.8,  # Clinical requirement for sensitivity
            "specificity": 0.9,  # Clinical requirement for specificity
            "ppv": 0.7,         # Positive predictive value threshold
            "time_to_detection": 5.0  # Maximum acceptable time to detection (seconds)
        }
        self.clinical_thresholds = clinical_thresholds or default_thresholds
        
    def compute_patient_clinical_metrics(self, 
                                       patient_data: Dict,
                                       session_metadata: Optional[Dict] = None) -> Dict:
        """
        Compute comprehensive clinical metrics for each patient.
        
        Args:
            patient_data: Patient-level predictions and labels from LoggingManager
            session_metadata: Optional metadata about sessions
            
        Returns:
            Dict with clinical metrics for each patient
        """
        patient_clinical_metrics = {}
        
        for patient_id in patient_data["labels"].keys():
            try:
                # Get patient data
                patient_labels = torch.cat(patient_data["labels"][patient_id]).cpu().numpy()
                patient_probas = torch.cat(patient_data["probas"][patient_id]).cpu().numpy()
                
                # Compute clinical metrics for this patient
                clinical_metrics = self._compute_single_patient_metrics(
                    patient_labels, patient_probas, patient_id
                )
                
                patient_clinical_metrics[patient_id] = clinical_metrics
                
            except Exception as e:
                logger.error(f"Error computing clinical metrics for patient {patient_id}: {e}")
                continue
                
        return patient_clinical_metrics
        
    def _compute_single_patient_metrics(self, 
                                       labels: np.ndarray, 
                                       probas: np.ndarray,
                                       patient_id: str) -> Dict:
        """Compute clinical metrics for a single patient."""
        
        # Handle different input shapes
        if labels.ndim == 2 and labels.shape[1] > 1:
            # Multi-class labels: check if any FoG class is active
            true_fog_binary = np.any(labels[:, 1:], axis=1).astype(int)
        else:
            # Binary labels
            true_fog_binary = labels.flatten()
            
        if probas.ndim == 2 and probas.shape[1] > 1:
            # Multi-class probabilities: max over FoG classes
            pred_fog_probas = np.max(probas[:, 1:], axis=1)
        else:
            # Binary probabilities
            pred_fog_probas = probas.flatten()
            
        # 1. Patch-level metrics
        patch_metrics = self._compute_patch_level_metrics(true_fog_binary, pred_fog_probas)
        
        # 2. Episode-level metrics
        episode_metrics = self._compute_episode_level_metrics(
            true_fog_binary, probas, patient_id
        )
        
        # 3. Time-to-detection metrics
        time_metrics = self._compute_time_to_detection_metrics(
            true_fog_binary, pred_fog_probas
        )
        
        # 4. Clinical thresholds assessment
        threshold_metrics = self._assess_clinical_thresholds(patch_metrics)
        
        return {
            "patient_id": patient_id,
            "patch_level": patch_metrics,
            "episode_level": episode_metrics, 
            "temporal": time_metrics,
            "clinical_assessment": threshold_metrics,
            "summary": {
                "clinically_acceptable": threshold_metrics["meets_clinical_requirements"],
                "primary_metric": patch_metrics["auc"],
                "episode_f1": episode_metrics["episode_f1"],
                "mean_time_to_detection": time_metrics["mean_time_to_detection"]
            }
        }
        
    def _compute_patch_level_metrics(self, y_true: np.ndarray, y_probas: np.ndarray) -> Dict:
        """Compute patch-level clinical metrics."""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, confusion_matrix,
            precision_recall_curve, roc_curve
        )
        
        # Basic metrics
        try:
            # Check for NaN values in probabilities and labels
            if np.any(np.isnan(y_probas)) or np.any(np.isnan(y_true)):
                logger.warning(f"NaN values detected in probabilities or labels")
                auc_score = 0.5
                ap_score = np.mean(y_true) if not np.isnan(np.mean(y_true)) else 0.5
            elif len(np.unique(y_true)) < 2:
                # Handle edge case where all labels are the same
                # logger.warning(f"All labels are the same class: {np.unique(y_true)}")
                auc_score = 0.5
                ap_score = np.mean(y_true)
            else:
                auc_score = roc_auc_score(y_true, y_probas)
                ap_score = average_precision_score(y_true, y_probas)
        except ValueError as e:
            # Handle other edge cases
            logger.warning(f"Error computing AUC/AP scores: {e}")
            auc_score = 0.5
            ap_score = np.mean(y_true) if not np.isnan(np.mean(y_true)) else 0.5
            
        # Find optimal threshold using Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_probas)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        # Compute metrics at optimal threshold
        y_pred = (y_probas >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle cases where confusion matrix is not 2x2
        if cm.size == 1:
            # Only one class present
            if y_true[0] == 0:  # All negatives
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:  # All positives
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            # Standard 2x2 case
            cm_flat = cm.ravel()
            if len(cm_flat) == 4:
                tn, fp, fn, tp = cm_flat
            else:
                # Fallback for unusual cases
                tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
                fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
                fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
                tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
        
        # Clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Ensure no NaN values in final metrics
        def safe_float(value):
            return 0.0 if np.isnan(value) or np.isinf(value) else float(value)
        
        return {
            "auc": safe_float(auc_score),
            "average_precision": safe_float(ap_score),
            "optimal_threshold": safe_float(optimal_threshold),
            "sensitivity": safe_float(sensitivity),
            "specificity": safe_float(specificity),
            "ppv": safe_float(ppv),  # Positive predictive value
            "npv": safe_float(npv),  # Negative predictive value
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "youden_j": safe_float(youden_j[optimal_idx] if len(youden_j) > optimal_idx else 0.0)
        }
        
    def _compute_episode_level_metrics(self, 
                                     y_true: np.ndarray, 
                                     probas: np.ndarray,
                                     patient_id: str) -> Dict:
        """Compute episode-level detection metrics."""
        
        # Convert ground truth to episodes
        true_episodes = self._extract_true_episodes(y_true)
        
        # Detect episodes from predictions
        pred_episodes = self.episode_detector.detect_episodes(probas)
        
        # Calculate episode overlap metrics
        overlap_metrics = self.episode_detector.calculate_episode_overlap(
            pred_episodes, true_episodes
        )
        
        # Additional episode-level metrics
        episode_sensitivity = overlap_metrics["recall"]
        episode_ppv = overlap_metrics["precision"]  
        episode_f1 = overlap_metrics["f1"]
        
        # False alarm rate (episodes per hour)
        total_duration_hours = len(y_true) * self.patch_duration / 3600
        false_alarm_rate = (overlap_metrics["n_predicted"] - overlap_metrics["n_matched_pred"]) / total_duration_hours if total_duration_hours > 0 else 0.0
        
        return {
            "n_true_episodes": len(true_episodes),
            "n_predicted_episodes": len(pred_episodes),
            "n_matched_episodes": overlap_metrics["n_matched_true"],
            "episode_sensitivity": episode_sensitivity,
            "episode_ppv": episode_ppv,
            "episode_f1": episode_f1,
            "mean_iou": overlap_metrics["iou"],
            "false_alarm_rate_per_hour": false_alarm_rate,
            "true_episodes": true_episodes,
            "predicted_episodes": pred_episodes
        }
        
    def _compute_time_to_detection_metrics(self, 
                                         y_true: np.ndarray, 
                                         y_probas: np.ndarray) -> Dict:
        """Compute time-to-detection metrics."""
        
        true_episodes = self._extract_true_episodes(y_true)
        detection_times = []
        
        for episode_start, episode_end in true_episodes:
            # Find first detection within this episode
            episode_probas = y_probas[episode_start:episode_end]
            detection_indices = np.where(episode_probas >= self.episode_detector.confidence_threshold)[0]
            
            if len(detection_indices) > 0:
                # Time from episode start to first detection
                time_to_detection = detection_indices[0] * self.patch_duration
                detection_times.append(time_to_detection)
            else:
                # Episode not detected - use max time
                detection_times.append(float('inf'))
                
        # Calculate statistics
        finite_times = [t for t in detection_times if np.isfinite(t)]
        
        if finite_times:
            mean_time = np.mean(finite_times)
            median_time = np.median(finite_times)
            std_time = np.std(finite_times)
        else:
            mean_time = median_time = std_time = float('inf')
            
        detection_rate = len(finite_times) / len(detection_times) if detection_times else 0.0
        
        return {
            "mean_time_to_detection": mean_time,
            "median_time_to_detection": median_time,
            "std_time_to_detection": std_time,
            "detection_rate": detection_rate,
            "n_episodes_detected": len(finite_times),
            "n_total_episodes": len(detection_times),
            "detection_times": finite_times
        }
        
    def _extract_true_episodes(self, y_true: np.ndarray) -> List[Tuple[int, int]]:
        """Extract episode boundaries from ground truth labels."""
        
        # Find transitions
        diff = np.diff(y_true.astype(int))
        starts = np.where(diff == 1)[0] + 1  
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if y_true[0] == 1:
            starts = np.concatenate([[0], starts])
        if y_true[-1] == 1:
            ends = np.concatenate([ends, [len(y_true)]])
            
        return list(zip(starts, ends))
        
    def _assess_clinical_thresholds(self, patch_metrics: Dict) -> Dict:
        """Assess whether performance meets clinical requirements."""
        
        meets_sensitivity = patch_metrics["sensitivity"] >= self.clinical_thresholds["sensitivity"]
        meets_specificity = patch_metrics["specificity"] >= self.clinical_thresholds["specificity"]
        meets_ppv = patch_metrics["ppv"] >= self.clinical_thresholds["ppv"]
        
        clinical_score = (
            int(meets_sensitivity) + 
            int(meets_specificity) + 
            int(meets_ppv)
        ) / 3.0
        
        return {
            "meets_sensitivity_req": meets_sensitivity,
            "meets_specificity_req": meets_specificity,
            "meets_ppv_req": meets_ppv,
            "meets_clinical_requirements": all([meets_sensitivity, meets_specificity, meets_ppv]),
            "clinical_score": clinical_score,
            "requirements": self.clinical_thresholds
        }
        
    def compute_cohort_statistics(self, patient_metrics: Dict) -> Dict:
        """Compute cohort-level statistics and clinical significance."""
        
        if not patient_metrics:
            return {}
            
        # Extract metrics across patients, filtering out NaN values
        def safe_extract(patient_metrics, path):
            values = []
            for p in patient_metrics.values():
                try:
                    keys = path.split(".")
                    value = p
                    for key in keys:
                        value = value[key]
                    if not (np.isnan(value) or np.isinf(value)):
                        values.append(float(value))
                except (KeyError, TypeError, ValueError):
                    continue
            return values if values else [0.0]
        
        aucs = safe_extract(patient_metrics, "patch_level.auc")
        sensitivities = safe_extract(patient_metrics, "patch_level.sensitivity")
        specificities = safe_extract(patient_metrics, "patch_level.specificity")
        episode_f1s = safe_extract(patient_metrics, "episode_level.episode_f1")
        
        # Clinical acceptability
        clinically_acceptable = [
            p["clinical_assessment"]["meets_clinical_requirements"] 
            for p in patient_metrics.values()
        ]
        
        # Statistical tests
        cohort_stats = {
            "n_patients": len(patient_metrics),
            "auc": {
                "mean": np.mean(aucs),
                "std": np.std(aucs),
                "median": np.median(aucs),
                "min": np.min(aucs),
                "max": np.max(aucs),
                "ci_95": self._compute_confidence_interval(aucs)
            },
            "sensitivity": {
                "mean": np.mean(sensitivities),
                "std": np.std(sensitivities),
                "median": np.median(sensitivities),
                "ci_95": self._compute_confidence_interval(sensitivities),
                "patients_above_threshold": np.sum([s >= self.clinical_thresholds["sensitivity"] for s in sensitivities])
            },
            "specificity": {
                "mean": np.mean(specificities),
                "std": np.std(specificities),
                "median": np.median(specificities),
                "ci_95": self._compute_confidence_interval(specificities),
                "patients_above_threshold": np.sum([s >= self.clinical_thresholds["specificity"] for s in specificities])
            },
            "episode_f1": {
                "mean": np.mean(episode_f1s),
                "std": np.std(episode_f1s),
                "median": np.median(episode_f1s),
                "ci_95": self._compute_confidence_interval(episode_f1s)
            },
            "clinical_acceptability": {
                "n_acceptable": np.sum(clinically_acceptable),
                "percentage_acceptable": np.mean(clinically_acceptable) * 100,
                "acceptable_patients": [pid for pid, p in patient_metrics.items() 
                                      if p["clinical_assessment"]["meets_clinical_requirements"]]
            }
        }
        
        return cohort_stats
        
    def _compute_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for a list of values."""
        if not values or len(values) < 2:
            mean_val = np.mean(values) if values else 0.0
            return (mean_val, mean_val)
            
        # Filter out NaN/inf values
        clean_values = [v for v in values if np.isfinite(v)]
        if not clean_values or len(clean_values) < 2:
            mean_val = np.mean(clean_values) if clean_values else 0.0
            return (mean_val, mean_val)
            
        try:
            mean = np.mean(clean_values)
            sem = stats.sem(clean_values)
            if np.isfinite(sem) and sem > 0:
                ci_range = sem * stats.t.ppf((1 + confidence) / 2, len(clean_values) - 1)
                return (mean - ci_range, mean + ci_range)
            else:
                return (mean, mean)
        except Exception:
            mean_val = np.mean(clean_values) if clean_values else 0.0
            return (mean_val, mean_val)
        
    def generate_clinical_report(self, 
                               patient_metrics: Dict, 
                               cohort_stats: Dict,
                               model_name: str = "FoG Detection Model") -> str:
        """Generate a comprehensive clinical evaluation report."""
        
        report_lines = [
            f"# Clinical Evaluation Report: {model_name}",
            f"Generated for {cohort_stats['n_patients']} patients\n",
            
            "## Executive Summary",
            f"- **Overall Performance**: AUC = {cohort_stats['auc']['mean']:.3f} ± {cohort_stats['auc']['std']:.3f}",
            f"- **Clinical Acceptability**: {cohort_stats['clinical_acceptability']['percentage_acceptable']:.1f}% of patients meet clinical requirements",
            f"- **Sensitivity**: {cohort_stats['sensitivity']['mean']:.3f} (target: ≥{self.clinical_thresholds['sensitivity']})",
            f"- **Specificity**: {cohort_stats['specificity']['mean']:.3f} (target: ≥{self.clinical_thresholds['specificity']})",
            f"- **Episode-level F1**: {cohort_stats['episode_f1']['mean']:.3f}\n",
            
            "## Patient-Level Analysis",
            f"**AUC Distribution:**",
            f"- Mean: {cohort_stats['auc']['mean']:.3f} (95% CI: {cohort_stats['auc']['ci_95'][0]:.3f}-{cohort_stats['auc']['ci_95'][1]:.3f})",
            f"- Range: {cohort_stats['auc']['min']:.3f} - {cohort_stats['auc']['max']:.3f}",
            f"- Median: {cohort_stats['auc']['median']:.3f}\n",
            
            "## Clinical Requirements Assessment",
            f"**Sensitivity Threshold (≥{self.clinical_thresholds['sensitivity']}):**",
            f"- Patients meeting requirement: {cohort_stats['sensitivity']['patients_above_threshold']}/{cohort_stats['n_patients']}",
            f"- Cohort mean: {cohort_stats['sensitivity']['mean']:.3f} ± {cohort_stats['sensitivity']['std']:.3f}\n",
            
            f"**Specificity Threshold (≥{self.clinical_thresholds['specificity']}):**", 
            f"- Patients meeting requirement: {cohort_stats['specificity']['patients_above_threshold']}/{cohort_stats['n_patients']}",
            f"- Cohort mean: {cohort_stats['specificity']['mean']:.3f} ± {cohort_stats['specificity']['std']:.3f}\n",
            
            "## Episode-Level Performance",
            f"- Mean episode-level F1: {cohort_stats['episode_f1']['mean']:.3f} ± {cohort_stats['episode_f1']['std']:.3f}",
            f"- Median episode-level F1: {cohort_stats['episode_f1']['median']:.3f}\n",
            
            "## Clinical Deployment Readiness",
        ]
        
        if cohort_stats['clinical_acceptability']['percentage_acceptable'] >= 80:
            report_lines.append("✅ **READY FOR CLINICAL VALIDATION** - >80% of patients meet clinical requirements")
        elif cohort_stats['clinical_acceptability']['percentage_acceptable'] >= 60:
            report_lines.append("⚠️ **REQUIRES IMPROVEMENT** - 60-80% of patients meet clinical requirements")
        else:
            report_lines.append("❌ **NOT READY FOR CLINICAL USE** - <60% of patients meet clinical requirements")
            
        report_lines.extend([
            f"\n**Clinically Acceptable Patients ({len(cohort_stats['clinical_acceptability']['acceptable_patients'])}):**",
            ", ".join(cohort_stats['clinical_acceptability']['acceptable_patients'][:10]) + 
            ("..." if len(cohort_stats['clinical_acceptability']['acceptable_patients']) > 10 else ""),
            "\n---",
            "Report generated by Clinical Metrics Manager"
        ])
        
        return "\n".join(report_lines)

    def compute_clinical_metrics(self, session_data: Dict, classes: List[str], sess2patient: Optional[Dict] = None) -> Optional[Dict]:
        """
        Compute clinical metrics from session-level data with proper patient-level aggregation.

        Args:
            session_data: Session-level predictions and labels
            classes: List of class names
            sess2patient: Optional session-to-patient mapping for true patient-level aggregation

        Returns:
            Dict with patient-level clinical metrics or None if disabled
        """
        try:
            # Convert session data to patient data format expected by compute_patient_clinical_metrics
            patient_data = {"labels": defaultdict(list), "probas": defaultdict(list)}

            # Group sessions by patient using proper mapping
            for session_id, session_content in session_data.items():
                # Get actual patient ID from mapping, or fallback to session ID
                if sess2patient and session_id in sess2patient:
                    patient_id = sess2patient[session_id]
                else:
                    # Fallback: use session_id as patient_id for backward compatibility
                    patient_id = f"session_{session_id}"
                
                # Get labels and probabilities
                if "labels" in session_content and "probas" in session_content:
                    labels = session_content["labels"]
                    probas = session_content["probas"]
                    
                    # Convert to tensors if needed
                    if not isinstance(labels, torch.Tensor):
                        labels = torch.tensor(labels)
                    if not isinstance(probas, torch.Tensor):
                        probas = torch.tensor(probas)
                    
                    patient_data["labels"][patient_id].append(labels)
                    patient_data["probas"][patient_id].append(probas)
            
            if not patient_data["labels"]:
                return None
                
            # Compute clinical metrics using the existing method
            return self.compute_patient_clinical_metrics(patient_data)
            
        except Exception as e:
            logger.error(f"Error computing clinical metrics: {e}")
            return None