#!/usr/bin/env python3
"""
Interactive Dataset Inspector for FoG Detection

A Dash web application to interactively explore and visualize data samples
from the FoG dataset.

Usage:
1. Run the application on the server:
   python scripts/interactive_inspector.py --experiment classification/best_fixed

2. Access the dashboard:
   Option A - Direct network access (from any device on the same network):
     http://SERVER_IP:8052 (replace SERVER_IP with your server's IP address)

   Option B - SSH port forwarding (if firewall blocks direct access):
     ssh -L 8052:localhost:8052 your_user@your_server_address
     Then open: http://localhost:8052
"""

import argparse
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.datamodule.datamodule import FOGDataModule
from utils.config_loaders import load_pydantic_config
from utils.paths import normalize_data_paths
from data.datamodule.config import DataConfig
# TaskManager removed in refactoring - no longer needed
from model.transforms import (
    STFTTransform,
    MelSpectrogramTransform,
    WaveletTransform,
)
from model.augmentors import (
    SIGNAL_AUGMENTORS_MAP,
    SPECTRAL_AUGMENTORS_MAP,
)
from model.preprocessors import (
    PREPROCESSORS_MAP,
)

# --- Global Variables ---
DATA_MODULE: FOGDataModule = None
CFG: DictConfig = None
DEVICE: str = "cpu"
TRANSFORMS: dict = {}
CHANNEL_MAP = {0: "AccV", 1: "AccML", 2: "AccAP"}
SAMPLE_RATE = 100  # Hz, fixed for FOG datasets


def extract_patient_session_info(metadata_dict):
    """
    Extract patient and session information from metadata.

    Args:
        metadata_dict: Metadata dictionary from dataset sample

    Returns:
        dict: Contains patient_id, session_id, session_idx, global_idx
    """
    # Use the proper fields from metadata if available
    patient_id = metadata_dict.get('patient_id', 'Unknown')
    session_id = metadata_dict.get('session_id', 'Unknown')
    session_idx = metadata_dict.get('session_idx', 0)
    global_idx = metadata_dict.get('global_idx', 0)

    # Fallback: if patient_id or session_id are missing, try to parse from 'id' field
    if patient_id == 'Unknown' or session_id == 'Unknown':
        full_id = metadata_dict.get('id', 'Unknown')

        # Extract patient ID from session ID format: patient_id_session_id
        if '_' in full_id:
            parts = full_id.split('_')
            if len(parts) >= 2:
                if parts[0] == 'session' and len(parts) >= 3:
                    # Format: session_XXXXXX_YY -> patient: XXXXXX, session: YY
                    if patient_id == 'Unknown':
                        patient_id = parts[1]
                    if session_id == 'Unknown':
                        session_id = parts[2]
                else:
                    # Format: patient_session -> patient: patient, session: session
                    if patient_id == 'Unknown':
                        patient_id = parts[0]
                    if session_id == 'Unknown':
                        session_id = parts[1]

    return {
        'patient_id': patient_id,
        'session_id': session_id,
        'full_session_id': metadata_dict.get('id', 'Unknown'),
        'session_idx': session_idx,
        'global_idx': global_idx
    }


def search_dataset_samples(dataset, patient_id=None, session_id=None, session_index=None, max_results=10):
    """
    Search for samples in dataset based on patient ID, session ID, and/or session index.

    Args:
        dataset: FOGDataset instance
        patient_id (str, optional): Patient ID to match
        session_id (str, optional): Session ID to match
        session_index (int, optional): Index within the session (session_idx)
        max_results (int): Maximum number of results to return

    Returns:
        list: List of tuples (dataset_idx, metadata_dict) for matching samples
    """
    if DATA_MODULE is None:
        return []

    matches = []

    try:
        # Access the metadata DataFrame directly from the dataset
        # The dataset already has filtered metadata for the current split
        if hasattr(dataset, 'metadata_df') and dataset.metadata_df is not None:
            metadata_df = dataset.metadata_df

            # Create filter conditions
            conditions = []

            if patient_id:
                if 'patient_id' in metadata_df.columns:
                    conditions.append(metadata_df['patient_id'] == patient_id)
                else:
                    return []  # Column doesn't exist

            if session_id:
                if 'session_id' in metadata_df.columns:
                    conditions.append(metadata_df['session_id'] == session_id)
                else:
                    return []  # Column doesn't exist

            if session_index is not None:
                if 'session_idx' in metadata_df.columns:
                    conditions.append(metadata_df['session_idx'] == session_index)
                else:
                    return []  # Column doesn't exist

            if conditions:
                # Combine all conditions with AND
                combined_condition = conditions[0]
                for condition in conditions[1:]:
                    combined_condition = combined_condition & condition

                # Filter the dataframe
                filtered_df = metadata_df[combined_condition]

                # Convert to list of matches (dataset index, metadata)
                # Dataset uses iloc internally, so we need positional indices
                # Get the iloc positions of matching rows in metadata_df
                matching_mask = combined_condition.values
                matching_iloc_positions = np.where(matching_mask)[0]

                for dataset_idx in matching_iloc_positions[:max_results]:
                    row = metadata_df.iloc[dataset_idx]
                    metadata_dict = {
                        'patient_id': row.get('patient_id', 'Unknown'),
                        'session_id': row.get('session_id', 'Unknown'),
                        'session_idx': row.get('session_idx', 0),
                        'global_idx': row.get('global_idx', 0),
                        'protocol': row.get('protocol', 'Unknown')
                    }
                    matches.append((int(dataset_idx), metadata_dict))

    except Exception as e:
        print(f"Error in search_dataset_samples: {e}")
        import traceback
        traceback.print_exc()
        return []

    return matches


def create_augmentor(augmentor_name, force_apply=False):
    """
    Create an augmentor instance with default or forced configuration.

    Args:
        augmentor_name (str): Name of the augmentor class
        force_apply (bool): If True, set probability to 1.0 to force application

    Returns:
        nn.Module: Instantiated augmentor or None if not found
    """
    if not augmentor_name:
        return None

    # Check signal augmentors first
    if augmentor_name in SIGNAL_AUGMENTORS_MAP:
        augmentor_class = SIGNAL_AUGMENTORS_MAP[augmentor_name]
    elif augmentor_name in SPECTRAL_AUGMENTORS_MAP:
        augmentor_class = SPECTRAL_AUGMENTORS_MAP[augmentor_name]
    else:
        return None

    # Create default configuration for the augmentor
    default_config = {"probability": 1.0 if force_apply else 0.8}

    # Add specific default parameters for certain augmentors
    if augmentor_name == 'NoiseAugmentor':
        default_config.update({
            "noise_probability": 1.0 if force_apply else 0.5,
            "noise_level": 0.05,
            "impulse_probability": 1.0 if force_apply else 0.2,
            "impulse_level": 0.2
        })
    elif augmentor_name == 'TremorAugmentor':
        default_config.update({
            "tremor_freq_range": [4.0, 6.0],
            "tremor_amplitude_range": [0.1, 0.3],
            "tremor_duration_range": [0.3, 0.8]
        })
    elif augmentor_name == 'BradykinesiaAugmentor':
        default_config.update({
            "slowdown_range": [0.6, 0.9],
            "segment_fraction_range": [0.3, 0.7]
        })
    elif augmentor_name == 'FoGSynthesisAugmentor':
        default_config.update({
            "fog_duration_range": [1.0, 4.0],
            "fog_freq_range": [3.0, 8.0],
            "fog_amplitude_range": [0.2, 0.5]
        })
    elif augmentor_name == 'LRFlipAugmentor':
        default_config.update({
            "probability": 1.0 if force_apply else 0.5
        })
    elif augmentor_name == 'ResizeAugmentor':
        default_config.update({
            "stretch_std": 0.3
        })
    elif augmentor_name == 'MultiplicativeNoiseAugmentor':
        default_config.update({
            "amp_gn": 0.2,
            "amp_ch_gn": 0.1,
            "amp_gna": 0.05,
            "amp_ch_gna": 0.02
        })
    elif augmentor_name == 'SpectralNoiseAugmentor':
        default_config.update({
            "noise_level_range": [0.01, 0.05]
        })
    elif augmentor_name == 'FrequencyMaskingAugmentor':
        default_config.update({
            "max_masks": 2,
            "min_mask_fraction": 0.05,
            "max_mask_fraction": 0.2
        })
    elif augmentor_name == 'TimeMaskingAugmentor':
        default_config.update({
            "max_masks": 2,
            "min_mask_fraction": 0.05,
            "max_mask_fraction": 0.2
        })

    try:
        return augmentor_class(default_config)
    except Exception as e:
        print(f"Error creating augmentor {augmentor_name}: {e}")
        return None


def apply_augmentation(signal, augmentor, is_spectral=False):
    """
    Apply augmentation to signal data.

    Args:
        signal (torch.Tensor): Input signal tensor
        augmentor: Augmentor instance
        is_spectral (bool): Whether this is applied to spectral data

    Returns:
        torch.Tensor: Augmented signal
    """
    if augmentor is None:
        return signal

    try:
        # Apply augmentation
        with torch.no_grad():
            augmented = augmentor(signal)
        return augmented
    except Exception as e:
        print(f"Error applying augmentation: {e}")
        return signal


def create_augmentor_chain(augmentor_names, force_apply=False):
    """
    Create a chain of augmentors from a list of augmentor names.

    Args:
        augmentor_names (list): List of augmentor names in order of application
        force_apply (bool): If True, set probability to 1.0 to force application

    Returns:
        list: List of instantiated augmentors
    """
    if not augmentor_names:
        return []

    # Create individual augmentors in the specified order
    augmentor_list = []
    for name in augmentor_names:
        augmentor = create_augmentor(name, force_apply=force_apply)
        if augmentor is not None:
            augmentor_list.append(augmentor)

    return augmentor_list


def apply_augmentation_chain(signal, augmentor_list, is_spectral=False):
    """
    Apply a chain of augmentations to signal data in sequence.

    Args:
        signal (torch.Tensor): Input signal tensor
        augmentor_list (list): List of augmentor instances
        is_spectral (bool): Whether this is applied to spectral data

    Returns:
        torch.Tensor: Augmented signal
    """
    if not augmentor_list:
        return signal

    current_signal = signal
    for augmentor in augmentor_list:
        current_signal = apply_augmentation(current_signal, augmentor, is_spectral)

    return current_signal


def create_preprocessor(preprocessor_name, custom_params=None):
    """
    Create a single preprocessor instance with default or custom configuration.

    Args:
        preprocessor_name (str): Name of the preprocessor class
        custom_params (dict): Custom parameters to override defaults

    Returns:
        nn.Module: Instantiated preprocessor or None if not found
    """
    if not preprocessor_name:
        return None

    # Check if preprocessor exists in registry
    if preprocessor_name not in PREPROCESSORS_MAP:
        return None

    preprocessor_class = PREPROCESSORS_MAP[preprocessor_name]

    # Create default configuration for each preprocessor
    if preprocessor_name == 'DetrendPreprocessor':
        default_config = {"method": "linear"}
    elif preprocessor_name == 'BandpassFilterPreprocessor':
        default_config = {
            "low_freq": 0.5,
            "high_freq": 20.0,
            "order": 4,
            "sample_rate": 100
        }
    elif preprocessor_name == 'NormalizePreprocessor':
        default_config = {
            "method": "zscore",
            "per_channel": True,
            "epsilon": 1e-8
        }
    elif preprocessor_name == 'BaselineCorrectPreprocessor':
        default_config = {"method": "mean"}
    elif preprocessor_name == 'SmoothingPreprocessor':
        default_config = {
            "method": "gaussian",
            "kernel_size": 5,
            "sigma": 1.0
        }
    elif preprocessor_name == 'ResamplePreprocessor':
        default_config = {
            "target_rate": 100,
            "original_rate": 100  # No resampling by default
        }
    else:
        default_config = {}

    # Override with custom parameters if provided
    if custom_params:
        default_config.update(custom_params)

    try:
        return preprocessor_class(**default_config)
    except Exception as e:
        print(f"Error creating preprocessor {preprocessor_name}: {e}")
        return None


def create_preprocessor_chain(preprocessor_names):
    """
    Create a chain of preprocessors from a list of preprocessor names.

    Args:
        preprocessor_names (list): List of preprocessor names in order of application

    Returns:
        nn.Module: SignalPreprocessor chain or None if no valid preprocessors
    """
    if not preprocessor_names:
        return None

    from model.preprocessors import SignalPreprocessor

    # Create individual preprocessors in the specified order
    preprocessor_list = []
    for name in preprocessor_names:
        preprocessor = create_preprocessor(name)
        if preprocessor is not None:
            preprocessor_list.append(preprocessor)

    # Return chained preprocessor if we have any valid ones
    if preprocessor_list:
        return SignalPreprocessor(preprocessor_list)
    else:
        return None


def apply_preprocessing(signal, preprocessor):
    """
    Apply preprocessing to signal data.

    Args:
        signal (torch.Tensor): Input signal tensor
        preprocessor: Preprocessor instance

    Returns:
        torch.Tensor: Preprocessed signal
    """
    if preprocessor is None:
        return signal

    try:
        # Apply preprocessing
        with torch.no_grad():
            preprocessed = preprocessor(signal)
        return preprocessed
    except Exception as e:
        print(f"Error applying preprocessing: {e}")
        return signal


# --- Dash App Layout ---
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.CYBORG])

# --- UI Components ---
controls_card = dbc.Accordion([
    # Data Selection Section
    dbc.AccordionItem([
        dbc.Row([
            dbc.Col([
                html.Label("Split:", className="mb-1", style={'font-size': '12px'}),
                dcc.Dropdown(id='split-dropdown', options=[
                    {'label': 'Train', 'value': 'train'},
                    {'label': 'Val', 'value': 'val'},
                    {'label': 'Test', 'value': 'test'}
                ], value='val', style={'font-size': '12px'})
            ], width=6),
            dbc.Col([
                html.Label("Index:", className="mb-1", style={'font-size': '12px'}),
                dbc.Input(id='sample-index-input', type='number', value=0, min=0, size="sm")
            ], width=6)
        ])
    ], title="Data Selection", item_id="data-selection"),

    # Search Section
    dbc.AccordionItem([
        # Row 1: Patient and Session search
        dbc.Row([
            dbc.Col([
                html.Label("Patient:", className="mb-1", style={'font-size': '12px'}),
                dbc.Input(id='patient-id-input', type='text', placeholder='0489dc', size="sm")
            ], width=6),
            dbc.Col([
                html.Label("Session:", className="mb-1", style={'font-size': '12px'}),
                dbc.Input(id='session-id-input', type='text', placeholder='055a5be', size="sm")
            ], width=6)
        ], className="mb-2"),

        # Row 2: Session Index and Search button
        dbc.Row([
            dbc.Col([
                html.Label("S.Idx:", className="mb-1", style={'font-size': '12px'}),
                dbc.Input(id='session-index-input', type='number', placeholder='0', min=0, size="sm")
            ], width=6),
            dbc.Col([
                html.Label("", className="mb-1", style={'font-size': '12px'}),  # Empty label for alignment
                dbc.Button('Search', id='search-button', n_clicks=0, color="secondary", size="sm", className="w-100")
            ], width=6)
        ], className="mb-2"),

        html.Div(id='search-results', style={'color': '#ffc107', 'font-size': '11px', 'max-height': '60px', 'overflow-y': 'auto'})
    ], title="Search", item_id="search"),

    # Visualization Settings Section
    dbc.AccordionItem([
        # Transform and Channels
        dbc.Row([
            dbc.Col([
                html.Label("Transform:", className="mb-1", style={'font-size': '12px'}),
                dcc.Dropdown(id='transform-type-dropdown', options=[
                    {'label': 'Raw', 'value': 'raw'},
                    {'label': 'STFT', 'value': 'stft'},
                    {'label': 'Mel', 'value': 'mel_spectrogram'},
                    {'label': 'CWT', 'value': 'cwt'},
                ], value='raw', style={'font-size': '12px'})
            ], width=12)
        ], className="mb-2"),

        dbc.Row([
            dbc.Col([
                html.Label("Channels:", className="mb-1", style={'font-size': '12px'}),
                dbc.Checklist(id='channel-checklist', options=[
                    {'label': name, 'value': i} for i, name in CHANNEL_MAP.items()
                ], value=[0, 1, 2], inline=True, inputClassName="me-1", style={'font-size': '11px'})
            ], width=12)
        ])
    ], title="Visualization", item_id="visualization"),

    # Spectral Parameters Section
    dbc.AccordionItem([
        dbc.Row([
            dbc.Col([
                html.Label("N_FFT:", className="mb-1", style={'font-size': '12px'}),
                dbc.Input(id='n-fft-input', type='number', value=128, min=32, max=512, size="sm")
            ], width=6),
            dbc.Col([
                html.Label("Hop:", className="mb-1", style={'font-size': '12px'}),
                dbc.Input(id='hop-length-input', type='number', value=32, min=4, max=128, size="sm")
            ], width=6)
        ])
    ], title="Spectral Parameters", item_id="spectral-params"),

    # Preprocessors Section
    dbc.AccordionItem([
        html.Label("Preprocessors (applied in selection order):", className="mb-1", style={'font-size': '12px'}),
        dcc.Dropdown(id='preprocessor-dropdown', options=[
            {'label': 'Detrend', 'value': 'DetrendPreprocessor'},
            {'label': 'Bandpass Filter', 'value': 'BandpassFilterPreprocessor'},
            {'label': 'Normalize', 'value': 'NormalizePreprocessor'},
            {'label': 'Baseline Correct', 'value': 'BaselineCorrectPreprocessor'},
            {'label': 'Smoothing', 'value': 'SmoothingPreprocessor'},
        ], value=[], multi=True, style={'font-size': '12px'}, className="mb-2"),
        html.Div([
            html.Small("Tip: Use backspace to remove items, order matters!",
                      style={'color': '#6c757d', 'font-size': '10px'})
        ])
    ], title="Preprocessors", item_id="preprocessors"),

    # Augmentations Section
    dbc.AccordionItem([
        html.Label("Signal (applied in selection order):", className="mb-1", style={'font-size': '12px'}),
        dcc.Dropdown(id='signal-augmentation-dropdown', options=[
            {'label': 'Noise', 'value': 'NoiseAugmentor'},
            {'label': 'Temporal', 'value': 'TemporalAugmentor'},
            {'label': 'Amplitude', 'value': 'AmplitudeAugmentor'},
            {'label': 'Time Warp', 'value': 'TimeWarpingAugmentor'},
            {'label': 'Shift', 'value': 'RandomShiftAugmentor'},
            {'label': 'LR Flip', 'value': 'LRFlipAugmentor'},
            {'label': 'Resize', 'value': 'ResizeAugmentor'},
            {'label': 'Excise', 'value': 'ExciseAugmentor'},
            {'label': 'Mult Noise', 'value': 'MultiplicativeNoiseAugmentor'},
            {'label': 'Crop', 'value': 'CropAugmentor'},
            {'label': 'Tremor', 'value': 'TremorAugmentor'},
            {'label': 'Bradykinesia', 'value': 'BradykinesiaAugmentor'},
            {'label': 'FoG Synth', 'value': 'FoGSynthesisAugmentor'},
            {'label': 'Gait Asym', 'value': 'GaitAsymmetryAugmentor'},
            {'label': 'Device Orient', 'value': 'DeviceOrientationAugmentor'},
            {'label': 'Signal Drop', 'value': 'SignalDropoutAugmentor'},
        ], value=[], multi=True, style={'font-size': '12px'}, className="mb-2"),

        html.Label("Spectral (applied in selection order):", className="mb-1", style={'font-size': '12px'}),
        dcc.Dropdown(id='spectral-augmentation-dropdown', options=[
            {'label': 'Spec Noise', 'value': 'SpectralNoiseAugmentor'},
            {'label': 'Freq Mask', 'value': 'FrequencyMaskingAugmentor'},
            {'label': 'Time Mask', 'value': 'TimeMaskingAugmentor'},
            {'label': 'Spec Roll', 'value': 'SpectralRollAugmentor'},
            {'label': 'Spec Mixup', 'value': 'SpectralMixupAugmentor'},
            {'label': 'Spec CutMix', 'value': 'SpectralCutMixAugmentor'},
            {'label': 'Contrast Jit', 'value': 'ContrastJitterAugmentor'},
            {'label': 'Edge Emph', 'value': 'EdgeEmphasisAugmentor'},
        ], value=[], multi=True, style={'font-size': '12px'}, className="mb-2"),

        dbc.Checklist(id='augmentation-options', options=[
            {'label': 'Show Orig', 'value': 'show_original'},
            {'label': 'Force Apply', 'value': 'force_apply'},
        ], value=['show_original'], inline=True, inputClassName="me-1", style={'font-size': '11px'}),
        html.Div([
            html.Small("Tip: Use backspace to remove items, order matters for both signal and spectral!",
                      style={'color': '#6c757d', 'font-size': '10px'})
        ])
    ], title="Augmentations", item_id="augmentations"),

    # Display Options Section
    dbc.AccordionItem([
        dbc.Checklist(id='metadata-checklist', options=[
            {'label': 'Labels', 'value': 'labels'},
            {'label': 'Validity', 'value': 'validity'},
        ], value=['labels', 'validity'], inline=True, inputClassName="me-1", style={'font-size': '11px'})
    ], title="Display Options", item_id="display"),

], start_collapsed=False, always_open=True, className="mb-2")

# Create controls container with accordion + load button
controls_container = html.Div([
    controls_card,
    html.Div(className="mb-2"),  # Add some spacing
    dbc.Button('Load Sample', id='load-button', n_clicks=0, color="primary", size="md", className="w-100", style={'font-weight': 'bold'})
])

visualization_pane = dcc.Loading(
    id="loading-icon", 
    children=[
        html.Div(id='session-info-text', style={'color': 'white', 'padding': '10px', 'font-family': 'monospace', 'font-size': '14px', 'background-color': '#2c3e50', 'border-radius': '5px', 'margin-bottom': '10px'}),
        dcc.Graph(id='signal-plot', style={'height': '85vh'})
    ], 
    type="graph"
)

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(controls_container, width=12, lg=3),
            dbc.Col(visualization_pane, width=12, lg=9),
        ], className="p-3"),
    ],
    fluid=True,
)


# --- Callbacks ---
# Note: Spectral params visibility is handled by accordion collapse, not a dedicated callback

@app.callback(
    [Output('sample-index-input', 'value'),
     Output('search-results', 'children')],
    [Input('search-button', 'n_clicks')],
    [State('split-dropdown', 'value'),
     State('patient-id-input', 'value'),
     State('session-id-input', 'value'),
     State('session-index-input', 'value')]
)
def search_samples(n_clicks, split, patient_id, session_id, session_index):
    """Handle search functionality and update sample index accordingly."""
    if n_clicks == 0:
        return 0, ""

    if DATA_MODULE is None:
        return 0, "⚠️ Data module not loaded"

    # Get the current dataset
    dataset = getattr(DATA_MODULE, f"{split}_dataset")

    # Clean up input values
    patient_id = patient_id.strip() if patient_id and patient_id.strip() else None
    session_id = session_id.strip() if session_id and session_id.strip() else None

    # Validate that at least one search criterion is provided
    if not patient_id and not session_id and session_index is None:
        return 0, "❌ Please provide at least one search criterion"

    # Validate session_index
    if session_index is not None:
        if session_index < 0:
            return 0, "❌ Session index must be non-negative"
        try:
            session_index = int(session_index)
        except (ValueError, TypeError):
            return 0, "❌ Session index must be a valid integer"

    # Perform the search
    matches = search_dataset_samples(
        dataset=dataset,
        patient_id=patient_id,
        session_id=session_id,
        session_index=session_index,
        max_results=10
    )

    if not matches:
        search_criteria = []
        if patient_id:
            search_criteria.append(f"Patient: {patient_id}")
        if session_id:
            search_criteria.append(f"Session: {session_id}")
        if session_index is not None:
            search_criteria.append(f"Index: {session_index}")

        return 0, f"❌ No matches found for {', '.join(search_criteria)}"

    # Use the first match and update the sample index
    first_match_idx, first_match_metadata = matches[0]

    # Create results summary
    results_text = []
    results_text.append(f"✅ Found {len(matches)} match(es). Loading first result:")
    results_text.append(f"Patient: {first_match_metadata['patient_id']}, Session: {first_match_metadata['session_id']}")
    results_text.append(f"Session Index: {first_match_metadata['session_idx']}")

    if len(matches) > 1:
        results_text.append(f"📋 Additional matches found (use sample index to access):")
        for i, (match_idx, match_metadata) in enumerate(matches[1:6], 1):  # Show up to 5 additional matches
            results_text.append(f"  • Index {match_idx}: P:{match_metadata['patient_id']}, S:{match_metadata['session_id']}, SI:{match_metadata['session_idx']}")
        if len(matches) > 6:
            results_text.append(f"  • ... and {len(matches) - 6} more")

    return first_match_idx, html.Div([html.P(line) for line in results_text])

@app.callback(
    [Output('signal-plot', 'figure'),
     Output('session-info-text', 'children')],
    [
        Input('load-button', 'n_clicks'),
        Input('split-dropdown', 'value'),
        Input('sample-index-input', 'value')],
    [
        State('transform-type-dropdown', 'value'),
        State('channel-checklist', 'value'),
        State('metadata-checklist', 'value'),
        State('n-fft-input', 'value'),
        State('hop-length-input', 'value'),
        State('preprocessor-dropdown', 'value'),
        State('signal-augmentation-dropdown', 'value'),
        State('spectral-augmentation-dropdown', 'value'),
        State('augmentation-options', 'value'),
    ]
)
def update_visualization(n_clicks, split, sample_index, transform_type, channels, metadata_to_show, n_fft, hop_length, preprocessor, signal_aug, spectral_aug, aug_options):
    # Only show initial message when truly nothing has been done yet
    if n_clicks == 0 and split == 'val' and sample_index == 0:
        fig = go.Figure(layout={'title': 'Select parameters and click Load'})
        fig.update_layout(template="plotly_dark")
        return (fig, "Click Load Sample to display session information")

    # Validation checks
    if DATA_MODULE is None:
        fig = go.Figure(layout={'title': 'Error: Data module not loaded'})
        fig.update_layout(template="plotly_dark")
        return (fig, "Error: Data module not loaded")

    if not channels or split is None:
        fig = go.Figure(layout={'title': 'Error: No channels selected or split not specified'})
        fig.update_layout(template="plotly_dark")
        return (fig, "Error: Please select at least one channel and specify a split")

    if sample_index is None or sample_index < 0:
        fig = go.Figure(layout={'title': 'Error: Invalid sample index'})
        fig.update_layout(template="plotly_dark")
        return (fig, "Error: Sample index must be non-negative")

    # Default to 'val' if split is somehow None
    split = split or 'val'
    dataset = getattr(DATA_MODULE, f"{split}_dataset")

    if sample_index >= len(dataset):
        fig = go.Figure(layout={'title': f"Index {sample_index} out of bounds for {split} split."})
        fig.update_layout(template="plotly_dark")
        return (fig, f"Index {sample_index} out of bounds for {split} split")

    sample = dataset[sample_index]
    signal_tensor = sample.signal
    labels_tensor = sample.labels
    valid_mask_tensor = sample.valid_mask
    metadata_dict = sample.metadata
    session_id = metadata_dict.get('session_id', 'Unknown')

    # Extract patient and session info
    patient_session_info = extract_patient_session_info(metadata_dict)
    protocol_type = metadata_dict.get('protocol', 'Unknown')

    if signal_tensor.dim() == 2:
        signal = signal_tensor[channels, :].unsqueeze(0).to(DEVICE)
        channel_names = [CHANNEL_MAP[i] for i in channels]
    elif signal_tensor.dim() == 1:
        signal = signal_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)
        channel_names = ["Signal"]
        channels = [0]
    else:
        fig = go.Figure(layout={'title': f"Unsupported signal dimension: {signal_tensor.dim()}"})
        fig.update_layout(template="plotly_dark")
        return (fig, f"Unsupported signal dimension: {signal_tensor.dim()}")

    # Store original signal for comparison
    original_signal = signal.clone()

    # Create preprocessor chain once (reused for both original and augmented if needed)
    signal_preprocessor = None
    if preprocessor:  # preprocessor is now a list
        signal_preprocessor = create_preprocessor_chain(preprocessor)

    # Apply preprocessing first (before augmentations)
    preprocessed_signal = original_signal
    if signal_preprocessor is not None:
        preprocessed_signal = apply_preprocessing(original_signal, signal_preprocessor)

    # Apply signal augmentations (after preprocessing)
    force_apply = 'force_apply' in (aug_options or [])
    show_augmentation_original = 'show_original' in (aug_options or [])

    augmented_signal = preprocessed_signal
    if signal_aug:  # signal_aug is now a list
        signal_augmentors = create_augmentor_chain(signal_aug, force_apply=force_apply)
        if signal_augmentors:
            augmented_signal = apply_augmentation_chain(preprocessed_signal, signal_augmentors, is_spectral=False)

    # Final signal is the augmented one
    signal = augmented_signal

    # --- Figure Creation ---
    # Determine figure layout based only on augmentation settings (not preprocessing)
    has_preprocessing = bool(preprocessor)  # preprocessor is now a list
    has_augmentation = bool(signal_aug) or bool(spectral_aug)  # both are now lists
    show_comparison = show_augmentation_original and has_augmentation
    num_cols = 2 if show_comparison else 1

    # For spectrograms, allocate separate rows for each channel
    if transform_type != 'raw' and len(channels) > 1:
        num_channel_rows = len(channels)
        num_subplots = num_channel_rows + len(metadata_to_show)
    else:
        num_channel_rows = 1
        num_subplots = 1 + len(metadata_to_show)

    # Calculate row heights: allocate space for channels + metadata
    channel_height = 0.7 / num_channel_rows if num_channel_rows > 1 else 0.7
    channel_heights = [channel_height] * num_channel_rows
    metadata_heights = [0.15] * len(metadata_to_show)
    row_heights = channel_heights + metadata_heights

    # Create subplot titles
    if show_comparison:
        aug_type = "Signal" if signal_aug else "Spectral"
        if num_channel_rows > 1:
            # Multi-channel spectrogram titles
            channel_names = [CHANNEL_MAP[i] for i in channels]
            channel_titles = []
            for ch_name in channel_names:
                channel_titles.extend([
                    f"{ch_name} Original {transform_type.replace('_', ' ').title()}",
                    f"{ch_name} {aug_type} Augmented"
                ])
        else:
            channel_titles = [
                f"Original {transform_type.replace('_', ' ').title()}",
                f"{aug_type} Augmented {transform_type.replace('_', ' ').title()}"
            ]

        metadata_titles = [m.title() for m in metadata_to_show] + [""] * len(metadata_to_show)
        subplot_titles = channel_titles + metadata_titles

        fig = make_subplots(
            rows=num_subplots, cols=num_cols,
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
    else:
        aug_names = signal_aug or spectral_aug or []
        aug_suffix = f" ({', '.join(aug_names)})" if aug_names else ""

        if num_channel_rows > 1:
            # Multi-channel spectrogram titles
            channel_names = [CHANNEL_MAP[i] for i in channels]
            channel_titles = [f"{ch_name} {transform_type.replace('_', ' ').title()}{aug_suffix}"
                            for ch_name in channel_names]
        else:
            channel_titles = [f"{transform_type.replace('_', ' ').title()}{aug_suffix}"]

        metadata_titles = [m.title() for m in metadata_to_show]
        subplot_titles = channel_titles + metadata_titles

        fig = make_subplots(
            rows=num_subplots, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )
    
    # Create copyable session info text
    session_info_text = (f"Sample: {sample_index} ({split.upper()}) | "
                        f"Patient: {patient_session_info['patient_id']} | "
                        f"Session: {patient_session_info['session_id']} | "
                        f"Patch: {patient_session_info['session_idx']} (Global: {patient_session_info['global_idx']}) | "
                        f"Protocol: {protocol_type}")

    # Simplified plot title with protocol type
    plot_title = f"{transform_type.replace('_', ' ').title()} Visualization - Protocol: {protocol_type}"
    
    fig.update_layout(
        template="plotly_dark", 
        showlegend=True, 
        title_text=plot_title,
        margin=dict(t=60, b=50, l=20, r=20)
    )

    # Store transformed time dimensions for metadata synchronization
    transformed_time_steps = None

    # --- Main Plot (Signal or Transform) ---
    def plot_signal_data(signal_data, row, col, name_suffix=""):
        """Helper function to plot signal or transform data"""
        nonlocal transformed_time_steps
        nonlocal num_channel_rows

        if transform_type == 'raw':
            for i, ch_name in enumerate(channel_names):
                trace_name = f"{ch_name}{name_suffix}"
                fig.add_trace(go.Scatter(y=signal_data[0, i, :].cpu().numpy(),
                                       mode='lines', name=trace_name), row=row, col=col)
            fig.update_yaxes(title_text="Amplitude", row=row, col=col)
            # For raw signals, time dimension matches original
            if transformed_time_steps is None:
                transformed_time_steps = signal_data.shape[-1]
        else:
            # Apply transform and spectral augmentation
            # Validate and set defaults for spectral parameters
            actual_n_fft = n_fft or 128
            actual_hop_length = hop_length or 32

            # Recreate transform with user-specified parameters
            if transform_type == 'stft':
                sample_rate = SAMPLE_RATE
                transform = STFTTransform(
                    sample_rate=sample_rate,
                    n_fft=actual_n_fft,
                    hop_length=actual_hop_length,
                    win_length=min(actual_n_fft, actual_hop_length * 4),
                    device=DEVICE,
                    interpolate=True
                )
            elif transform_type == 'mel_spectrogram':
                sample_rate = SAMPLE_RATE
                transform = MelSpectrogramTransform(
                    sample_rate=sample_rate,
                    n_fft=actual_n_fft,
                    hop_length=actual_hop_length,
                    win_length=min(actual_n_fft, actual_hop_length * 4),
                    n_mels=80,
                    f_min=0.5,
                    f_max=20.0,
                    interpolate=True,
                    device=DEVICE
                )
            else:
                transform = TRANSFORMS[transform_type]

            with torch.no_grad():
                transformed_signal = transform(signal_data).squeeze(0)

                # Apply spectral augmentation if specified and this is the augmented version
                if spectral_aug and name_suffix:  # spectral_aug is now a list
                    spectral_augmentors = create_augmentor_chain(spectral_aug, force_apply=force_apply)
                    if spectral_augmentors:
                        # Add batch dimension for spectral augmentation
                        transformed_signal = transformed_signal.unsqueeze(0)
                        transformed_signal = apply_augmentation_chain(transformed_signal, spectral_augmentors, is_spectral=True)
                        transformed_signal = transformed_signal.squeeze(0)

                # Store the transformed time dimension for metadata synchronization
                if transformed_time_steps is None:
                    transformed_time_steps = transformed_signal.shape[-1]

            # Continue with existing plotting logic for transforms
            if len(transformed_signal.shape) == 3:  # [channels, freq, time]
                num_channels, num_freqs, num_times = transformed_signal.shape

                # Calculate frequency axis for cleaner labeling
                sample_rate = SAMPLE_RATE

                if transform_type == 'stft':
                    freq_axis = np.linspace(0, sample_rate/2, num_freqs)
                    y_title = "Frequency (Hz)"
                elif transform_type == 'mel_spectrogram':
                    freq_axis = np.linspace(0, sample_rate/2, num_freqs)  # Approximate
                    y_title = "Mel Frequency (Hz)"
                elif transform_type == 'cwt':
                    scales = np.linspace(1, num_freqs, num_freqs)
                    freq_axis = sample_rate / (2 * scales)
                    y_title = "Wavelet Frequency (Hz)"
                else:
                    freq_axis = np.arange(num_freqs)
                    y_title = "Frequency Bins"

                # Plot each channel in its own row for multi-channel spectrograms
                for ch_idx, ch_name in enumerate(channel_names):
                    ch_data = transformed_signal[ch_idx].cpu().numpy()  # [freq, time]

                    # Normalize each channel separately
                    ch_min, ch_max = ch_data.min(), ch_data.max()
                    p5, p95 = np.percentile(ch_data, [5, 95])

                    if abs(p95 - p5) < 1e-6 or p95 == p5:
                        zmin, zmax = ch_min, ch_max
                    else:
                        zmin, zmax = p5, p95

                    # Determine which row to use based on multi-channel layout
                    if num_channel_rows > 1:
                        target_row = row + ch_idx  # Each channel gets its own row
                    else:
                        target_row = row  # Single row for all channels (fallback)

                    fig.add_trace(go.Heatmap(
                        z=ch_data,
                        colorscale='Viridis',
                        name=f"{ch_name}{name_suffix}",
                        showscale=(ch_idx == 0),  # Only show colorbar for first channel
                        zmin=zmin,
                        zmax=zmax,
                        x=list(range(num_times)),
                        y=list(range(num_freqs))
                    ), row=target_row, col=col)

            else:  # [freq, time] - single channel
                ch_data = transformed_signal.cpu().numpy()
                num_freqs, num_times = ch_data.shape

                sample_rate = SAMPLE_RATE
                if transform_type == 'stft':
                    freq_axis = np.linspace(0, sample_rate/2, num_freqs)
                    y_title = "Frequency (Hz)"
                elif transform_type == 'mel_spectrogram':
                    freq_axis = np.linspace(0, sample_rate/2, num_freqs)
                    y_title = "Mel Frequency (Hz)"
                elif transform_type == 'cwt':
                    scales = np.linspace(1, num_freqs, num_freqs)
                    freq_axis = sample_rate / (2 * scales)
                    y_title = "Wavelet Frequency (Hz)"
                else:
                    freq_axis = np.arange(num_freqs)
                    y_title = "Frequency Bins"

                # Normalize for better contrast
                ch_min, ch_max = ch_data.min(), ch_data.max()
                p5, p95 = np.percentile(ch_data, [5, 95])

                if abs(p95 - p5) < 1e-6 or p95 == p5:
                    zmin, zmax = ch_min, ch_max
                else:
                    zmin, zmax = p5, p95

                fig.add_trace(go.Heatmap(
                    z=ch_data,
                    colorscale='Viridis',
                    name=f"{transform_type}{name_suffix}",
                    showscale=True,
                    zmin=zmin,
                    zmax=zmax,
                    x=list(range(num_times)),
                    y=list(range(num_freqs))
                ), row=row, col=col)

            # Update axis labels for each channel row
            if len(transformed_signal.shape) == 3 and num_channel_rows > 1:
                # Multi-channel spectrogram: update y-axis for each channel row
                for ch_idx in range(num_channel_rows):
                    target_row = row + ch_idx
                    fig.update_yaxes(title_text=y_title, row=target_row, col=col)
                    fig.update_xaxes(title_text="Time Steps", row=target_row, col=col)
            else:
                # Single channel or raw signal
                fig.update_yaxes(title_text=y_title, row=row, col=col)
                fig.update_xaxes(title_text="Time Steps", row=row, col=col)

    # Plot signals with appropriate labels based on what processing was applied
    if show_comparison:
        # Only show comparison for augmentations (not preprocessors)
        # Plot preprocessed signal (before augmentation) vs augmented signal
        plot_signal_data(preprocessed_signal, 1, 1, " (Original)")
        plot_signal_data(augmented_signal, 1, 2, " (Aug)")
    else:
        # Plot single signal with appropriate label
        if has_preprocessing and has_augmentation:
            signal_label = " (Preprocessed + Aug)"
        elif has_preprocessing:
            signal_label = " (Preprocessed)"
        elif has_augmentation:
            signal_label = " (Aug)"
        else:
            signal_label = ""
        plot_signal_data(signal, 1, 1, signal_label)

    # --- Metadata Subplots ---
    # Synchronize metadata with transformed signal dimensions
    synced_labels_tensor = labels_tensor
    synced_valid_mask_tensor = valid_mask_tensor

    if transformed_time_steps is not None and transformed_time_steps != labels_tensor.shape[0]:
        # Resize labels and validity mask to match transformed signal time dimension
        # Use nearest interpolation to preserve discrete label values
        synced_labels_tensor = torch.nn.functional.interpolate(
            labels_tensor.float().unsqueeze(0).unsqueeze(0),  # [1, 1, time]
            size=transformed_time_steps,
            mode='nearest'
        ).squeeze(0).squeeze(0).long()  # Back to [time]

        synced_valid_mask_tensor = torch.nn.functional.interpolate(
            valid_mask_tensor.float().unsqueeze(0).unsqueeze(0),  # [1, 1, time]
            size=transformed_time_steps,
            mode='nearest'
        ).squeeze(0).squeeze(0).bool()  # Back to [time]

    current_row = num_channel_rows + 1
    if 'labels' in metadata_to_show:
        labels_np = synced_labels_tensor.cpu().numpy().reshape(1, -1)

        # Define class names based on dataset configuration
        # Datasets no longer have .classes attribute - read from config instead
        classification_strategy = dataset.dataset_cfg.classification_strategy
        if classification_strategy == "binary_any_fog":
            class_names = ["NoFOG", "FOG"]
        elif classification_strategy == "multiclass_with_background":
            class_names = ["NoFOG", "StartHesitation", "Turn", "Walking"]
        else:
            # Fallback for unknown strategies - generate from actual labels
            num_classes = int(labels_np.max()) + 1
            class_names = [f"Class_{i}" for i in range(num_classes)]

        hover_text_labels = np.array([[class_names[int(i)] for i in row] for row in labels_np])

        # Plot labels on first column (and second if comparison)
        fig.add_trace(go.Heatmap(
            z=labels_np,
            hovertext=hover_text_labels,
            hoverinfo='text',
            colorscale='Plasma',
            showscale=False
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="Label", showticklabels=False, row=current_row, col=1)

        if show_comparison:
            fig.add_trace(go.Heatmap(
                z=labels_np,
                hovertext=hover_text_labels,
                hoverinfo='text',
                colorscale='Plasma',
                showscale=False
            ), row=current_row, col=2)
            fig.update_yaxes(title_text="Label", showticklabels=False, row=current_row, col=2)
        current_row += 1

    if 'validity' in metadata_to_show:
        valid_mask_np = synced_valid_mask_tensor.cpu().numpy().reshape(1, -1).astype(float)
        hover_text_validity = np.array([["Valid" if v == 1 else "Invalid" for v in row] for row in valid_mask_np])

        # Plot validity on first column (and second if comparison)
        fig.add_trace(go.Heatmap(
            z=valid_mask_np,
            hovertext=hover_text_validity,
            hoverinfo='text',
            colorscale=[[0, '#d73027'], [1, '#4575b4']],  # Red-Blue
            showscale=False,
            zmin=0, zmax=1
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="Validity", showticklabels=False, row=current_row, col=1)

        if show_comparison:
            fig.add_trace(go.Heatmap(
                z=valid_mask_np,
                hovertext=hover_text_validity,
                hoverinfo='text',
                colorscale=[[0, '#d73027'], [1, '#4575b4']],
                showscale=False,
                zmin=0, zmax=1
            ), row=current_row, col=2)
            fig.update_yaxes(title_text="Validity", showticklabels=False, row=current_row, col=2)
        current_row += 1

    # Update x-axis for the bottom row
    fig.update_xaxes(title_text="Time Steps", row=num_subplots, col=1)
    if show_comparison:
        fig.update_xaxes(title_text="Time Steps", row=num_subplots, col=2)

    return (fig, session_info_text)

# --- Main Execution ---
def setup_device():
    """Sets up the compute device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {'GPU: ' + torch.cuda.get_device_name() if device == 'cuda' else 'CPU'}")
    return device

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Interactive FoG Dataset Inspector.")
    parser.add_argument("--experiment", "-e", type=str, default="classification/best_fixed", help="Hydra experiment configuration.")
    parser.add_argument("--config_dir", type=str, default=None, help="Custom configs directory path.")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args

def load_hydra_config(config_dir: str, experiment: str, overrides: list[str] = None) -> DictConfig:
    """Load Hydra configuration."""
    all_overrides = [f"experiment={experiment}"] + (overrides or [])
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=all_overrides)

def setup_data_module(cfg: DictConfig) -> FOGDataModule:
    """Setup FOG data module."""
    normalize_data_paths(cfg.data.paths)
    data_cfg = load_pydantic_config(cfg.data, DataConfig)
    # Use 'classification' task type for inspector (needs labels for visualization)
    fog_data = FOGDataModule(data_cfg=data_cfg, task_type='classification')
    fog_data.setup(stage=None)  # Setup all datasets (train/val/test)
    return fog_data

def setup_transforms(device: str):
    """Instantiate default transformation models for interactive visualization."""
    return {
        'stft': STFTTransform(sample_rate=SAMPLE_RATE, n_fft=256, hop_length=32, win_length=128, device=device, interpolate=True),
        'mel_spectrogram': MelSpectrogramTransform(sample_rate=SAMPLE_RATE, n_fft=256, hop_length=32, win_length=128, n_mels=80, f_min=0.5, f_max=20.0, interpolate=True, device=device),
        'cwt': WaveletTransform(n_scales=64, wavelet='morl', max_freq=10.0, interpolate=True)
    }

if __name__ == '__main__':
    args = parse_arguments()
    
    config_dir = args.config_dir or str(PROJECT_ROOT / "configs")
    CFG = load_hydra_config(config_dir, args.experiment, args.overrides)
    DEVICE = setup_device()
    DATA_MODULE = setup_data_module(CFG)
    TRANSFORMS = setup_transforms(DEVICE)
    
    print("--- Interactive Inspector Ready ---")
    print(f"Data loaded for experiment: {args.experiment}")
    print(f"Train: {len(DATA_MODULE.train_dataset)} samples, Val: {len(DATA_MODULE.val_dataset)} samples, Test: {len(DATA_MODULE.test_dataset)} samples")
    print("\nAccess options:")
    print("  Option A - Direct access: http://YOUR_SERVER_IP:8052")
    print("  Option B - SSH forwarding: ssh -L 8052:localhost:8052 user@server")
    print("             Then open: http://localhost:8052")
    
    app.run(debug=True, host='0.0.0.0', port=8052)

