"""Dataset-specific configuration."""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class DatasetConfig(BaseModel):
    """Dataset-specific configuration for PyTorch datasets."""

    axis: List[str] = Field(
        default=['AccV', 'AccML', 'AccAP'],
        description="Accelerometer axes to load"
    )

    normalize_per_session: bool = Field(
        default=False,
        description="Enable per-session normalization"
    )

    session_stats_path: Optional[str] = Field(
        default=None,
        description="Path to session-level normalization statistics"
    )

    # Classification-specific fields
    validate_labels: bool = Field(
        default=False,
        description="Validate labels during dataset initialization"
    )

    classification_strategy: Optional[str] = Field(
        default=None,
        description="Strategy for classification (e.g., 'binary_any_fog')"
    )

    # Train-time data budget: keep only the first N minutes of each patient's
    # recording in the TRAIN split (no effect on val/test). Used for the
    # fine-tuning-data-size curve (Fig 12). None = use all data.
    max_train_minutes: Optional[float] = Field(
        default=None,
        description="Cap TRAIN data to the first N minutes per patient (None = no cap)"
    )

    # Cap the TRAIN split to K patients (label-efficiency curve, Fig S4). Patients
    # are chosen FOG-stratified + seeded so the same K are picked for every arm.
    max_train_patients: Optional[int] = Field(
        default=None,
        description="Cap TRAIN data to K FOG-stratified patients (None = no cap)"
    )
    train_subset_seed: int = Field(
        default=0,
        description="Seed for max_train_patients selection (shared across arms for fairness)"
    )

    sampling_rate_hz: int = Field(
        default=100,
        description="Accelerometer sampling rate (Hz), used to convert max_train_minutes to frames"
    )

    # Optional metadata filter applied at dataset-build time. Replaces the legacy
    # gitignored "view" YAML files for evaluation: instead of pointing dataset_path
    # at a view, point it at the physical .zarr and set this query. Uses the same
    # pandas-query semantics as data.views.filters.apply_filters, e.g. "validity == 1.0".
    metadata_query: Optional[str] = Field(
        default=None,
        description="Optional pandas-query filter on patch metadata, e.g. 'validity == 1.0' (None = no filter)"
    )

    # Hydra instantiation target (excluded from validation)
    target_: Optional[str] = Field(
        default=None,
        alias='_target_',
        description="Hydra _target_ for dataset instantiation",
        exclude=True
    )

    @field_validator('axis')
    @classmethod
    def validate_axis(cls, v: List[str]) -> List[str]:
        """Validate axis list contains only valid accelerometer axes."""
        if not v:
            raise ValueError("axis list cannot be empty")

        valid_axes = {'AccV', 'AccML', 'AccAP'}
        for ax in v:
            if ax not in valid_axes:
                raise ValueError(
                    f"Invalid axis '{ax}'. Must be one of {valid_axes}"
                )
        return v

    model_config = {
        "frozen": True,  # Immutable
        "extra": "forbid",  # No extra fields allowed
        "populate_by_name": True  # Allow population by field name or alias
    }
