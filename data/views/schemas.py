"""Pydantic schemas for virtual dataset view configuration."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any


class ViewMetadata(BaseModel):
    """Metadata about the view itself."""
    name: str = Field(description="View name (kebab-case recommended)")
    description: str = Field(description="Human-readable description of view purpose")
    created_at: str = Field(description="ISO 8601 timestamp of view creation")
    version: str = Field(default="1.0", description="View config format version")
    format_version: str = Field(default="1.0", description="View schema version")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate view name follows conventions."""
        if not v or not v.strip():
            raise ValueError("View name cannot be empty")
        # Allow alphanumeric, underscore, hyphen
        if not all(c.isalnum() or c in ('_', '-') for c in v):
            raise ValueError(
                f"View name '{v}' contains invalid characters. "
                "Use only alphanumeric, underscore, or hyphen."
            )
        return v.strip()


class SourceConfig(BaseModel):
    """Configuration for view source dataset."""
    dataset_path: str = Field(description="Path to physical Zarr dataset")
    dataset_type: str = Field(default="physical", description="Dataset type (must be 'physical')")

    @field_validator('dataset_type')
    @classmethod
    def validate_dataset_type(cls, v: str) -> str:
        """Ensure only physical datasets can be sources."""
        if v != 'physical':
            raise ValueError(
                f"dataset_type must be 'physical'. Got: '{v}'. "
                "Views cannot reference other views (nested views not supported)."
            )
        return v

    @field_validator('dataset_path')
    @classmethod
    def validate_dataset_path(cls, v: str) -> str:
        """Validate dataset path format."""
        if not v or not v.strip():
            raise ValueError("dataset_path cannot be empty")

        # Check if path looks like a view config (prevent nesting)
        if v.endswith('.yaml') or v.endswith('.yml'):
            raise ValueError(
                f"dataset_path '{v}' appears to be a view config. "
                "Views cannot reference other views. Use a physical .zarr dataset."
            )

        return v.strip()


class FilterConfig(BaseModel):
    """Filter criteria for view subsetting."""
    protocol: Optional[List[str]] = Field(
        default=None,
        description="Filter by protocol(s). None = no filter."
    )
    class_label: Optional[List[int]] = Field(
        default=None,
        description="Filter by class label(s). None = no filter."
    )
    pure_patches_only: Optional[bool] = Field(
        default=False,
        description="If True, include only patches with purity=1.0"
    )
    patient_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter by specific patient IDs. None = all patients."
    )
    custom_query: Optional[str] = Field(
        default=None,
        description="Advanced: pandas query string for custom filtering"
    )

    model_config = {
        "extra": "forbid"  # Prevent unknown filter fields
    }


class ViewConfig(BaseModel):
    """Complete view configuration schema."""
    view_metadata: ViewMetadata
    source: SourceConfig
    filters: FilterConfig
    computed: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Computed statistics (auto-generated, optional)"
    )

    model_config = {
        "extra": "forbid"  # Strict schema - no extra fields
    }
