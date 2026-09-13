"""
Protocol definitions for FOG datasets.

Defines structural subtyping contracts that datasets can satisfy
without explicit inheritance. Uses PEP 544 Protocol for type safety.
"""

from typing import Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class SamplableDataset(Protocol):
    """
    Protocol for datasets that support balanced sampling.

    Any dataset that provides sampling metadata can be used with
    ProbabilisticBalancedSampler, regardless of inheritance hierarchy.

    Required Methods:
        get_sampling_metadata(): Returns DataFrame with required columns
        __len__(): Returns number of samples

    Required Metadata Columns:
        - class_label (int): Class index for class balancing
        - protocol (str): Protocol identifier for protocol balancing
        - patient_id (str): Patient identifier for patient balancing

    Example:
        >>> dataset = FOGClassificationDataset(...)
        >>> isinstance(dataset, SamplableDataset)  # True
        >>> metadata = dataset.get_sampling_metadata()
        >>> metadata.columns
        Index(['class_label', 'protocol', 'patient_id'], dtype='object')
    """

    def get_sampling_metadata(self) -> pd.DataFrame:
        """
        Return metadata DataFrame for balanced sampling.

        Returns:
            DataFrame with at least these columns:
                - class_label (int): Class labels for each sample
                - protocol (str): Protocol identifier
                - patient_id (str): Patient identifier

            Additional columns are allowed but not used by sampler.

        Raises:
            RuntimeError: If metadata not available (e.g., dataset not initialized)

        Note:
            - DataFrame length must match len(dataset)
            - class_label can be uniform (all 0) for unlabeled datasets
            - Order must match dataset indexing (metadata[i] describes dataset[i])
        """
        ...

    def __len__(self) -> int:
        """
        Return number of samples in dataset.

        Returns:
            Number of samples available for sampling
        """
        ...
