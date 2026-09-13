"""Command-line interface for managing virtual dataset views."""
import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from utils.zarr_helpers import read_zarr_metadata
from data.views.filters import apply_filters, summarize_filtered_metadata
from data.views.validator import validate_view_config
from data.views.resolver import resolve_dataset_path


def create_view(args):
    """Create a new view config file."""
    # Resolve source path
    source_path = Path(args.source).resolve()
    if not source_path.exists():
        print(f"Error: Source dataset not found: {source_path}")
        sys.exit(1)

    print(f"Loading source dataset: {source_path}")

    # Load metadata
    try:
        metadata_df = read_zarr_metadata(str(source_path), '/metadata')
    except Exception as e:
        print(f"Error loading source metadata: {e}")
        sys.exit(1)

    print(f"Source dataset: {len(metadata_df)} patches")

    # Parse filters from command-line arguments
    filters = parse_filter_args(args.filter)

    if not filters:
        print("Warning: No filters specified. View will contain all patches from source.")

    # Apply filters to preview results
    print(f"\nApplying filters...")
    for key, value in filters.items():
        if value is not None:
            print(f"  {key}: {value}")

    try:
        filtered_df = apply_filters(metadata_df, filters)
    except Exception as e:
        print(f"Error applying filters: {e}")
        sys.exit(1)

    print(f"\nFiltered result: {len(filtered_df)} patches")

    # Check for empty view
    if len(filtered_df) == 0:
        print("\nError: Filters produced empty view. No patches match the criteria.")
        print("Please adjust your filter criteria and try again.")
        sys.exit(1)

    # Compute statistics
    computed_stats = summarize_filtered_metadata(filtered_df)

    # Display statistics
    print("\nView statistics:")
    print(f"  Total patches: {computed_stats['total_patches']}")
    if 'patient_count' in computed_stats:
        print(f"  Patients: {computed_stats['patient_count']}")
    if 'session_count' in computed_stats:
        print(f"  Sessions: {computed_stats['session_count']}")
    if 'protocol_distribution' in computed_stats:
        print(f"  Protocols: {computed_stats['protocol_distribution']}")
    if 'class_distribution' in computed_stats:
        print(f"  Classes: {computed_stats['class_distribution']}")

    # Build view config
    view_config = {
        'view_metadata': {
            'name': args.name,
            'description': args.description or f"View: {args.name}",
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'version': '1.0',
            'format_version': '1.0'
        },
        'source': {
            'dataset_path': str(source_path),
            'dataset_type': 'physical'
        },
        'filters': filters,
        'computed': computed_stats
    }

    # Validate config
    try:
        validate_view_config(view_config)
    except Exception as e:
        print(f"Error: View config validation failed: {e}")
        sys.exit(1)

    # Write to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w') as f:
            yaml.dump(view_config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Error writing view config: {e}")
        sys.exit(1)

    print(f"\nView created successfully: {output_path}")
    print(f"\nTo use this view, reference it in your config:")
    print(f"  paths:")
    print(f"    processed_dataset_path: {output_path}")


def parse_filter_args(filter_args):
    """
    Parse --filter key=value arguments into filters dictionary.

    Supports:
    - Single values: --filter protocol=defog
    - Lists (comma-separated): --filter protocol=defog,tdcsfog
    - Booleans: --filter pure_patches_only=true
    - Multiple filters: --filter protocol=defog --filter class_label=1

    Args:
        filter_args: List of "key=value" strings from argparse

    Returns:
        Dictionary of filters
    """
    filters = {}

    if not filter_args:
        return filters

    for filter_spec in filter_args:
        if '=' not in filter_spec:
            print(f"Error: Invalid filter format: '{filter_spec}'. Use key=value")
            sys.exit(1)

        key, value = filter_spec.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Handle list values (comma-separated)
        if ',' in value:
            value = [v.strip() for v in value.split(',')]
            # Try to convert to int if possible (for class_label)
            try:
                value = [int(v) if v.isdigit() else v for v in value]
            except ValueError:
                pass

        # Handle boolean values
        elif value.lower() in ['true', 'false']:
            value = value.lower() == 'true'

        # Handle null
        elif value.lower() == 'null' or value.lower() == 'none':
            value = None

        # For protocol and class_label, always use list format (Pydantic expects lists)
        elif key in ['protocol', 'class_label']:
            # Convert single value to list
            if value.isdigit():
                value = [int(value)]
            else:
                value = [value]

        # Try to convert to int if it's a single numeric value (for other fields)
        elif value.isdigit():
            value = int(value)

        filters[key] = value

    return filters


def list_views(args):
    """List all available views."""
    views_dir = Path(args.views_dir)

    if not views_dir.exists():
        print(f"No views directory found: {views_dir}")
        return

    view_files = sorted(list(views_dir.glob('*.yaml')) + list(views_dir.glob('*.yml')))

    if not view_files:
        print(f"No views found in {views_dir}")
        return

    print(f"Available views in {views_dir}:\n")
    print(f"{'Name':<30} {'Patches':>10} {'Description'}")
    print("-" * 80)

    for view_file in view_files:
        try:
            with open(view_file, 'r') as f:
                config = yaml.safe_load(f)

            name = config['view_metadata']['name']
            patches = config.get('computed', {}).get('total_patches', '?')
            desc = config['view_metadata'].get('description', '')

            # Truncate description if too long
            if len(desc) > 35:
                desc = desc[:32] + '...'

            print(f"{name:<30} {patches:>10} {desc}")

        except Exception as e:
            print(f"{view_file.name:<30} {'ERROR':>10} Failed to load: {e}")

    print()


def validate_view(args):
    """Validate a view config file."""
    view_path = Path(args.view_config)

    if not view_path.exists():
        print(f"Error: View config not found: {view_path}")
        sys.exit(1)

    # Load config
    try:
        with open(view_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading view config: {e}")
        sys.exit(1)

    # Validate
    try:
        view_config = validate_view_config(config, config_path=str(view_path))
        print(f"✓ View config is valid: {view_path}")
        print(f"  View name: {view_config.view_metadata.name}")
        print(f"  Source: {view_config.source.dataset_path}")

        # Try to resolve the view
        try:
            resolved = resolve_dataset_path(str(view_path))
            print(f"  ✓ Successfully resolved view: {len(resolved.metadata_df)} patches")
        except Exception as e:
            print(f"  ⚠ Warning: View config is valid but resolution failed: {e}")
            sys.exit(1)

    except Exception as e:
        print(f"✗ View config validation failed:")
        print(f"  {e}")
        sys.exit(1)


def show_info(args):
    """Show detailed information about a view."""
    view_path = Path(args.view_config)

    if not view_path.exists():
        print(f"Error: View config not found: {view_path}")
        sys.exit(1)

    # Load config
    try:
        with open(view_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading view config: {e}")
        sys.exit(1)

    # Display info
    print(f"View: {config['view_metadata']['name']}")
    print(f"Description: {config['view_metadata'].get('description', 'N/A')}")
    print(f"Created: {config['view_metadata'].get('created_at', 'N/A')}")
    print(f"Version: {config['view_metadata'].get('version', 'N/A')}")

    print(f"\nSource Dataset:")
    print(f"  Path: {config['source']['dataset_path']}")
    print(f"  Type: {config['source']['dataset_type']}")

    print(f"\nFilters:")
    filters_applied = False
    for key, value in config['filters'].items():
        if value is not None:
            print(f"  {key}: {value}")
            filters_applied = True

    if not filters_applied:
        print("  (No filters applied)")

    # Statistics
    if 'computed' in config and config['computed']:
        stats = config['computed']
        print(f"\nStatistics:")
        print(f"  Total patches: {stats.get('total_patches', '?')}")

        if 'patient_count' in stats:
            print(f"  Patients: {stats['patient_count']}")
        if 'session_count' in stats:
            print(f"  Sessions: {stats['session_count']}")

        if 'class_distribution' in stats:
            print(f"  Class distribution:")
            for label, count in stats['class_distribution'].items():
                print(f"    Class {label}: {count}")

        if 'protocol_distribution' in stats:
            print(f"  Protocol distribution:")
            for protocol, count in stats['protocol_distribution'].items():
                print(f"    {protocol}: {count}")

        if 'purity_stats' in stats:
            print(f"  Purity:")
            print(f"    Mean: {stats['purity_stats']['mean']:.3f}")
            print(f"    Pure patches: {stats['purity_stats']['pure_count']}")

    # Try to resolve view
    print(f"\nResolution test:")
    try:
        resolved = resolve_dataset_path(str(view_path))
        print(f"  ✓ View resolves successfully")
        print(f"  Physical dataset: {resolved.zarr_path}")
        print(f"  Patches in view: {len(resolved.metadata_df)}")
    except Exception as e:
        print(f"  ✗ Failed to resolve view: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Manage virtual dataset views for FOG pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a view filtering for DEFOG protocol
  python -m data.views create \\
      --source data/processed/len200_stride100_kaggle.zarr \\
      --name defog_only \\
      --output data/processed/views/defog_only.yaml \\
      --filter protocol=defog

  # Create view with multiple filters
  python -m data.views create \\
      --source data/processed/len200_stride100_kaggle.zarr \\
      --name fog_high_purity \\
      --filter protocol=defog,tdcsfog \\
      --filter class_label=1 \\
      --filter pure_patches_only=true \\
      --output data/processed/views/fog_high_purity.yaml

  # List all views
  python -m data.views list

  # Validate view
  python -m data.views validate data/processed/views/defog_only.yaml

  # Show view info
  python -m data.views info data/processed/views/defog_only.yaml
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new view')
    create_parser.add_argument(
        '--source',
        required=True,
        help='Source .zarr dataset path'
    )
    create_parser.add_argument(
        '--name',
        required=True,
        help='View name (alphanumeric, underscore, hyphen)'
    )
    create_parser.add_argument(
        '--output',
        required=True,
        help='Output .yaml file path'
    )
    create_parser.add_argument(
        '--description',
        help='View description'
    )
    create_parser.add_argument(
        '--filter',
        action='append',
        help='Filter: key=value (can be repeated). Use comma for lists.'
    )

    # List command
    list_parser = subparsers.add_parser('list', help='List all views')
    list_parser.add_argument(
        '--views-dir',
        default='data/processed/views',
        help='Views directory (default: data/processed/views)'
    )

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate view config')
    validate_parser.add_argument(
        'view_config',
        help='Path to view .yaml file'
    )

    # Info command
    info_parser = subparsers.add_parser('info', help='Show view information')
    info_parser.add_argument(
        'view_config',
        help='Path to view .yaml file'
    )

    args = parser.parse_args()

    if args.command == 'create':
        create_view(args)
    elif args.command == 'list':
        list_views(args)
    elif args.command == 'validate':
        validate_view(args)
    elif args.command == 'info':
        show_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
