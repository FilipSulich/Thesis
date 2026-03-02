"""
Example: MIT-BIH Visualization

Shows how to use the simplified MITBIHVisualizer API.
"""

from pathlib import Path
from mitbih_visualizer import MITBIHVisualizer, OUTPUT_DIR

# =============================================================================
# Configuration
# =============================================================================

LEAD_TIME = 5  # Which lead time dataset to load

# Subband parameters for RQA (optional - uses defaults if not provided)
subband_params = {
    "original": {"tau": 21, "m": 7},
    "D1": {"tau": 6, "m": 7},
    "D2": {"tau": 4, "m": 7},
    "D3": {"tau": 2, "m": 7},
    "D4": {"tau": 2, "m": 7},
    "A4": {"tau": 16, "m": 8},
}

# Output directory
VIZ_DIR = OUTPUT_DIR / 'visualizations'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Initialize visualizer (loads CSV with both pre-AF and normal signals)
    print(f"Initializing visualizer for lead time {LEAD_TIME} minutes...")
    viz = MITBIHVisualizer(subband_params=subband_params, lead_time=LEAD_TIME)

    print("\n" + "="*80)
    print("EXAMPLE 1: Visualize Pre-AF Signal with Subbands and RPs")
    print("="*80)

    # Plot random pre-AF signal
    # Shows: Original signal + RP, then D1-D4 + A4 subbands with their RPs
    viz.plot_subband_rps(
        label_filter=1,  # 1 = pre-AF only
        save_path=VIZ_DIR / f'example_preaf_lt{LEAD_TIME}.png'
    )

    print("\n" + "="*80)
    print("EXAMPLE 2: Visualize Normal Signal with Subbands and RPs")
    print("="*80)

    # Plot random normal signal
    viz.plot_subband_rps(
        label_filter=0,  # 0 = normal only
        save_path=VIZ_DIR / f'example_normal_lt{LEAD_TIME}.png'
    )

    print("\n" + "="*80)
    print("EXAMPLE 3: Compare Multiple Pre-AF vs Normal")
    print("="*80)

    # Plot 3 examples of each
    for i in range(3):
        print(f"  Generating pair {i+1}/3...")

        # Pre-AF
        viz.plot_subband_rps(
            label_filter=1,
            save_path=VIZ_DIR / f'preaf_{i+1}_lt{LEAD_TIME}.png'
        )

        # Normal
        viz.plot_subband_rps(
            label_filter=0,
            save_path=VIZ_DIR / f'normal_{i+1}_lt{LEAD_TIME}.png'
        )

    print("\n" + "="*80)
    print("EXAMPLE 4: Specific Signal by Index")
    print("="*80)

    # Get a specific pre-AF signal
    indices = viz.get_random_indices(label=1, n=1)
    if indices:
        idx = indices[0]
        print(f"Selected pre-AF signal index: {idx}")

        viz.plot_subband_rps(
            idx=idx,
            save_path=VIZ_DIR / f'specific_preaf_idx{idx}.png'
        )

    # Get a specific normal signal
    indices = viz.get_random_indices(label=0, n=1)
    if indices:
        idx = indices[0]
        print(f"Selected normal signal index: {idx}")

        viz.plot_subband_rps(
            idx=idx,
            save_path=VIZ_DIR / f'specific_normal_idx{idx}.png'
        )

    print("\n" + "="*80)
    print("EXAMPLE 5: Subband Decomposition (No RPs)")
    print("="*80)

    # Just show the decomposed subbands without RPs
    indices = viz.get_random_indices(label=1, n=1)
    if indices:
        idx = indices[0]
        viz.plot_subbands(
            idx,
            apply_cleaning=False,  # Already cleaned
            normalize_per_subband=True,
            save_path=VIZ_DIR / f'subbands_only_idx{idx}.png'
        )

    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {VIZ_DIR}")
    print(f"{'='*80}\n")

    # List created files
    print("Created files:")
    for file in sorted(VIZ_DIR.glob('*.png')):
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name} ({size_kb:.1f} KB)")
