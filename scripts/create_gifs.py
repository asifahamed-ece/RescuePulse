#!/usr/bin/env python3
"""
Generate animated GIFs showing RescuePulse execution flows.
Creates visualizations for Phase 1 (Acoustic Detection) and Phase 2 (Traffic Control).
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.animation import PillowWriter
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "assets" / "gifs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_phase1_gif():
    """Create animated GIF for Phase 1: Acoustic Detection Flow"""

    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Phase 1: Acoustic Siren Detection Flow',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # Define stages
    stages = [
        {'pos': (1, 7.5), 'label': 'Microphones\n(Stereo INMP441)', 'color': '#FFB6C1'},
        {'pos': (2.5, 5.5), 'label': 'I2S DMA Capture\n(16 kHz, 256 samples)', 'color': '#87CEEB'},
        {'pos': (4, 3.5), 'label': 'MFCC Extraction\n(64 frames, 13 coeff)', 'color': '#90EE90'},
        {'pos': (5.5, 1.5), 'label': 'Quantization\n(INT8)', 'color': '#FFD700'},
        {'pos': (7, 3.5), 'label': 'TFLite Inference\n(1D CNN)', 'color': '#DDA0DD'},
        {'pos': (8.5, 5.5), 'label': 'TDOA DoA Est.\n(Cross-corr)', 'color': '#F0E68C'},
        {'pos': (9, 7.5), 'label': 'Output\n(Confidence)', 'color': '#98FB98'},
    ]

    frames = []
    total_frames = 120

    for frame in range(total_frames):
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        ax.text(5, 9.5, 'Phase 1: Acoustic Siren Detection Flow',
                ha='center', va='top', fontsize=18, fontweight='bold')

        # Animate stages
        for i, stage in enumerate(stages):
            # Calculate animation timing
            stage_start = (i / len(stages)) * total_frames
            stage_end = stage_start + (total_frames / len(stages)) * 0.8

            if frame >= stage_start and frame < stage_end:
                # Active stage - highlight
                alpha = min((frame - stage_start) / 10, 1.0)
                color = stage['color']
                edge_color = 'red'
                edge_width = 3
                fontsize = 11
                fontweight = 'bold'
            elif frame >= stage_end:
                # Completed stage - normal
                alpha = 0.7
                color = stage['color']
                edge_color = 'black'
                edge_width = 1.5
                fontsize = 10
                fontweight = 'normal'
            else:
                # Not yet active
                alpha = 0.2
                color = 'lightgray'
                edge_color = 'gray'
                edge_width = 0.5
                fontsize = 9
                fontweight = 'normal'

            # Draw box
            box = FancyBboxPatch((stage['pos'][0] - 0.6, stage['pos'][1] - 0.4),
                                1.2, 0.8,
                                boxstyle="round,pad=0.05",
                                edgecolor=edge_color, facecolor=color,
                                alpha=alpha, linewidth=edge_width)
            ax.add_patch(box)

            # Add text
            ax.text(stage['pos'][0], stage['pos'][1], stage['label'],
                   ha='center', va='center', fontsize=fontsize,
                   fontweight=fontweight)

        # Draw arrows with animation
        arrows = [
            ((1.6, 7.5), (2.5, 6.3)),
            ((3.2, 5.5), (4, 4.3)),
            ((4.6, 3.5), (5.5, 2.3)),
            ((6.2, 1.5), (7, 2.3)),
            ((7.6, 3.5), (8.5, 4.3)),
            ((8.9, 6.3), (9, 7.1)),
        ]

        for idx, (start, end) in enumerate(arrows):
            arrow_start = max(0, min(1.0, (frame - idx * 15) / 15))
            if arrow_start > 0:
                arrow = FancyArrowPatch(start, end,
                                      arrowstyle='->', mutation_scale=20,
                                      color='gray', alpha=min(arrow_start, 0.7),
                                      linewidth=2)
                ax.add_patch(arrow)

        # Add metrics at bottom
        ax.text(1, 0.3, 'Latency: ~1.0-1.6s', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax.text(5, 0.3, 'Confidence: 0.70-1.00', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
        ax.text(9, 0.3, 'Model: INT8 Quantized', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.7))

        plt.tight_layout()
        frames.append(fig)

    # Save as GIF
    writer = PillowWriter(fps=15)
    gif_path = OUTPUT_DIR / 'phase1-acoustic-detection-flow.gif'
    with writer.saving(fig, str(gif_path), dpi=100):
        for frame in range(total_frames):
            writer.grab_frame()

    plt.close(fig)
    print(f"✅ Phase 1 GIF created: {gif_path}")


def create_phase2_gif():
    """Create animated GIF for Phase 2: Traffic Control State Machine"""

    fig, ax = plt.subplots(figsize=(14, 9), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Phase 2: Traffic Control State Machine',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # Define states
    states = {
        'NORMAL': {'pos': (2, 6.5), 'color': '#90EE90'},
        'CLEARANCE': {'pos': (5, 6.5), 'color': '#FFB6C1'},
        'EMERGENCY': {'pos': (8, 6.5), 'color': '#FF6B6B'},
    }

    frames = []
    total_frames = 200

    # Define state sequence
    state_sequence = [
        ('NORMAL', 40),      # Green lane cycling (40 frames = 2.67s)
        ('CLEARANCE', 30),   # All red (30 frames = 2s)
        ('EMERGENCY', 50),   # Emergency green (50 frames = 3.33s)
        ('CLEARANCE', 30),   # Clear again
        ('NORMAL', 20),      # Resume normal
    ]

    frame_idx = 0
    current_sequence_idx = 0
    current_state_frames = 0

    for frame in range(total_frames):
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        ax.text(5, 9.5, 'Phase 2: Traffic Control State Machine',
                ha='center', va='top', fontsize=18, fontweight='bold')

        # Determine current state
        current_state, state_duration = state_sequence[current_sequence_idx % len(state_sequence)]
        current_state_frames += 1

        if current_state_frames >= state_duration:
            current_sequence_idx = (current_sequence_idx + 1) % len(state_sequence)
            current_state_frames = 0
            current_state, _ = state_sequence[current_sequence_idx]

        # Draw states
        for state_name, state_info in states.items():
            if state_name == current_state:
                # Active state
                box = FancyBboxPatch((state_info['pos'][0] - 0.8, state_info['pos'][1] - 0.5),
                                    1.6, 1.0,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='darkred', facecolor=state_info['color'],
                                    alpha=1.0, linewidth=4)
                fontsize = 12
                fontweight = 'bold'
            else:
                # Inactive state
                box = FancyBboxPatch((state_info['pos'][0] - 0.8, state_info['pos'][1] - 0.5),
                                    1.6, 1.0,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='gray', facecolor=state_info['color'],
                                    alpha=0.5, linewidth=1)
                fontsize = 10
                fontweight = 'normal'

            ax.add_patch(box)
            ax.text(state_info['pos'][0], state_info['pos'][1], state_name,
                   ha='center', va='center', fontsize=fontsize, fontweight=fontweight)

        # Draw transitions
        transitions = [
            ((2.8, 6.5), (4.2, 6.5)),  # NORMAL -> CLEARANCE
            ((5.8, 6.5), (7.2, 6.5)),  # CLEARANCE -> EMERGENCY
            ((8, 5.9), (5, 5.9)),      # EMERGENCY -> CLEARANCE (curved)
            ((5, 7.1), (2, 7.1)),      # CLEARANCE -> NORMAL (curved)
        ]

        for idx, (start, end) in enumerate(transitions):
            arrow = FancyArrowPatch(start, end,
                                  arrowstyle='->', mutation_scale=20,
                                  color='black', alpha=0.6, linewidth=2,
                                  connectionstyle="arc3,rad=0.3")
            ax.add_patch(arrow)

        # Draw traffic lanes visualization
        lane_y = 3.5
        lanes = [
            {'name': 'LEFT', 'x': 1.5, 'color_map': {'NORMAL': 'green', 'CLEARANCE': 'red', 'EMERGENCY': 'red'}},
            {'name': 'CENTER', 'x': 5, 'color_map': {'NORMAL': 'red', 'CLEARANCE': 'red', 'EMERGENCY': 'red'}},
            {'name': 'RIGHT', 'x': 8.5, 'color_map': {'NORMAL': 'red', 'CLEARANCE': 'red', 'EMERGENCY': 'green'}},
        ]

        # Update lane colors based on state
        for lane in lanes:
            if current_state == 'NORMAL':
                # Cycle through lanes
                cycle_pos = (current_state_frames / state_duration) * 3
                if int(cycle_pos) == lanes.index(lane):
                    lane_color = 'green'
                else:
                    lane_color = 'red'
            else:
                lane_color = lane['color_map'].get(current_state, 'red')

            # Draw traffic light circle
            circle = mpatches.Circle((lane['x'], lane_y), 0.3,
                                    color=lane_color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(lane['x'], lane_y - 0.7, lane['name'],
                   ha='center', fontsize=9, fontweight='bold')

        ax.text(5, 2.3, 'Traffic Lane Status', ha='center', fontsize=11, fontweight='bold')

        # Add timing info
        timing_text = f"Elapsed: {current_state_frames}/{state_duration} frames"
        ax.text(5, 0.8, timing_text, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#90EE90', edgecolor='black', label='NORMAL: Autonomous Cycling'),
            mpatches.Patch(facecolor='#FFB6C1', edgecolor='black', label='CLEARANCE: 2s All-Red'),
            mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='EMERGENCY: Priority Green'),
        ]
        ax.legend(handles=legend_elements, loc='lower center',
                 ncol=3, fontsize=9, framealpha=0.9)

        plt.tight_layout()
        frames.append(fig)

    # Save as GIF
    writer = PillowWriter(fps=15)
    gif_path = OUTPUT_DIR / 'phase2-traffic-control-flow.gif'
    with writer.saving(fig, str(gif_path), dpi=100):
        for frame in range(total_frames):
            writer.grab_frame()

    plt.close(fig)
    print(f"✅ Phase 2 GIF created: {gif_path}")


def create_end_to_end_gif():
    """Create animated GIF showing end-to-end system flow"""

    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.7, 'RescuePulse: End-to-End System Flow',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # System components
    components = [
        # Phase 1
        {'name': 'Mic Input', 'pos': (1, 8), 'core': '—', 'phase': 1},
        {'name': 'I2S DMA', 'pos': (2.5, 8), 'core': '0', 'phase': 1},
        {'name': 'MFCC', 'pos': (4, 6), 'core': '1', 'phase': 1},
        {'name': 'TFLite', 'pos': (5.5, 6), 'core': '1', 'phase': 1},
        {'name': 'DoA Est.', 'pos': (7, 6), 'core': '1', 'phase': 1},
        # Phase 2
        {'name': 'Detect Msg', 'pos': (8.5, 8), 'core': '—', 'phase': 2},
        {'name': 'Traffic FSM', 'pos': (8.5, 5), 'core': '1', 'phase': 2},
        {'name': 'GPIO Ctrl', 'pos': (8.5, 2.5), 'core': '—', 'phase': 2},
        {'name': 'LEDs', 'pos': (7, 1), 'core': '—', 'phase': 2},
    ]

    frames = []
    total_frames = 150

    for frame in range(total_frames):
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        ax.text(5, 9.7, 'RescuePulse: End-to-End System Flow',
                ha='center', va='top', fontsize=18, fontweight='bold')

        # Draw Phase labels
        ax.text(3.5, 9.2, 'Phase 1: Acoustic Detection', fontsize=11,
               fontweight='bold', color='blue')
        ax.text(8.5, 9.2, 'Phase 2: Traffic Control', fontsize=11,
               fontweight='bold', color='red')

        # Draw components with animation
        for i, comp in enumerate(components):
            # Calculate animation timing
            comp_start = (i / len(components)) * total_frames
            comp_end = comp_start + (total_frames / len(components)) * 0.7

            if frame >= comp_start and frame < comp_end:
                # Active - highlight
                alpha = min((frame - comp_start) / 8, 1.0)
                color = '#FFD700' if comp['phase'] == 1 else '#FF6B6B'
                edge_width = 3
                fontweight = 'bold'
            elif frame >= comp_end:
                # Completed
                alpha = 0.7
                color = '#FFD700' if comp['phase'] == 1 else '#FF6B6B'
                edge_width = 1.5
                fontweight = 'normal'
            else:
                # Not yet
                alpha = 0.2
                color = 'lightgray'
                edge_width = 0.5
                fontweight = 'normal'

            # Draw box
            box = FancyBboxPatch((comp['pos'][0] - 0.55, comp['pos'][1] - 0.3),
                                1.1, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=color,
                                alpha=alpha, linewidth=edge_width)
            ax.add_patch(box)

            # Add text
            ax.text(comp['pos'][0], comp['pos'][1], comp['name'],
                   ha='center', va='center', fontsize=8.5, fontweight=fontweight)

            # Add core label
            ax.text(comp['pos'][0] + 0.65, comp['pos'][1], f"C{comp['core']}",
                   ha='center', va='center', fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.5))

        # Draw data flow arrows
        data_flows = [
            ((1.55, 8), (1.95, 8)),
            ((3.05, 8), (3.5, 6.3)),
            ((4.55, 5.7), (5.05, 5.7)),
            ((6.05, 5.7), (6.5, 5.7)),
            ((7.55, 6.3), (8.05, 8)),
            ((9.05, 7.7), (8.95, 5.3)),
            ((8.65, 4.7), (8.65, 2.8)),
            ((8.3, 2.2), (7.5, 1.3)),
        ]

        for idx, (start, end) in enumerate(data_flows):
            arrow_start = max(0, min(1.0, (frame - idx * 12) / 12))
            if arrow_start > 0.1:
                arrow = FancyArrowPatch(start, end,
                                      arrowstyle='->', mutation_scale=15,
                                      color='darkblue', alpha=min(arrow_start, 0.7),
                                      linewidth=2)
                ax.add_patch(arrow)

        # Add core visualization on left
        ax.text(0.3, 4.5, 'Core 0', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(0.3, 6, 'Core 1', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        # Add timing metrics
        metrics_y = 0.3
        ax.text(2, metrics_y, 'Phase 1 Latency: ~1.0-1.6s', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax.text(5.5, metrics_y, 'Phase 2 Response: <20ms', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
        ax.text(8.5, metrics_y, 'Total: ~1.0-1.6s', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.7))

        plt.tight_layout()
        frames.append(fig)

    # Save as GIF
    writer = PillowWriter(fps=15)
    gif_path = OUTPUT_DIR / 'end-to-end-system-flow.gif'
    with writer.saving(fig, str(gif_path), dpi=100):
        for frame in range(total_frames):
            writer.grab_frame()

    plt.close(fig)
    print(f"✅ End-to-End GIF created: {gif_path}")


if __name__ == '__main__':
    print("🎬 Creating animated flow GIFs for RescuePulse...\n")

    try:
        print("Creating Phase 1 (Acoustic Detection) GIF...")
        create_phase1_gif()

        print("\nCreating Phase 2 (Traffic Control) GIF...")
        create_phase2_gif()

        print("\nCreating End-to-End System Flow GIF...")
        create_end_to_end_gif()

        print(f"\n✅ All GIFs created successfully in: {OUTPUT_DIR}")
        print("\nNext steps:")
        print("1. Add GIFs to README.md sections")
        print("2. Add repository topics/tags on GitHub")
        print("3. Commit and push changes")

    except Exception as e:
        print(f"\n❌ Error creating GIFs: {e}")
        import traceback
        traceback.print_exc()
