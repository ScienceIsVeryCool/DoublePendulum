#!/usr/bin/env python3
"""
Double Pendulum Batch Visualization Creator
Converts batch simulation JSON logs into mesmerizing grid videos/GIFs
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
import glob
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import re

class PendulumVideoCreator:
    def __init__(self, config: Dict):
        self.config = config
        self.simulations = {}
        self.grid_shape = None
        self.max_ticks = 0
        
    def load_simulations(self, data_dir: str) -> None:
        """Load all JSON simulation files from directory"""
        json_files = glob.glob(os.path.join(data_dir, "pendulum_*.json"))
        
        print(f"Found {len(json_files)} simulation files")
        
        for file_path in json_files:
            try:
                # Extract angles from filename (pendulum_1_570_2_356.json)
                filename = Path(file_path).stem
                match = re.search(r'pendulum_(\d+_\d+)_(\d+_\d+)', filename)
                if match:
                    ang0_str = match.group(1).replace('_', '.')
                    ang1_str = match.group(2).replace('_', '.')
                    ang0 = float(ang0_str)
                    ang1 = float(ang1_str)
                else:
                    print(f"Skipping file with unexpected name format: {filename}")
                    continue
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                self.simulations[(ang0, ang1)] = {
                    'data': data,
                    'ang0': ang0,
                    'ang1': ang1,
                    'max_tick': max([entry['tick'] for entry in data])
                }
                
                self.max_ticks = max(self.max_ticks, len(data))
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(self.simulations)} simulations, max ticks: {self.max_ticks}")
    
    def calculate_grid_layout(self) -> Tuple[int, int]:
        """Calculate optimal grid layout based on angle ranges"""
        if not self.simulations:
            return (1, 1)
        
        # Get unique angles
        ang0_values = sorted(set(sim['ang0'] for sim in self.simulations.values()))
        ang1_values = sorted(set(sim['ang1'] for sim in self.simulations.values()))
        
        print(f"Grid dimensions: {len(ang0_values)} x {len(ang1_values)}")
        
        return len(ang1_values), len(ang0_values)  # rows, cols
    
    def get_simulation_stats(self, data: List[Dict]) -> Dict:
        """Calculate interesting statistics for a simulation"""
        if not data:
            return {}
        
        stats = {
            'total_energy': [],
            'upper_bob_path_length': 0,
            'lower_bob_path_length': 0,
            'max_angular_velocity': 0,
            'final_upper_spins': 0,
            'final_lower_spins': 0,
        }
        
        prev_upper_pos = None
        prev_lower_pos = None
        
        for entry in data:
            pendulum = entry['pendulumState']
            
            # Calculate total energy (kinetic + potential)
            l0, l1 = pendulum['l0'], pendulum['l1']
            m0, m1 = pendulum['m0'], pendulum['m1']
            g = pendulum['g']
            ang0, ang1 = pendulum['ang0'], pendulum['ang1']
            
            # Get bob positions
            x0, y0 = pendulum['x0'], pendulum['y0']
            upper_x = x0 + l0 * np.sin(ang0)
            upper_y = y0 + l0 * np.cos(ang0)
            lower_x = upper_x + l1 * np.sin(ang1)
            lower_y = upper_y + l1 * np.cos(ang1)
            
            # Calculate path lengths
            if prev_upper_pos:
                stats['upper_bob_path_length'] += np.sqrt(
                    (upper_x - prev_upper_pos[0])**2 + (upper_y - prev_upper_pos[1])**2
                )
            if prev_lower_pos:
                stats['lower_bob_path_length'] += np.sqrt(
                    (lower_x - prev_lower_pos[0])**2 + (lower_y - prev_lower_pos[1])**2
                )
            
            prev_upper_pos = (upper_x, upper_y)
            prev_lower_pos = (lower_x, lower_y)
            
            # Potential energy (taking y=0 as reference at the pivot)
            pe = -(m0 + m1) * g * l0 * np.cos(ang0) - m1 * g * l1 * np.cos(ang1)
            
            # Kinetic energy (from moments)
            moment0, moment1 = pendulum.get('moment0', 0), pendulum.get('moment1', 0)
            # Convert moments to velocities for energy calculation
            cos_diff = np.cos(ang0 - ang1)
            den = (m0 + m1) * m1 * l0**2 * l1**2 - (m1 * l0 * l1 * cos_diff)**2
            if abs(den) > 1e-9:
                v0 = (moment0 * m1 * l1**2 - moment1 * m1 * l0 * l1 * cos_diff) / den
                v1 = (moment1 * (m0 + m1) * l0**2 - moment0 * m1 * l0 * l1 * cos_diff) / den
            else:
                v0 = v1 = 0
            
            ke = 0.5 * (m0 + m1) * l0**2 * v0**2 + 0.5 * m1 * l1**2 * v1**2 + m1 * l0 * l1 * v0 * v1 * cos_diff
            
            total_energy = ke + pe
            stats['total_energy'].append(total_energy)
            stats['max_angular_velocity'] = max(stats['max_angular_velocity'], abs(v0), abs(v1))
        
        return stats
    
    def create_frame(self, tick: int, fig, axes) -> None:
        """Create a single frame of the animation"""
        rows, cols = self.grid_shape
        
        # Clear all subplots
        for ax in axes.flat:
            ax.clear()
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            ax.axis('off')
        
        # Get unique angles for positioning
        ang0_values = sorted(set(sim['ang0'] for sim in self.simulations.values()))
        ang1_values = sorted(set(sim['ang1'] for sim in self.simulations.values()))
        
        for (ang0, ang1), sim_data in self.simulations.items():
            # Find grid position
            try:
                col = ang0_values.index(ang0)
                row = ang1_values.index(ang1)
            except ValueError:
                continue
            
            if row >= rows or col >= cols:
                continue
                
            ax = axes[row, col] if rows > 1 else axes[col]
            data = sim_data['data']
            
            # Get data for this tick (handle different lengths)
            if tick < len(data):
                entry = data[tick]
                pendulum = entry['pendulumState']
                
                # Extract pendulum state
                x0, y0 = pendulum['x0'], pendulum['y0']
                l0, l1 = pendulum['l0'], pendulum['l1']
                ang0_curr, ang1_curr = pendulum['ang0'], pendulum['ang1']
                
                # Normalize coordinates to [-1, 1] range
                scale = max(l0 + l1, 1.0)
                x0_norm = (x0 - 50) / (50 * scale) * 0.8  # Assuming original coords are ~0-100
                y0_norm = (y0 - 50) / (50 * scale) * 0.8
                l0_norm = l0 / scale * 0.8
                l1_norm = l1 / scale * 0.8
                
                # Calculate bob positions
                upper_x = x0_norm + l0_norm * np.sin(ang0_curr)
                upper_y = y0_norm + l0_norm * np.cos(ang0_curr)
                lower_x = upper_x + l1_norm * np.sin(ang1_curr)
                lower_y = upper_y + l1_norm * np.cos(ang1_curr)
                
                # Draw pendulum
                if self.config['show_pendulum']:
                    # Strings
                    ax.plot([x0_norm, upper_x], [y0_norm, upper_y], 'w-', linewidth=1, alpha=0.8)
                    ax.plot([upper_x, lower_x], [upper_y, lower_y], 'w-', linewidth=1, alpha=0.8)
                    
                    # Bobs
                    bob_size = 0.03
                    ax.add_patch(Circle((x0_norm, y0_norm), bob_size, color='white', alpha=0.8))
                    ax.add_patch(Circle((upper_x, upper_y), bob_size, color='cyan', alpha=0.8))
                    ax.add_patch(Circle((lower_x, lower_y), bob_size, color='yellow', alpha=0.8))
                
                # Add trace if enabled
                if self.config['show_trace'] and tick > 0:
                    trace_length = min(tick, self.config['trace_length'])
                    trace_start = max(0, tick - trace_length)
                    
                    trace_x, trace_y = [], []
                    for t in range(trace_start, tick):
                        if t < len(data):
                            trace_entry = data[t]
                            trace_pendulum = trace_entry['pendulumState']
                            trace_x0 = (trace_pendulum['x0'] - 50) / (50 * scale) * 0.8
                            trace_y0 = (trace_pendulum['y0'] - 50) / (50 * scale) * 0.8
                            trace_l0 = trace_pendulum['l0'] / scale * 0.8
                            trace_l1 = trace_pendulum['l1'] / scale * 0.8
                            trace_ang0 = trace_pendulum['ang0']
                            trace_ang1 = trace_pendulum['ang1']
                            
                            trace_upper_x = trace_x0 + trace_l0 * np.sin(trace_ang0)
                            trace_upper_y = trace_y0 + trace_l0 * np.cos(trace_ang0)
                            trace_lower_x = trace_upper_x + trace_l1 * np.sin(trace_ang1)
                            trace_lower_y = trace_upper_y + trace_l1 * np.cos(trace_ang1)
                            
                            trace_x.append(trace_lower_x)
                            trace_y.append(trace_lower_y)
                    
                    if trace_x:
                        alpha_vals = np.linspace(0.1, 0.6, len(trace_x))
                        for i in range(len(trace_x)-1):
                            ax.plot([trace_x[i], trace_x[i+1]], [trace_y[i], trace_y[i+1]], 
                                   color='red', alpha=alpha_vals[i], linewidth=0.5)
                
                # Add statistics text
                if self.config['show_stats']:
                    stats_text = []
                    if 'total_energy' in self.config['displayed_stats']:
                        if tick < len(data):
                            # Calculate current energy
                            pe = -(pendulum.get('m0', 100) + pendulum.get('m1', 100)) * pendulum.get('g', 0.1) * \
                                 l0 * np.cos(ang0_curr) - pendulum.get('m1', 100) * pendulum.get('g', 0.1) * l1 * np.cos(ang1_curr)
                            stats_text.append(f"E: {pe:.1f}")
                    
                    if 'angular_velocity' in self.config['displayed_stats']:
                        moment0 = pendulum.get('moment0', 0)
                        moment1 = pendulum.get('moment1', 0)
                        stats_text.append(f"ω: {abs(moment0)+abs(moment1):.1f}")
                    
                    if 'tick' in self.config['displayed_stats']:
                        stats_text.append(f"T: {tick}")
                    
                    if stats_text:
                        ax.text(-1.1, 1.0, '\n'.join(stats_text), fontsize=6, 
                               color='white', verticalalignment='top', family='monospace')
            
            # Add initial condition labels
            if self.config['show_initial_conditions']:
                ax.text(0, -1.15, f"({ang0:.2f}, {ang1:.2f})", 
                       fontsize=6, color='gray', ha='center', family='monospace')
        
        # Add global title
        fig.suptitle(f"Double Pendulum Batch Simulation - Tick {tick}/{self.max_ticks-1}", 
                    fontsize=10, color='white')
    
    def create_video(self, output_path: str, fps: int = 30) -> None:
        """Create the final video/GIF"""
        if not self.simulations:
            print("No simulations loaded!")
            return
        
        self.grid_shape = self.calculate_grid_layout()
        rows, cols = self.grid_shape
        
        # Setup figure
        fig_width = min(20, cols * 2.5)
        fig_height = min(20, rows * 2.5)
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height),
                                facecolor='black', edgecolor='none')
        fig.patch.set_facecolor('black')
        
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.1, wspace=0.1)
        
        # Determine frame range
        max_frames = min(self.max_ticks, self.config['max_frames'])
        frame_step = max(1, self.max_ticks // max_frames)
        frames = list(range(0, self.max_ticks, frame_step))
        
        print(f"Creating animation with {len(frames)} frames...")
        
        def animate(frame_idx):
            tick = frames[frame_idx]
            self.create_frame(tick, fig, axes)
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx+1}/{len(frames)}")
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                     interval=1000//fps, blit=False, repeat=True)
        
        # Save
        if output_path.endswith('.gif'):
            print(f"Saving GIF to {output_path}...")
            anim.save(output_path, writer='pillow', fps=fps, dpi=100)
        else:
            print(f"Saving MP4 to {output_path}...")
            anim.save(output_path, writer='ffmpeg', fps=fps, dpi=150, bitrate=2000)
        
        plt.close()
        print("Video creation complete!")

def create_config():
    """Create default configuration"""
    return {
        # Visual elements
        'show_pendulum': True,
        'show_trace': True,
        'show_stats': True,
        'show_initial_conditions': True,
        
        # Trace settings
        'trace_length': 100,  # Number of previous positions to show
        
        # Statistics to display
        'displayed_stats': ['total_energy', 'tick'],  # Options: 'total_energy', 'angular_velocity', 'tick'
        
        # Animation settings
        'max_frames': 500,  # Maximum number of frames (will skip frames if needed)
        'fps': 24,
        
        # Output settings
        'output_format': 'gif',  # 'gif' or 'mp4'
    }

def main():
    parser = argparse.ArgumentParser(description='Create videos from double pendulum batch simulations')
    parser.add_argument('data_dir', help='Directory containing JSON simulation files')
    parser.add_argument('-o', '--output', default='pendulum_batch.gif', 
                       help='Output file path (default: pendulum_batch.gif)')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second (default: 24)')
    parser.add_argument('--max-frames', type=int, default=500, 
                       help='Maximum number of frames (default: 500)')
    parser.add_argument('--no-trace', action='store_true', help='Disable pendulum traces')
    parser.add_argument('--no-stats', action='store_true', help='Disable statistics display')
    parser.add_argument('--trace-length', type=int, default=100, 
                       help='Length of pendulum trace (default: 100)')
    
    args = parser.parse_args()
    
    # Create and customize config
    config = create_config()
    config['fps'] = args.fps
    config['max_frames'] = args.max_frames
    config['show_trace'] = not args.no_trace
    config['show_stats'] = not args.no_stats
    config['trace_length'] = args.trace_length
    
    # Determine output format from extension
    if args.output.endswith('.mp4'):
        config['output_format'] = 'mp4'
    
    print("Double Pendulum Batch Video Creator")
    print("=" * 40)
    print(f"Data directory: {args.data_dir}")
    print(f"Output file: {args.output}")
    print(f"Settings: {config}")
    print()
    
    # Create video
    creator = PendulumVideoCreator(config)
    creator.load_simulations(args.data_dir)
    
    if not creator.simulations:
        print("No valid simulation files found!")
        return
    
    creator.create_video(args.output, fps=args.fps)

if __name__ == "__main__":
    main()