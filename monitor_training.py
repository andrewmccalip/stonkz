#!/usr/bin/env python3
"""
Monitor training progress by watching the plot directory.
Run this in a separate terminal to see real-time training updates.
"""
import os
import time
from pathlib import Path
from datetime import datetime

PLOT_DIR = Path("training_plots")

def monitor_plots():
    """Monitor the training plots directory for updates."""
    print("🔍 Monitoring training plots...")
    print(f"📁 Plot directory: {PLOT_DIR}")
    print("=" * 50)
    
    if not PLOT_DIR.exists():
        print("⚠️  Plot directory doesn't exist yet. Waiting for training to start...")
        while not PLOT_DIR.exists():
            time.sleep(5)
        print("✅ Plot directory created!")
    
    last_mtime = 0
    
    while True:
        latest_plot = PLOT_DIR / "latest.png"
        
        if latest_plot.exists():
            current_mtime = os.path.getmtime(latest_plot)
            
            if current_mtime > last_mtime:
                last_mtime = current_mtime
                
                # Get all plot files
                plot_files = sorted(PLOT_DIR.glob("training_iter_*.png"))
                
                if plot_files:
                    latest_file = plot_files[-1]
                    timestamp = datetime.fromtimestamp(current_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    
                    print(f"\n📊 New plot generated at {timestamp}")
                    print(f"   Latest: {latest_file.name}")
                    print(f"   Total plots: {len(plot_files)}")
                    print(f"   View: {latest_plot.absolute()}")
                    
                    # Show plot info
                    file_size = latest_plot.stat().st_size / 1024  # KB
                    print(f"   Size: {file_size:.1f} KB")
        
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    try:
        monitor_plots()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped.")
