#!/usr/bin/env python3
"""
Test script to verify plot clearing functionality.
"""
from pathlib import Path

def clear_plot_directories():
    """Clear all plot directories including test files."""
    script_dir = Path(__file__).parent
    plot_dirs = [
        script_dir / "finetune_plots",
        script_dir / "plots", 
        script_dir / "stock_plots"
    ]
    
    print("🧹 Clearing plot directories...")
    
    for plot_dir in plot_dirs:
        if plot_dir.exists():
            # Count files before deletion (including test files)
            plot_files = list(plot_dir.glob("*.png")) + list(plot_dir.glob("*.jpg"))
            test_files = [f for f in plot_files if f.name.startswith("test_")]
            
            if plot_files:
                if test_files:
                    print(f"   Removing {len(plot_files)} files from {plot_dir.name}/ (including {len(test_files)} test files)")
                else:
                    print(f"   Removing {len(plot_files)} files from {plot_dir.name}/")
                
                for file in plot_files:
                    try:
                        file.unlink()
                        if file.name.startswith("test_"):
                            print(f"      ✓ Removed test file: {file.name}")
                    except Exception as e:
                        print(f"   ⚠️ Failed to delete {file.name}: {e}")
            else:
                print(f"   ✓ {plot_dir.name}/ is already empty")
        else:
            print(f"   ✓ {plot_dir.name}/ doesn't exist")
    
    print("   ✅ Plot directories cleaned")

def check_directories():
    """Check current state of plot directories."""
    script_dir = Path(__file__).parent
    plot_dirs = [
        script_dir / "finetune_plots",
        script_dir / "plots", 
        script_dir / "stock_plots"
    ]
    
    print("\n📊 Current state of plot directories:")
    total_files = 0
    total_test_files = 0
    
    for plot_dir in plot_dirs:
        if plot_dir.exists():
            all_files = list(plot_dir.glob("*.png")) + list(plot_dir.glob("*.jpg"))
            test_files = [f for f in all_files if f.name.startswith("test_")]
            total_files += len(all_files)
            total_test_files += len(test_files)
            
            print(f"   {plot_dir.name}/: {len(all_files)} files total ({len(test_files)} test files)")
            if test_files:
                for f in test_files:
                    print(f"      - {f.name}")
        else:
            print(f"   {plot_dir.name}/: Directory doesn't exist")
    
    print(f"\n   Total: {total_files} files ({total_test_files} test files)")
    return total_test_files

if __name__ == "__main__":
    print("🧪 Testing Plot Clearing Functionality")
    print("=" * 50)
    
    # Check before
    print("\n📋 BEFORE clearing:")
    test_files_before = check_directories()
    
    # Clear
    print("\n🔄 CLEARING plots...")
    clear_plot_directories()
    
    # Check after
    print("\n📋 AFTER clearing:")
    test_files_after = check_directories()
    
    # Summary
    print("\n✅ Summary:")
    print(f"   Test files before: {test_files_before}")
    print(f"   Test files after: {test_files_after}")
    if test_files_after == 0:
        print("   ✅ All test files successfully removed!")
    else:
        print("   ⚠️ Some test files remain!")
