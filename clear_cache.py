#!/usr/bin/env python3
"""
Utility script to manage the dataset cache for TimesFM fine-tuning.
"""
import shutil
from pathlib import Path

def main():
    """Clear the dataset cache."""
    cache_dir = Path(__file__).parent / "dataset_cache"
    
    if not cache_dir.exists():
        print("✅ No cache directory found.")
        return
    
    # List cache files
    cache_files = list(cache_dir.glob("*.pkl"))
    
    if not cache_files:
        print("✅ Cache directory is empty.")
        return
    
    print(f"🗑️  Found {len(cache_files)} cache files:")
    total_size = 0
    for f in cache_files:
        size = f.stat().st_size / (1024**2)
        total_size += size
        print(f"   - {f.name} ({size:.2f} MB)")
    
    print(f"\n   Total cache size: {total_size:.2f} MB")
    
    # Ask for confirmation
    response = input("\n⚠️  Delete all cache files? (y/N): ").strip().lower()
    
    if response == 'y':
        for f in cache_files:
            f.unlink()
        print("✅ Cache cleared successfully!")
    else:
        print("❌ Cache clearing cancelled.")

if __name__ == "__main__":
    main()
