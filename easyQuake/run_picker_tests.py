#!/usr/bin/env python3
"""
Copy the test script to ~/easyQuake/tests/ and run it.
"""

import os
import shutil
import sys
from pathlib import Path

def main():
    # Source and destination paths
    source = Path(__file__).parent / 'test_all_pickers.py'
    dest_dir = Path.home() / 'easyQuake' / 'tests'
    dest = dest_dir / 'test_all_pickers.py'
    
    # Create tests directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the test script
    shutil.copy2(source, dest)
    print(f"Copied test script to {dest}")
    
    # Make it executable
    os.chmod(dest, 0o755)
    
    # Run the test
    print("Running picker test suite...")
    os.chdir(dest_dir)
    os.system(f"python3 {dest}")

if __name__ == '__main__':
    main()
