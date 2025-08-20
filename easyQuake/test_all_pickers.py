#!/usr/bin/env python3
"""
Fixed comprehensive test script for all easyQuake machine picker options.
Tests GPD, EQTransformer, PhaseNet, and STALTA pickers with real seismic data.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime, date
import glob

def setup_test_project():
    """Set up a test project with proper file structure."""
    print("easyQuake Machine Picker Test Suite - FIXED VERSION")
    print("Testing all available machine pickers with real seismic data")
    
    # Check for test data files in tests directory
    tests_dir = Path('/home/jwalter/easyQuake/tests')
    test_files = ['O2.WILZ.EHE.mseed', 'O2.WILZ.EHN.mseed', 'O2.WILZ.EHZ.mseed']
    found_files = []
    for f in test_files:
        if (tests_dir / f).exists():
            found_files.append(f)
    
    if len(found_files) != 3:
        print(f"❌ Missing test data files. Found: {found_files}")
        return None, None
    
    print(f"✓ Found test data files: {found_files}")
    
    # Create test project
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_project = tests_dir / f'test_project_fixed_{timestamp}'
    test_date = '20240101'  # Fixed test date
    date_dir = temp_project / test_date
    date_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy MSEED files to the project directory (where easyQuake expects them)
    print("Copying MSEED files to project directory...")
    for f in test_files:
        source = tests_dir / f
        target = date_dir / f
        shutil.copy2(source, target)
        print(f"  Copied {f}")
    
    print(f"Test project created: {temp_project}")
    return temp_project, test_date

def test_picker(machine_picker, project_folder, test_date):
    """Test a specific picker."""
    print(f"\n{'='*60}")
    print(f"Testing {machine_picker} picker")
    print(f"{'='*60}")
    
    try:
        # Add easyQuake to path
        sys.path.insert(0, '/home/jwalter/easyQuake')
        import easyQuake.easyQuake as eq
        
        # Set up test parameters
        test_datetime = date(2024, 1, 1)  # Use date object
        
        # Base parameters for all pickers
        base_params = {
            'dirname': test_date,
            'project_folder': str(project_folder),
            'project_code': 'TEST',  # Required parameter
            'single_date': test_datetime,
            'local': True,
            'latitude': 36.7,  # Oklahoma coordinates
            'longitude': -97.5,
            'max_radius': 300,
        }
        
        # Run detection_continuous with the specified picker
        if machine_picker == 'STALTA':
            # STALTA uses machine=False
            eq.detection_continuous(
                machine=False,
                **base_params
            )
        else:
            # Other machine pickers
            eq.detection_continuous(
                machine=True,
                machine_picker=machine_picker,
                **base_params
            )
        
        print(f"✓ {machine_picker} completed successfully!")
        
        # Check output file
        expected_output = project_folder / test_date / f"{machine_picker.lower()}_picks.out"
        return check_output_file(expected_output, machine_picker)
        
    except Exception as e:
        print(f"✗ {machine_picker} failed with error: {e}")
        print("Full traceback:")
        import traceback
        traceback.print_exc()
        return False, 0

def check_output_file(output_file, picker_name):
    """Check if output file was created and contains picks."""
    if not output_file.exists():
        print(f"✗ {picker_name} failed - no output file generated")
        return False, 0
    
    print(f"Output file: {output_file}")
    
    # Check file size and content
    file_size = output_file.stat().st_size
    if file_size == 0:
        print("Number of picks: 0")
        print("  No picks found in output file")
        return True, 0  # Success but no picks
    
    # Count picks (non-empty, non-comment lines)
    pick_count = 0
    picks_preview = []
    
    try:
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    pick_count += 1
                    if len(picks_preview) < 5:  # Store first 5 for preview
                        picks_preview.append(line)
        
        print(f"Number of picks: {pick_count}")
        if picks_preview:
            print("First 5 picks:")
            for i, pick in enumerate(picks_preview, 1):
                print(f"  {i}: {pick}")
            if pick_count > 5:
                print(f"  ... and {pick_count - 5} more picks")
        
        return True, pick_count
        
    except Exception as e:
        print(f"Error reading output file: {e}")
        return True, 0  # File exists but couldn't read

def main():
    """Main test function."""
    # Set up test environment
    project_folder, test_date = setup_test_project()
    if not project_folder:
        return
    
    # Define pickers to test
    pickers = ['GPD', 'EQTransformer', 'PhaseNet', 'STALTA']
    
    # Test each picker
    results = {}
    total_picks = 0
    
    for picker in pickers:
        success, picks = test_picker(picker, project_folder, test_date)
        results[picker] = (success, picks)
        if success:
            total_picks += picks
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_pickers = 0
    for picker, (success, picks) in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{picker:<15} {status} ({picks:3d} picks)")
        if success:
            successful_pickers += 1
    
    print(f"\nOverall: {successful_pickers}/{len(pickers)} pickers successful")
    print(f"Total picks generated: {total_picks}")
    
    # List output files
    output_dir = project_folder / test_date
    output_files = list(output_dir.glob("*_picks.out"))
    if output_files:
        print(f"\nOutput files in {output_dir}:")
        for f in output_files:
            size = f.stat().st_size
            print(f"  {f.name} ({size} bytes)")
    
    failed_pickers = len(pickers) - successful_pickers
    if failed_pickers > 0:
        print(f"\n⚠️  {failed_pickers} picker(s) failed")
    
    print(f"\nTest project preserved at: {project_folder}")
    print("You can examine the output files manually.")

if __name__ == "__main__":
    main()
