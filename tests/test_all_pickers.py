#!/usr/bin/env python3
"""
Test script for all machine picker options in easyQuake detection_continuous.

This script tests each of the available machine picker algorithms:
- GPD (Generalized Phase Detection)
- EQTransformer 
- PhaseNet
- Seisbench (requires model path)
- STALTA (Short-Term Average/Long-Term Average)

Each picker is tested with the sample data in ~/easyQuake/tests/
and outputs are verified.

Author: easyQuake test suite
Date: August 2025
"""

import os
import sys
from pathlib import Path

# Add easyQuake package to path (since we're running from tests directory)
# Look for easyQuake package in parent directory
current_dir = Path(__file__).parent
easyquake_dir = current_dir.parent
if (easyquake_dir / 'easyQuake').exists():
    sys.path.insert(0, str(easyquake_dir))
elif (current_dir.parent / 'easyQuake').exists():
    sys.path.insert(0, str(current_dir.parent))
else:
    # Try to find easyQuake package
    for parent in current_dir.parents:
        if (parent / 'easyQuake').exists():
            sys.path.insert(0, str(parent))
            break
import shutil
import tempfile
from datetime import datetime, timedelta

def setup_test_environment():
    """Set up test environment with sample data."""
    # We're already in the tests directory - get current working directory
    tests_dir = Path.cwd()
    
    # Create temporary project folder in current directory (tests)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_project = tests_dir / f'test_project_{timestamp}'
    temp_project.mkdir(exist_ok=True)
    
    # Create test date directory
    test_date = '20240101'  # Using a fixed test date
    date_dir = temp_project / test_date
    date_dir.mkdir(exist_ok=True)
    
    # Create dayfile.in with paths to real test data (relative paths since we're in tests dir)
    dayfile_path = date_dir / 'dayfile.in'
    print(f"Creating dayfile at: {dayfile_path}")
    
    with open(dayfile_path, 'w') as f:
        # Write paths to the test mseed files (use absolute paths to be safe)
        n_file = tests_dir / 'O2.WILZ.EHN.mseed'
        e_file = tests_dir / 'O2.WILZ.EHE.mseed' 
        z_file = tests_dir / 'O2.WILZ.EHZ.mseed'
        
        print(f"Checking files: {n_file}, {e_file}, {z_file}")
        print(f"N file exists: {n_file.exists()}")
        print(f"E file exists: {e_file.exists()}")
        print(f"Z file exists: {z_file.exists()}")
        
        if all(f.exists() for f in [n_file, e_file, z_file]):
            dayfile_line = f"{n_file} {e_file} {z_file}\n"
            print(f"Writing to dayfile: {dayfile_line.strip()}")
            f.write(dayfile_line)
        else:
            # List what files we found for debugging
            available_files = list(tests_dir.glob('*.mseed'))
            raise FileNotFoundError(f"Test mseed files not found. Available files: {[f.name for f in available_files]}")
    
    # Verify dayfile was written correctly
    with open(dayfile_path, 'r') as f:
        dayfile_content = f.read()
        print(f"Dayfile content: '{dayfile_content}'")
    
    return temp_project, test_date

def test_picker(machine_picker, project_folder, test_date, seisbench_model=None):
    """Test a specific picker."""
    print(f"\n{'='*60}")
    print(f"Testing {machine_picker} picker")
    print(f"{'='*60}")
    
    try:
        # Import easyQuake after path setup
        import easyQuake
        
        # Set up test parameters
        test_datetime = datetime.strptime(test_date, '%Y%m%d')
        
        # Run detection_continuous with the specified picker
        if machine_picker == 'STALTA':
            # STALTA uses machine=False
            easyQuake.detection_continuous(
                dirname=test_date,
                project_folder=str(project_folder),
                project_code='test',
                local=True,
                machine=False,  # STALTA uses machine=False
                single_date=test_datetime,
                make3=False,
                # STALTA specific parameters
                filtmin=2,
                filtmax=15,
                t_sta=0.2,
                t_lta=2.5,
                trigger_on=4,
                trigger_off=2,
                trig_horz=6.0,
                trig_vert=10.0
            )
        else:
            # All other pickers use machine=True
            kwargs = {
                'dirname': test_date,
                'project_folder': str(project_folder),
                'project_code': 'test',
                'local': True,
                'machine': True,
                'machine_picker': machine_picker,
                'single_date': test_datetime,
                'make3': False
            }
            
            # Add seisbench model if specified
            if machine_picker == 'Seisbench' and seisbench_model:
                kwargs['seisbenchmodel'] = seisbench_model
            
            easyQuake.detection_continuous(**kwargs)
        
        # Check for output file
        expected_output = project_folder / test_date / f'{machine_picker.lower()}_picks.out'
        
        if expected_output.exists():
            # Read and display first few lines of output
            with open(expected_output, 'r') as f:
                lines = f.readlines()
            
            print(f"✓ {machine_picker} completed successfully!")
            print(f"Output file: {expected_output}")
            print(f"Number of picks: {len(lines)}")
            
            if lines:
                print("First 5 picks:")
                for i, line in enumerate(lines[:5]):
                    print(f"  {i+1}: {line.strip()}")
                
                if len(lines) > 5:
                    print(f"  ... and {len(lines) - 5} more picks")
            else:
                print("  No picks found in output file")
            
            return True, len(lines)
        else:
            print(f"✗ {machine_picker} failed - no output file generated")
            return False, 0
            
    except Exception as e:
        print(f"✗ {machine_picker} failed with error: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False, 0

def main():
    """Main test function."""
    print("easyQuake Machine Picker Test Suite")
    print("Testing all available machine pickers with real seismic data")
    
    # Check that we're running from the tests directory
    current_dir = Path.cwd()
    expected_files = ['O2.WILZ.EHE.mseed', 'O2.WILZ.EHN.mseed', 'O2.WILZ.EHZ.mseed']
    missing_files = [f for f in expected_files if not (current_dir / f).exists()]
    
    if missing_files:
        print(f"Error: This script should be run from ~/easyQuake/tests/ directory")
        print(f"Missing files: {missing_files}")
        print(f"Current directory: {current_dir}")
        print(f"Please cd to ~/easyQuake/tests/ and run this script again")
        return 1
    
    print(f"✓ Running from correct directory: {current_dir}")
    print(f"✓ Found test data files: {expected_files}")
    
    try:
        # Set up test environment
        project_folder, test_date = setup_test_environment()
        print(f"Test project created: {project_folder}")
        
        # Define pickers to test
        pickers = [
            'GPD',
            'EQTransformer', 
            'PhaseNet',
            'STALTA'
        ]
        
        # Note: Seisbench requires a model path, so we'll test it conditionally
        seisbench_model = os.environ.get('SEISBENCH_MODEL_PATH')
        if seisbench_model and os.path.exists(seisbench_model):
            pickers.append('Seisbench')
            print(f"Seisbench model found: {seisbench_model}")
        else:
            print("Seisbench model not found (set SEISBENCH_MODEL_PATH env var to test)")
        
        # Test each picker
        results = {}
        for picker in pickers:
            if picker == 'Seisbench':
                success, num_picks = test_picker(picker, project_folder, test_date, seisbench_model)
            else:
                success, num_picks = test_picker(picker, project_folder, test_date)
            
            results[picker] = {'success': success, 'num_picks': num_picks}
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        successful = 0
        total_picks = 0
        
        for picker, result in results.items():
            status = "PASS" if result['success'] else "FAIL"
            print(f"{picker:15} {status:4} ({result['num_picks']:3} picks)")
            if result['success']:
                successful += 1
                total_picks += result['num_picks']
        
        print(f"\nOverall: {successful}/{len(results)} pickers successful")
        print(f"Total picks generated: {total_picks}")
        
        # List output files
        print(f"\nOutput files in {project_folder / test_date}:")
        output_dir = project_folder / test_date
        for file in sorted(output_dir.glob('*_picks.out')):
            size = file.stat().st_size
            print(f"  {file.name} ({size} bytes)")
        
        # Test completion message
        if successful == len(results):
            print(f"\n🎉 All {len(results)} pickers tested successfully!")
        else:
            print(f"\n⚠️  {len(results) - successful} picker(s) failed")
        
        print(f"\nTest project preserved at: {project_folder}")
        print("You can examine the output files manually.")
        
    except Exception as e:
        print(f"Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
