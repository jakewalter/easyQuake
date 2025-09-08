#!/usr/bin/env python3
"""Quick test of GPD picker with adjusted threshold."""

import sys
import os
sys.path.insert(0, '/home/jwalter/easyQuake')

from easyQuake import detection_continuous

def test_gpd_adjusted():
    """Test GPD with the adjusted threshold."""
    print("Testing GPD picker with adjusted threshold (0.35)...")
    
    try:
        # Create a simple project
        project_folder = "/tmp/test_gpd_adjusted"
        if not os.path.exists(project_folder):
            os.makedirs(project_folder)
        
        # Write a simple station file
        station_file = os.path.join(project_folder, "station_file.txt")
        with open(station_file, 'w') as f:
            f.write("XX TEST 0.0 0.0 0.0\n")
        
        # Create some dummy seismic data
        from obspy import Trace, Stream, UTCDateTime
        import numpy as np
        
        data_dir = os.path.join(project_folder, "continuous_waveforms")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Generate 3-component data for 10 minutes at 100 Hz
        sr = 100.0
        duration = 600  # 10 minutes
        n_samples = int(sr * duration)
        t0 = UTCDateTime()
        
        for comp in ['Z', 'N', 'E']:
            tr = Trace()
            tr.stats.sampling_rate = sr
            tr.stats.starttime = t0
            tr.stats.network = 'XX'
            tr.stats.station = 'TEST'
            tr.stats.channel = f'HH{comp}'
            
            # Create realistic noise with some signal spikes
            data = 0.01 * np.random.randn(n_samples)
            # Add some P-wave like spikes
            for spike_time in [120, 240, 360]:  # At 2, 4, 6 minutes
                spike_idx = int(spike_time * sr)
                if spike_idx < n_samples - 200:
                    # P-wave like signal
                    data[spike_idx:spike_idx+200] += 0.05 * np.exp(-np.arange(200)/50) * np.sin(np.arange(200)*0.5)
            
            tr.data = data.astype('float32')
            st = Stream([tr])
            filename = os.path.join(data_dir, f"XX.TEST..HH{comp}.{t0.strftime('%Y.%j')}.mseed")
            st.write(filename, format='MSEED')
        
        print(f"Created test data in: {project_folder}")
        
        # Run GPD detection
        output_file = os.path.join(project_folder, "gpd_picks.txt")
        
        detection_continuous(
            project_folder=project_folder,
            station_file=station_file,
            continuous_waveform_directory=data_dir,
            output_file=output_file,
            machine_picker='GPD',
            starttime=t0,
            endtime=t0 + 600,
            local_to_utc=0,
            plot=False
        )
        
        # Check results
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                picks = f.readlines()
            print(f"✓ GPD generated {len(picks)} picks!")
            
            # Show first few picks
            for i, pick in enumerate(picks[:5]):
                print(f"  Pick {i+1}: {pick.strip()}")
        else:
            print("✗ No output file generated")
            
    except Exception as e:
        print(f"Error testing GPD: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpd_adjusted()
