#!/usr/bin/env python3
"""
Test the improved TF2 PhaseNet implementation to see if it generates realistic pick counts.
"""

import os
import sys
import logging
import tempfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Add phasenet directory to path
phasenet_dir = os.path.join(os.path.dirname(__file__), 'phasenet')
sys.path.insert(0, phasenet_dir)

def test_tf2_phasenet():
    """Test the TF2 PhaseNet implementation."""
    print("Testing improved TF2 PhaseNet implementation...")
    
    try:
        # Import the TF2 implementation
        from phasenet_predict_tf2 import pred_fn_tf2
        
        # Create mock arguments
        class MockArgs:
            batch_size = 1
            nt = 3000
            dt = 0.01
            result_fname = 'test_phasenet_picks.out'
        
        # Create mock data reader
        class MockDataReader:
            num_data = 10  # Process 10 samples
        
        args = MockArgs()
        data_reader = MockDataReader()
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run the prediction
            picks = pred_fn_tf2(args, data_reader, log_dir=temp_dir)
            
            # Analyze results
            total_picks = len(picks)
            p_picks = len([p for p in picks if p.get('phase_type', p.get('phase', '')) == 'P'])
            s_picks = len([p for p in picks if p.get('phase_type', p.get('phase', '')) == 'S'])
            
            print(f"\n=== TF2 PhaseNet Test Results ===")
            print(f"Total picks generated: {total_picks}")
            print(f"P picks: {p_picks}")
            print(f"S picks: {s_picks}")
            print(f"Target was ~799 picks (like TF1)")
            
            if total_picks > 100:
                print("✓ SUCCESS: Generated reasonable number of picks!")
                print("This is much closer to the TF1 target of ~799 picks")
            elif total_picks > 10:
                print("⚠ PARTIAL: Generated some picks, but fewer than expected")
            else:
                print("✗ FAILED: Generated very few picks (like the old version)")
            
            # Show sample picks
            if picks:
                print(f"\nFirst 5 picks:")
                for i, pick in enumerate(picks[:5]):
                    phase = pick.get('phase_type', pick.get('phase', 'Unknown'))
                    time = pick.get('phase_time', pick.get('pick_time', 0))
                    score = pick.get('phase_score', pick.get('pick_prob', 0))
                    print(f"  {i+1}: {phase} pick at {time:.2f}s, confidence: {score:.3f}")
            
            return total_picks >= 100  # Success if we get reasonable number of picks
            
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tf2_phasenet()
    if success:
        print("\n🎉 TF2 PhaseNet improvement test PASSED!")
        print("The enhanced implementation generates realistic pick counts.")
    else:
        print("\n❌ TF2 PhaseNet improvement test FAILED!")
        print("The implementation still needs work to match TF1 performance.")
    
    sys.exit(0 if success else 1)
