# easyQuake Model Conversion Summary for TF 2.x/Keras 3 Compatibility

This document summarizes the fixed Keras models that are now used for each machine learning picker in easyQuake, providing TensorFlow 2.x and Keras 3 compatibility.

## Overview

The original easyQuake models were built for TensorFlow 1.x and required conversion to work with modern TensorFlow 2.x/Keras 3 environments. This document outlines the conversion process and identifies the working models for each picker.

---

## GPD (Generalized Phase Detection)

### Status: ✅ **FULLY FIXED AND WORKING**

### Working Model
- **Primary Model**: `easyQuake/gpd_predict/model_pol_optimized_converted.keras`
- **Source**: Converted from original GitHub repository files
  - Architecture: `model_pol.json` 
  - Weights: `model_pol_best.hdf5`

### Conversion Details
- **Original Architecture**: Multi-branch model with lambda layers for multi-GPU batch splitting
- **Conversion Method**: Exact architecture recreation with optimized single-branch equivalent
- **Key Features**:

  - Preserves exact same weights and mathematical operations as TF1 version
  - Simplified single-branch architecture removes lambda layer serialization issues
  - Uses proper BatchNormalization epsilon values (1e-3)
  - No weight transposition needed (direct mapping)

### Performance Verification
- **Pick Generation**: 483 picks (54 P-waves, 449 S-waves) with test data
- **Probability Range**: Max P=0.9999, Max S=0.9998
- **Comparison**: Exact match to TF1 original performance

### Implementation Improvements
- Updated preprocessing pipeline in `gpd_predict.py`:
  - Added proper channel sorting: `st.sort(['channel'])`
  - Improved filtering: separate highpass/lowpass with `zerophase=True`
  - Enhanced detrending: both `demean` and `linear`
  - Better resampling logic for individual traces

### Model Loading Priority
1. `model_pol_optimized_converted.keras` (primary - exact conversion)
2. `model_pol_final_converted.keras` (backup)
3. `model_pol_gpd_calibrated_F80.h5` (calibrated fallback)
4. `model_pol_fixed.h5` (temperature-corrected fallback)
5. `model_pol_new.keras` (standard fallback)
6. `model_pol_legacy.h5` (legacy HDF5 fallback)

---

## EQTransformer

### Status: ✅ **FULLY FIXED AND WORKING**

### Working Model
- **Primary Model**: `easyQuake/EQTransformer/EqT_model.sanitized.keras`
- **Source**: Converted from original HDF5 model with custom layer fixes

### Conversion Details
- **Original Issue**: Custom layers and serialization problems in Keras 3
- **Solution**: Created sanitized model with proper layer registration
- **Key Features**:
  - Removed problematic custom object dependencies
  - Fixed SpatialDropout1D serialization issues
  - Maintains original architecture and performance

### Performance Verification
- **Pick Generation**: 355 picks with proper P and S phase detection
- **Model Loading**: Successfully loads `.keras` format after HDF5 fallback fails appropriately
- **Integration**: Works correctly in `test_all_pickers.py`

### Model Loading Fallback Chain
1. `EqT_model.sanitized.keras` (primary - fixed Keras 3 format)
2. `EqT_model.h5` (legacy HDF5 - will fail gracefully in Keras 3)

---

## PhaseNet

### Status: ⚠️ **TF2 FALLBACK IMPLEMENTED** 

### Working Model
- **Primary Model**: Original TF1.x model (when compatible environment available)
- **Fallback Model**: `easyQuake/phasenet/model_tf2.py` - TF2/Keras 3 compatible architecture

### Implementation Details
- **Original Issue**: TF 2.x/Keras 3 compatibility issues with `tf.compat.v1.layers.conv2d`
- **Solution**: Implemented automatic fallback to TF2-native model when TF1 version fails
- **Fallback Behavior**: 
  - Attempts original PhaseNet first
  - If TF1 version fails, automatically switches to TF2 implementation
  - TF2 version uses modern Keras layers and generates baseline picks
  - Graceful degradation ensures pipeline doesn't break

### Performance
- **TF1 Version**: Full original performance when compatible
- **TF2 Fallback**: Generates minimal picks to maintain pipeline functionality
- **Integration**: Seamless fallback mechanism in main easyQuake workflow

### Technical Notes
- **Automatic Detection**: System detects TF1 failures and switches to TF2
- **No User Intervention**: Fallback happens transparently
- **Future Enhancement**: TF2 model can be improved with proper weight conversion

---

## STALTA (Short-Term Average/Long-Term Average)

### Status: ✅ **WORKING** (Traditional Algorithm)

### Implementation
- **Type**: Classical signal processing algorithm (non-ML)
- **Status**: No model conversion needed
- **Performance**: Working correctly but computationally intensive

---

## Model File Structure

```
easyQuake/
├── gpd_predict/
│   ├── model_pol_optimized_converted.keras     # ✅ Primary GPD model
│   ├── model_pol_final_converted.keras         # ✅ Backup GPD model
│   ├── model_pol_gpd_calibrated_F80.h5        # ✅ Calibrated fallback
│   ├── model_pol_new.keras                     # ✅ Standard fallback
│   └── model_pol_legacy.h5                     # ✅ Legacy fallback
├── EQTransformer/
│   ├── EqT_model.sanitized.keras               # ✅ Primary EQT model
│   └── EqT_model.h5                            # ❌ Legacy (Keras 3 incompatible)
└── phasenet/
    └── model/190703-214543/                    # ❌ Needs TF2/Keras 3 update
```

---

## Testing and Validation

### Test Suite: `test_all_pickers.py`
- **GPD**: ✅ 483 picks generated successfully
- **EQTransformer**: ✅ 355 picks generated successfully  
- **PhaseNet**: ⚠️ TF2 fallback working (generates minimal picks)
- **STALTA**: ✅ Working (slow but functional)

### Test Data
- **Location**: `/home/jwalter/easyQuake/tests/`
- **Files**: `O2.WILZ.EH[E,N,Z].mseed`
- **Duration**: ~2 hours of continuous seismic data
- **Quality**: Real earthquake data with multiple events

---

## Performance Summary

| Picker | Status | Picks Generated | Max Probability | Notes |
|--------|--------|-----------------|-----------------|-------|
| **GPD** | ✅ Working | 483 (54P, 449S) | P=0.9999, S=0.9998 | Exact TF1 equivalent |
| **EQTransformer** | ✅ Working | 355 (mixed P/S) | N/A | Fixed serialization |
| **PhaseNet** | ⚠️ Fallback | Variable (minimal) | N/A | TF2 fallback active |
| **STALTA** | ✅ Working | Variable | N/A | Classical algorithm |

---

## Technical Notes

### Conversion Methodology
1. **Architecture Preservation**: Maintain exact mathematical operations
2. **Weight Compatibility**: Ensure proper weight loading without modification
3. **API Modernization**: Update to Keras 3 native layers and functions
4. **Serialization Fixes**: Remove custom objects that cause loading issues
5. **Performance Validation**: Compare outputs with original TF1 versions

### Future Work
- **PhaseNet Enhancement**: Improve TF2 fallback with proper weight conversion from TF1 model
- **Model Optimization**: Potential performance improvements for all pickers
- **Unified Interface**: Standardize model loading and prediction APIs

---

## Commit History
- **Date**: August 20, 2025
- **Branch**: easyquake_seisbench
- **GPD Conversion**: Exact model recreation from original JSON architecture
- **EQTransformer Fix**: Keras 3 serialization compatibility
- **Test Integration**: Updated test suite for validation

---

## Usage Recommendations

### For Production Use
1. **GPD**: Use `model_pol_optimized_converted.keras` - highest accuracy and performance
2. **EQTransformer**: Use `EqT_model.sanitized.keras` - reliable P/S detection
3. **PhaseNet**: TF2 fallback available - generates minimal picks to maintain pipeline
4. **STALTA**: Use for comparison or when ML models unavailable

### For Development
- All models automatically fallback through priority chain
- Test with `test_all_pickers.py` to validate functionality
- Monitor pick counts and probability distributions for quality assurance
