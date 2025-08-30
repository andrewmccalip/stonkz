# FFT + TimesFM Hybrid Prediction System - Project Plan

## Project Overview
Build a hybrid prediction system that combines FFT-based pattern matching with TimesFM predictions using cosine similarity on frequency domain signatures.

## Phase 1: FFT Signature Database Generation

### Task 1.1: Database Infrastructure Setup
- [ ] Create `datasets/fft/` directory structure
- [ ] Create `datasets/fft/signatures/` for individual signature files
- [ ] Create `datasets/fft/metadata/` for database metadata
- [ ] Design signature file naming convention: `{original_csv_stem}_{sequence_idx}.csv`

### Task 1.2: FFT Processing Pipeline
- [ ] Create `fft_database_builder.py` script
- [ ] Implement sliding window extraction (416 minutes, working back from end of day)
- [ ] Implement FFT computation with both magnitude and phase preservation
- [ ] Add frequency band filtering for target patterns:
  - High frequency: 1-5 minute patterns
  - Medium frequency: 15-30 minute patterns  
  - Low frequency: 60-90 minute patterns
- [ ] Store corresponding outcome data (next 96 minutes)

### Task 1.3: Data Processing Logic
- [ ] Process all CSV files in `datasets/ES/`
- [ ] For each CSV:
  - Identify end-of-day periods (prioritize market hours)
  - Extract overlapping 416-minute windows working backwards
  - Skip overnight/low-volume periods
  - Generate FFT signatures for each window
- [ ] Handle edge cases (insufficient data, gaps, etc.)

### Task 1.4: Signature File Format
Each signature CSV should contain:
- [ ] `sequence_id`: Unique identifier
- [ ] `source_file`: Original CSV filename
- [ ] `start_idx`, `end_idx`: Window boundaries in original data
- [ ] `timestamp_start`, `timestamp_end`: Time boundaries
- [ ] `context_data`: The 416 price points used for FFT
- [ ] `outcome_data`: The subsequent 96 price points
- [ ] `fft_magnitude`: Magnitude spectrum (416 values)
- [ ] `fft_phase`: Phase spectrum (416 values)
- [ ] `dominant_freq_1min`, `dominant_freq_15min`, `dominant_freq_60min`: Key frequencies in each band

### Task 1.5: Database Metadata
- [ ] Create `database_info.json` with:
  - Total number of signatures
  - Date range covered
  - Source files processed
  - Processing parameters used
  - Statistics (mean/std of various metrics)

## Phase 2: FFT Similarity Search Engine

### Task 2.1: Signature Loading System
- [ ] Create `fft_signature_loader.py`
- [ ] Implement efficient loading of all signatures into memory
- [ ] Add signature indexing for fast access
- [ ] Memory optimization for large databases

### Task 2.2: Similarity Computation
- [ ] Implement cosine similarity for magnitude spectra
- [ ] Implement phase-aware similarity metric
- [ ] Create composite similarity score combining magnitude and phase
- [ ] Add frequency band weighting (emphasize target frequency ranges)

### Task 2.3: Pattern Matching Pipeline
- [ ] Create `fft_pattern_matcher.py`
- [ ] Implement k-nearest neighbors search in FFT space
- [ ] Add similarity threshold filtering
- [ ] Create outcome aggregation methods (weighted average, median, etc.)

## Phase 3: Hybrid Prediction Model

### Task 3.1: FFT-Only Predictor
- [ ] Create `prediction_fft.py`
- [ ] Implement standalone FFT-based prediction
- [ ] Add prediction confidence scoring
- [ ] Include error handling and validation

### Task 3.2: Hybrid Ensemble Model
- [ ] Create `prediction_fft_timesfm_hybrid.py`
- [ ] Integrate FFT predictor with TimesFM
- [ ] Implement ensemble weighting strategies:
  - Static weighting (α parameter)
  - Dynamic weighting based on FFT similarity confidence
  - Adaptive weighting based on market conditions
- [ ] Add prediction explanation/interpretability features

### Task 3.3: Model Integration
- [ ] Add hybrid models to `MODEL_REGISTRY` in `backtest_unified.py`
- [ ] Update model info functions
- [ ] Add hybrid-specific plotting and analysis features

## Phase 4: Analysis & Optimization

### Task 4.1: Frequency Analysis Tools
- [ ] Create `fft_analysis_tools.py`
- [ ] Implement dominant frequency detection
- [ ] Add frequency band energy analysis  
- [ ] Create frequency pattern visualization tools

### Task 4.2: Hyperparameter Optimization
- [ ] Optimize similarity thresholds
- [ ] Tune ensemble weights
- [ ] Optimize frequency band weights
- [ ] Test different k values for nearest neighbors

### Task 4.3: Performance Analysis
- [ ] Create FFT-specific evaluation metrics
- [ ] Add frequency domain prediction analysis
- [ ] Compare FFT predictions vs TimesFM vs hybrid
- [ ] Analyze performance by market conditions

## Phase 5: Advanced Features (Future)

### Task 5.1: Market Regime Detection
- [ ] Use FFT signatures to detect market regimes
- [ ] Adaptive prediction strategies per regime
- [ ] Regime-specific ensemble weighting

### Task 5.2: Multi-Scale Analysis
- [ ] Implement multiple context window sizes
- [ ] Multi-scale FFT ensemble
- [ ] Hierarchical pattern matching

### Task 5.3: Real-time Optimization
- [ ] Incremental database updates
- [ ] Fast similarity search optimizations
- [ ] Memory-efficient streaming processing

## Implementation Priority

### Sprint 1 (Week 1): Foundation
- Tasks 1.1, 1.2, 1.3 - Core database generation

### Sprint 2 (Week 1-2): Database Completion  
- Tasks 1.4, 1.5 - Complete signature database

### Sprint 3 (Week 2): Search Engine
- Tasks 2.1, 2.2, 2.3 - Similarity search system

### Sprint 4 (Week 2-3): Prediction Models
- Tasks 3.1, 3.2, 3.3 - Hybrid prediction system

### Sprint 5 (Week 3): Integration & Testing
- Integration with backtest framework
- Initial performance evaluation

## Success Metrics

1. **Database Quality**:
   - Successfully process >95% of available CSV data
   - Generate >10,000 FFT signatures
   - Cover diverse market conditions

2. **Prediction Performance**:
   - FFT-only model beats random baseline (>50% directional accuracy)
   - Hybrid model outperforms TimesFM-only on key metrics
   - Ensemble provides consistent improvement

3. **System Performance**:
   - Similarity search completes in <1 second
   - Memory usage manageable (<8GB for full database)
   - Integration seamless with existing backtest framework

## File Structure
```
datasets/
├── fft/
│   ├── signatures/           # Individual signature CSV files
│   ├── metadata/            # Database metadata
│   └── analysis/           # Analysis outputs
├── ES/                     # Original data (existing)
src/
├── fft_database_builder.py    # Database generation
├── fft_signature_loader.py    # Database loading
├── fft_pattern_matcher.py     # Similarity search
├── fft_analysis_tools.py      # Analysis utilities
├── prediction_fft.py          # FFT-only predictor
└── prediction_fft_timesfm_hybrid.py  # Hybrid model
```

## Next Steps
1. Start with Task 1.1 - Create directory structure
2. Begin Task 1.2 - Implement FFT processing pipeline
3. Test on small subset of data before full database generation
