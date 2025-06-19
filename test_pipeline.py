#!/usr/bin/env python3
"""
Test script for the Hybrid CV + LLM Archaeological Discovery System
This runs the pipeline without OpenAI API calls to demonstrate the CV components
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append('/home/myuser/OpenAI_to_z_Challenge')

# Import our solution
from hybrid_cv_llm_solution import *

def test_mock_data_generation():
    """Test the mock data generation functionality."""
    print("Testing mock data generation...")
    
    # Setup directories
    setup_environment()
    
    # Create mock raster
    source_path = create_mock_raster()
    
    # Verify file was created
    if source_path.exists():
        print("✓ Mock raster created successfully")
        
        # Check file properties
        with rasterio.open(source_path) as src:
            print(f"  - Dimensions: {src.width} x {src.height}")
            print(f"  - Bands: {src.count}")
            print(f"  - CRS: {src.crs}")
            print(f"  - Bounds: {src.bounds}")
        
        return source_path
    else:
        print("✗ Mock raster creation failed")
        return None

def test_tiling_functionality(source_path):
    """Test the raster tiling functionality."""
    print("\nTesting raster tiling...")
    
    if source_path is None:
        print("✗ Cannot test tiling without source raster")
        return None
    
    # Tile the data
    df_tiles = tile_raster_data(source_path)
    
    if not df_tiles.empty:
        print(f"✓ Tiling successful - Created {len(df_tiles)} tiles")
        print(f"  - Sample tile: {df_tiles.iloc[0]['tile_id']}")
        
        # Verify tile files exist
        sample_tile = df_tiles.iloc[0]
        if os.path.exists(sample_tile['rgb_path']):
            print("  - RGB tiles created successfully")
        if os.path.exists(sample_tile['ndvi_path']):
            print("  - NDVI tiles created successfully")
        if os.path.exists(sample_tile['slope_path']):
            print("  - Slope tiles created successfully")
        
        return df_tiles
    else:
        print("✗ Tiling failed")
        return None

def test_cv_filtering(df_tiles):
    """Test the classical CV filtering functionality."""
    print("\nTesting classical CV filtering...")
    
    if df_tiles is None or df_tiles.empty:
        print("✗ Cannot test CV filtering without tiles")
        return None, None
    
    # Apply CV filtering
    df_analysis, df_candidates = filter_candidates_with_cv(df_tiles)
    
    if not df_analysis.empty:
        print(f"✓ CV filtering successful")
        print(f"  - Total tiles analyzed: {len(df_analysis)}")
        print(f"  - Candidates found: {len(df_candidates)}")
        print(f"  - Success rate: {len(df_candidates)/len(df_analysis)*100:.1f}%")
        
        # Show geometry scores
        scores = df_analysis['geometry_score']
        print(f"  - Geometry scores: min={scores.min()}, max={scores.max()}, mean={scores.mean():.1f}")
        
        return df_analysis, df_candidates
    else:
        print("✗ CV filtering failed")
        return None, None

def test_composite_image_creation(df_candidates):
    """Test composite image creation for LLM analysis."""
    print("\nTesting composite image creation...")
    
    if df_candidates is None or df_candidates.empty:
        print("✗ Cannot test composite creation without candidates")
        return
    
    # Create composite for first candidate
    sample_candidate = df_candidates.iloc[0]
    try:
        composite = create_composite_image(sample_candidate)
        print(f"✓ Composite image created successfully")
        print(f"  - Shape: {composite.shape}")
        print(f"  - Type: {composite.dtype}")
        
        # Save composite for inspection
        composite_path = CONFIG['RESULTS_DIR'] / 'test_composite.png'
        cv2.imwrite(str(composite_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        print(f"  - Saved test composite: {composite_path}")
        
    except Exception as e:
        print(f"✗ Composite image creation failed: {e}")

def test_visualization(df_analysis, df_candidates):
    """Test the visualization functionality."""
    print("\nTesting visualization...")
    
    try:
        # Create a mock final results dataframe to test visualization
        mock_final = df_candidates.head(2).copy() if not df_candidates.empty else pd.DataFrame()
        
        if not mock_final.empty:
            # Add mock LLM results
            mock_final['contains_anthropogenic_features'] = [True, False]
            mock_final['confidence_score'] = [0.85, 0.23]
            mock_final['feature_type_guess'] = ['ancient settlement', 'natural formation']
            mock_final['rationale'] = [
                'Geometric patterns suggest human construction',
                'Irregular shapes appear natural'
            ]
            
            visualize_results(df_analysis, mock_final)
            print("✓ Visualization completed successfully")
        else:
            print("⚠ No candidates available for visualization")
            
    except Exception as e:
        print(f"✗ Visualization failed: {e}")

def run_comprehensive_test():
    """Run comprehensive test of the pipeline (without OpenAI API)."""
    print("=" * 60)
    print("HYBRID CV + LLM PIPELINE TEST")
    print("=" * 60)
    
    # Test 1: Mock data generation
    source_path = test_mock_data_generation()
    
    # Test 2: Tiling
    df_tiles = test_tiling_functionality(source_path)
    
    # Test 3: CV filtering
    df_analysis, df_candidates = test_cv_filtering(df_tiles)
    
    # Test 4: Composite image creation
    test_composite_image_creation(df_candidates)
    
    # Test 5: Visualization
    test_visualization(df_analysis, df_candidates)
    
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)
    
    # Summary
    if df_tiles is not None and not df_tiles.empty:
        print(f"✓ Mock data generation: SUCCESS")
        print(f"✓ Tiling functionality: SUCCESS ({len(df_tiles)} tiles)")
        
        if df_analysis is not None and not df_analysis.empty:
            print(f"✓ CV filtering: SUCCESS ({len(df_candidates)} candidates)")
            print(f"✓ Pipeline ready for LLM integration")
        else:
            print(f"✗ CV filtering: FAILED")
    else:
        print(f"✗ Pipeline test: FAILED")
    
    print(f"\nTo run with OpenAI API:")
    print(f"1. Set OPENAI_API_KEY environment variable")
    print(f"2. Run: python hybrid_cv_llm_solution.py")

if __name__ == "__main__":
    run_comprehensive_test()