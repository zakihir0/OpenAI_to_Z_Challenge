#!/usr/bin/env python3
"""
Hybrid CV + LLM Archaeological Discovery System
Based on zdanovic's approach from the OpenAI A-to-Z Challenge

This script implements a two-stage pipeline:
1. Classical Computer Vision for broad filtering
2. GPT-4o analysis for detailed evaluation
"""

import os
import json
import time
import base64
import warnings
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Core libraries
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Geospatial libraries  
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin

# OpenRouter + Deepseek integration
import openai

# Configuration
CONFIG = {
    'USE_MOCK_DATA': True,
    
    # Area of Interest: Amazon region in Brazil 
    'AOI_BOUNDS': {'west': -70.8, 'south': -10.5, 'east': -69.8, 'north': -9.5},
    
    # Processing parameters
    'TILE_SIZE_PX': 512,
    'TILE_OVERLAP_PX': 64,
    'GEOMETRY_SCORE_THRESHOLD': 5,
    'VISUALIZATION_SAMPLES': 3,
    'LLM_ANALYSIS_COUNT': 5,
    
    # Directories
    'BASE_DIR': Path('./data'),
    'SOURCE_DATA_DIR': Path('./data/source'),
    'TILES_DIR': Path('./data/tiles'),
    'RESULTS_DIR': Path('./results'),
    'CACHE_DIR': Path('./data/cache')
}

def setup_environment():
    """Initialize project directories and environment."""
    print("Setting up environment...")
    
    # Create directories
    for dir_key in ['BASE_DIR', 'SOURCE_DATA_DIR', 'TILES_DIR', 'RESULTS_DIR', 'CACHE_DIR']:
        CONFIG[dir_key].mkdir(exist_ok=True, parents=True)
    
    # Check API key for OpenRouter
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        return False
    print("Environment setup complete.")
    return True

def create_mock_raster():
    """Creates a mock raster file with simulated archaeological anomalies."""
    print("Creating mock Amazon raster with embedded anomalies...")
    
    mock_path = CONFIG['SOURCE_DATA_DIR'] / 'amazon_mock.tif'
    shape = (2048, 2048)
    
    # Create natural background (forest)
    background = np.random.normal(80, 20, shape).astype(np.uint8)
    background = cv2.GaussianBlur(background, (9, 9), 0)
    
    # Add geometric anomalies (potential archaeological features)
    anomalies = np.zeros(shape, dtype=np.uint8)
    
    # Rectangle (ancient settlement)
    cv2.rectangle(anomalies, (200, 200), (500, 500), 200, -1)
    
    # Circle (geoglyph)
    cv2.circle(anomalies, (1300, 1300), 150, 180, -1)
    
    # Lines (causeways)
    cv2.line(anomalies, (800, 100), (800, 600), 190, 8)
    cv2.line(anomalies, (700, 400), (950, 400), 190, 8)
    
    # Blur anomalies to make them more realistic
    anomalies = cv2.GaussianBlur(anomalies, (11, 11), 0)
    
    # Combine background and anomalies
    final_raster = cv2.addWeighted(background, 0.7, anomalies, 0.9, 0)
    
    # Create geo-referenced raster
    aoi_bounds = CONFIG['AOI_BOUNDS']
    transform = from_origin(aoi_bounds['west'], aoi_bounds['north'], 0.0001, 0.0001)
    
    profile = {
        'driver': 'GTiff',
        'count': 5,  # RGB + NDVI + Slope
        'dtype': 'uint8',
        'width': shape[1],
        'height': shape[0], 
        'crs': 'EPSG:4326',
        'transform': transform
    }
    
    with rasterio.open(mock_path, 'w', **profile) as dst:
        # Write RGB bands
        for i in range(1, 4):
            dst.write(final_raster, i)
        
        # Mock NDVI (vegetation index)
        ndvi = np.random.normal(120, 30, shape).astype(np.uint8)
        dst.write(ndvi, 4)
        
        # Mock Slope (topography)
        slope = np.random.normal(100, 25, shape).astype(np.uint8)
        dst.write(slope, 5)
    
    print(f"Mock raster saved: {mock_path}")
    return mock_path

def tile_raster_data(source_path):
    """Tiles the source raster into smaller overlapping chips."""
    print(f"Tiling raster data from {source_path}...")
    
    tile_metadata = []
    tile_size = CONFIG['TILE_SIZE_PX']
    step = tile_size - CONFIG['TILE_OVERLAP_PX']
    
    with rasterio.open(source_path) as src:
        total_tiles = ((src.height - tile_size) // step + 1) * ((src.width - tile_size) // step + 1)
        
        with tqdm(total=total_tiles, desc="Creating tiles") as pbar:
            for y in range(0, src.height - tile_size, step):
                for x in range(0, src.width - tile_size, step):
                    window = Window(x, y, tile_size, tile_size)
                    tile_dir = CONFIG['TILES_DIR'] / f"tile_{x}_{y}"
                    tile_dir.mkdir(exist_ok=True)
                    
                    # Read all bands
                    bands = src.read(window=window)
                    
                    # Save RGB image
                    rgb_path = tile_dir / "rgb.png"
                    rgb_bands = bands[:3]
                    v_min, v_max = np.percentile(rgb_bands, [2, 98])
                    rgb_stretched = np.clip((rgb_bands - v_min) * 255.0 / (v_max - v_min), 0, 255).astype(np.uint8)
                    rgb_img = np.dstack(rgb_stretched)
                    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                    
                    # Save NDVI and Slope as separate files
                    ndvi_path = tile_dir / "ndvi.tif"
                    slope_path = tile_dir / "slope.tif"
                    
                    base_profile = src.profile.copy()
                    base_profile.update(
                        width=tile_size,
                        height=tile_size,
                        transform=src.window_transform(window),
                        count=1,
                        dtype='uint8'
                    )
                    
                    with rasterio.open(ndvi_path, 'w', **base_profile) as dst:
                        dst.write(bands[3], 1)
                    with rasterio.open(slope_path, 'w', **base_profile) as dst:
                        dst.write(bands[4], 1)
                    
                    # Get tile center coordinates
                    coords = src.xy(y + tile_size // 2, x + tile_size // 2)
                    
                    tile_metadata.append({
                        'tile_id': f"tile_{x}_{y}",
                        'rgb_path': str(rgb_path),
                        'ndvi_path': str(ndvi_path),
                        'slope_path': str(slope_path),
                        'lon': coords[0],
                        'lat': coords[1]
                    })
                    
                    pbar.update(1)
    
    df_tiles = pd.DataFrame(tile_metadata)
    print(f"Tiling complete. Created {len(df_tiles)} tiles.")
    return df_tiles

def detect_geometric_anomalies(image_path):
    """
    Applies classical CV to detect geometric anomalies.
    Returns geometry score and visualization images.
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0, None, None
        
        # Apply Canny edge detection
        edges = cv2.Canny(img, threshold1=30, threshold2=150)
        
        # Apply Hough line transform to detect straight lines
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180,
            threshold=25,
            minLineLength=30,
            maxLineGap=10
        )
        
        # Create visualization
        viz_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        num_lines = 0
        
        if lines is not None:
            num_lines = len(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return num_lines, edges, viz_img
        
    except Exception as e:
        print(f"CV processing error for {image_path}: {e}")
        return 0, None, None

def filter_candidates_with_cv(df_tiles):
    """Apply classical CV filtering to identify potential candidates."""
    print("Filtering tiles with classical computer vision...")
    
    analysis_results = []
    
    for _, row in tqdm(df_tiles.iterrows(), total=len(df_tiles), desc="CV Analysis"):
        geometry_score, edges, viz = detect_geometric_anomalies(row['rgb_path'])
        
        result = row.to_dict()
        result.update({
            'geometry_score': geometry_score,
            'is_candidate': geometry_score >= CONFIG['GEOMETRY_SCORE_THRESHOLD']
        })
        analysis_results.append(result)
    
    df_analysis = pd.DataFrame(analysis_results)
    df_candidates = df_analysis[df_analysis['is_candidate']].copy()
    
    print(f"CV filtering complete. Found {len(df_candidates)} candidates out of {len(df_analysis)} tiles.")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    candidates_path = CONFIG['RESULTS_DIR'] / f'cv_candidates_{timestamp}.csv'
    df_candidates.to_csv(candidates_path, index=False)
    
    return df_analysis, df_candidates

def create_composite_image(row):
    """Creates a composite image (RGB, NDVI, Slope) for LLM analysis."""
    # Load RGB
    rgb_img = cv2.imread(row['rgb_path'])
    
    # Load NDVI and Slope
    with rasterio.open(row['ndvi_path']) as src:
        ndvi_raw = src.read(1)
    with rasterio.open(row['slope_path']) as src:
        slope_raw = src.read(1)
    
    # Convert to color maps
    ndvi_color = (plt.cm.viridis(cv2.normalize(ndvi_raw, None, 0, 255, cv2.NORM_MINMAX))[:, :, :3] * 255).astype(np.uint8)
    slope_color = (plt.cm.magma(cv2.normalize(slope_raw, None, 0, 255, cv2.NORM_MINMAX))[:, :, :3] * 255).astype(np.uint8)
    
    # Create horizontal composite
    composite = np.hstack([
        cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB),
        ndvi_color,
        slope_color
    ])
    
    # Add labels
    cv2.putText(composite, 'RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(composite, 'NDVI', (522, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(composite, 'Slope', (1034, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return composite

def encode_image_base64(image_np):
    """Encode numpy image to base64 string."""
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpeg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def analyze_with_gpt4o(row):
    """Send candidate to GPT-4o for expert analysis with caching."""
    cache_file = CONFIG['CACHE_DIR'] / f"{row['tile_id']}.json"
    
    # Check cache first
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    
    try:
        # Create composite image
        composite_img = create_composite_image(row)
        base64_image = encode_image_base64(composite_img)
        
        # Expert prompt for archaeological analysis
        expert_prompt = (
            "You are an expert remote sensing archaeologist specializing in the Amazon Basin. "
            "Analyze the following composite image containing three panels: "
            "1. True-color RGB, 2. NDVI (vegetation index), 3. Slope (topography). "
            
            "Your task is to identify potential anthropogenic features such as "
            "geoglyphs, earthworks, or ancient settlements. "
            
            "Look for unnatural geometric patterns (straight lines, right angles, circles), "
            "unusual vegetation patterns, or subtle earthworks. "
            
            "Respond ONLY with a valid JSON object: "
            "{\"contains_anthropogenic_features\": boolean, "
            "\"confidence_score\": float (0.0-1.0), "
            "\"rationale\": string, "
            "\"feature_type_guess\": string}."
        )
        
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')
        )
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'deepseek/deepseek-r1-0528:free'),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": expert_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("API returned None content")
        
        result = json.loads(content)
        
        # Cache the result
        cache_file.write_text(json.dumps(result))
        return result
        
    except Exception as e:
        print(f"LLM analysis failed for tile {row['tile_id']}: {e}")
        return {
            "rationale": f"API call failed: {str(e)}",
            "confidence_score": 0.0,
            "contains_anthropogenic_features": False,
            "feature_type_guess": "error"
        }

def analyze_candidates_with_llm(df_candidates):
    """Analyze top candidates with GPT-4o."""
    if df_candidates.empty:
        print("No candidates to analyze with LLM.")
        return pd.DataFrame()
    
    num_to_analyze = min(CONFIG['LLM_ANALYSIS_COUNT'], len(df_candidates))
    print(f"Analyzing {num_to_analyze} candidates with GPT-4o...")
    
    llm_results = []
    
    for _, row in tqdm(df_candidates.head(num_to_analyze).iterrows(), 
                       total=num_to_analyze, desc="LLM Analysis"):
        analysis = analyze_with_gpt4o(row)
        llm_results.append({**row.to_dict(), **analysis})
    
    df_final = pd.DataFrame(llm_results)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_path = CONFIG['RESULTS_DIR'] / f'final_llm_analysis_{timestamp}.csv'
    df_final.to_csv(final_path, index=False)
    
    print(f"LLM analysis complete. Results saved to {final_path}")
    return df_final

def visualize_results(df_analysis, df_final):
    """Create visualizations of the analysis results."""
    print("Creating result visualizations...")
    
    # Show CV filtering examples
    if not df_analysis.empty:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('CV Filtering Results', fontsize=16)
        
        # Show passed and failed examples
        passed = df_analysis[df_analysis['is_candidate']].head(3)
        failed = df_analysis[~df_analysis['is_candidate']].head(3)
        
        for i, (label, data) in enumerate([('PASSED', passed), ('FAILED', failed)]):
            for j, (_, row) in enumerate(data.iterrows()):
                if j >= 3:
                    break
                    
                ax = axes[i, j]
                img = cv2.imread(row['rgb_path'])
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(f"{label}: {row['tile_id']}\nScore: {row['geometry_score']}")
                ax.axis('off')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(CONFIG['RESULTS_DIR'] / f'cv_filtering_results_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Show LLM analysis results
    if not df_final.empty:
        positive_results = df_final[df_final['contains_anthropogenic_features'] == True]
        
        if not positive_results.empty:
            print(f"\nFound {len(positive_results)} potential archaeological sites!")
            
            fig, axes = plt.subplots(len(positive_results), 1, figsize=(12, 4*len(positive_results)))
            if len(positive_results) == 1:
                axes = [axes]
            
            for i, (_, row) in enumerate(positive_results.iterrows()):
                composite = create_composite_image(row)
                axes[i].imshow(composite)
                axes[i].set_title(
                    f"DISCOVERY: {row['tile_id']} - {row['feature_type_guess']}\n"
                    f"Confidence: {row['confidence_score']:.2f} | {row['rationale'][:100]}..."
                )
                axes[i].axis('off')
            
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(CONFIG['RESULTS_DIR'] / f'llm_discoveries_{timestamp}.png', dpi=150, bbox_inches='tight')
            plt.show()
        else:
            print("No positive archaeological features identified by GPT-4o.")

def run_full_pipeline():
    """Execute the complete hybrid CV + LLM pipeline."""
    print("=" * 60)
    print("HYBRID CV + LLM ARCHAEOLOGICAL DISCOVERY PIPELINE")
    print("=" * 60)
    
    # Setup
    if not setup_environment():
        print("Environment setup failed. Exiting.")
        return
    
    # Stage 1: Data preparation
    print("\n--- STAGE 1: Data Preparation ---")
    if CONFIG['USE_MOCK_DATA']:
        source_path = create_mock_raster()
    else:
        print("Real satellite data pipeline not implemented in this demo.")
        return
    
    df_tiles = tile_raster_data(source_path)
    
    # Stage 2: CV filtering
    print("\n--- STAGE 2: Classical CV Filtering ---")
    df_analysis, df_candidates = filter_candidates_with_cv(df_tiles)
    
    # Stage 3: LLM analysis
    print("\n--- STAGE 3: GPT-4o Deep Analysis ---")
    df_final = analyze_candidates_with_llm(df_candidates)
    
    # Stage 4: Visualization
    print("\n--- STAGE 4: Results Visualization ---")
    visualize_results(df_analysis, df_final)
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Total tiles processed: {len(df_tiles) if not df_tiles.empty else 0}")
    print(f"CV candidates identified: {len(df_candidates) if not df_candidates.empty else 0}")
    print(f"LLM analyses performed: {len(df_final) if not df_final.empty else 0}")
    
    if not df_final.empty:
        positive_count = sum(df_final['contains_anthropogenic_features'])
        print(f"Potential archaeological sites found: {positive_count}")
    
    print(f"\nResults saved in: {CONFIG['RESULTS_DIR']}")

if __name__ == "__main__":
    run_full_pipeline()