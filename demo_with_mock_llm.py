#!/usr/bin/env python3
"""
Demo script with mock LLM responses to show complete pipeline functionality
"""

import os
import sys
import json
import random
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/myuser/OpenAI_to_z_Challenge')
from hybrid_cv_llm_solution import *

def mock_gpt4o_analysis(row):
    """Mock GPT-4o analysis that simulates real LLM responses."""
    
    # Check if this tile has high geometry score (indicating our embedded anomalies)
    cache_file = CONFIG['CACHE_DIR'] / f"{row['tile_id']}.json"
    
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    
    # Simulate realistic responses based on geometry score
    geometry_score = row.get('geometry_score', 0)
    
    if geometry_score > 1000:  # High score indicates our embedded geometric features
        # Simulate positive detection
        responses = [
            {
                "contains_anthropogenic_features": True,
                "confidence_score": random.uniform(0.7, 0.95),
                "rationale": "Strong geometric patterns visible in RGB imagery suggest human-made structures. Rectangular formations and linear features indicate potential ancient settlement or agricultural terracing.",
                "feature_type_guess": "ancient settlement"
            },
            {
                "contains_anthropogenic_features": True, 
                "confidence_score": random.uniform(0.65, 0.85),
                "rationale": "Circular patterns in the vegetation index combined with geometric soil patterns suggest possible geoglyph or ceremonial earthwork construction.",
                "feature_type_guess": "geoglyph"
            },
            {
                "contains_anthropogenic_features": True,
                "confidence_score": random.uniform(0.6, 0.8),
                "rationale": "Linear features extending across the landscape with associated clearings indicate potential causeway or ancient road system.",
                "feature_type_guess": "causeway system"
            }
        ]
        result = random.choice(responses)
    else:
        # Simulate negative detection for low scores
        responses = [
            {
                "contains_anthropogenic_features": False,
                "confidence_score": random.uniform(0.1, 0.4),
                "rationale": "Patterns appear consistent with natural forest growth and river meanders. No clear geometric anomalies detected.",
                "feature_type_guess": "natural formation"
            },
            {
                "contains_anthropogenic_features": False,
                "confidence_score": random.uniform(0.15, 0.35),
                "rationale": "Vegetation patterns show natural variation typical of Amazon rainforest. Topographic features appear consistent with natural erosion processes.",
                "feature_type_guess": "natural terrain"
            }
        ]
        result = random.choice(responses)
    
    # Cache the result
    cache_file.write_text(json.dumps(result))
    return result

def run_demo_with_mock_llm():
    """Run complete pipeline demonstration with mock LLM responses."""
    print("=" * 70)
    print("HYBRID CV + LLM ARCHAEOLOGICAL DISCOVERY DEMO")
    print("(Using Mock LLM Responses)")
    print("=" * 70)
    
    # Setup environment
    setup_environment()
    
    # Stage 1: Data Preparation
    print("\n🗺️  STAGE 1: Generating Mock Amazon Satellite Data")
    print("-" * 50)
    source_path = create_mock_raster()
    df_tiles = tile_raster_data(source_path)
    print(f"✅ Created {len(df_tiles)} tiles from 2048x2048 pixel raster")
    
    # Stage 2: Classical CV Filtering  
    print("\n🔍 STAGE 2: Classical Computer Vision Analysis")
    print("-" * 50)
    df_analysis, df_candidates = filter_candidates_with_cv(df_tiles)
    print(f"✅ CV Analysis Complete:")
    print(f"   • Total tiles processed: {len(df_analysis)}")
    print(f"   • Candidates identified: {len(df_candidates)}")
    print(f"   • Filter efficiency: {(len(df_analysis) - len(df_candidates))/len(df_analysis)*100:.1f}% rejected")
    
    # Stage 3: Mock LLM Analysis
    print("\n🧠 STAGE 3: GPT-4o Deep Analysis (Mock)")
    print("-" * 50)
    
    # Analyze top candidates with mock LLM
    num_to_analyze = min(CONFIG['LLM_ANALYSIS_COUNT'], len(df_candidates))
    llm_results = []
    
    print(f"Analyzing {num_to_analyze} top candidates...")
    
    for i, (_, row) in enumerate(df_candidates.head(num_to_analyze).iterrows()):
        print(f"  📡 Analyzing {row['tile_id']} (geometry score: {row['geometry_score']})")
        
        # Create composite image (for demonstration)
        composite = create_composite_image(row)
        
        # Get mock analysis
        analysis = mock_gpt4o_analysis(row)
        
        result = {**row.to_dict(), **analysis}
        llm_results.append(result)
        
        # Show result
        status = "🎯 POSITIVE" if analysis['contains_anthropogenic_features'] else "❌ NEGATIVE"
        print(f"     {status} - {analysis['feature_type_guess']} (confidence: {analysis['confidence_score']:.2f})")
    
    df_final = pd.DataFrame(llm_results)
    
    # Save results
    final_path = CONFIG['RESULTS_DIR'] / 'demo_final_results.csv'
    df_final.to_csv(final_path, index=False)
    
    # Stage 4: Results Analysis
    print("\n📊 STAGE 4: Discovery Summary")
    print("-" * 50)
    
    positive_results = df_final[df_final['contains_anthropogenic_features'] == True]
    
    print(f"✅ Analysis Complete!")
    print(f"   • LLM analyses performed: {len(df_final)}")
    print(f"   • Potential archaeological sites discovered: {len(positive_results)}")
    
    if not positive_results.empty:
        print(f"\n🏛️  ARCHAEOLOGICAL DISCOVERIES:")
        for i, (_, discovery) in enumerate(positive_results.iterrows(), 1):
            print(f"   {i}. {discovery['tile_id']}")
            print(f"      Type: {discovery['feature_type_guess']}")
            print(f"      Confidence: {discovery['confidence_score']:.2f}")
            print(f"      Location: {discovery['lat']:.4f}°N, {discovery['lon']:.4f}°W")
            print(f"      Rationale: {discovery['rationale'][:80]}...")
            print()
    
    # Create final visualization
    print("🎨 Generating visualizations...")
    visualize_results(df_analysis, df_final)
    
    # Summary statistics
    total_area_km2 = (CONFIG['AOI_BOUNDS']['east'] - CONFIG['AOI_BOUNDS']['west']) * 111 * \
                     (CONFIG['AOI_BOUNDS']['north'] - CONFIG['AOI_BOUNDS']['south']) * 111
    
    print("\n📈 PIPELINE EFFICIENCY METRICS:")
    print(f"   • Area analyzed: {total_area_km2:.0f} km²")
    print(f"   • CV processing efficiency: {len(df_candidates)/len(df_analysis)*100:.1f}% candidates")
    print(f"   • LLM cost optimization: {len(df_final)/len(df_analysis)*100:.1f}% sent to LLM")
    print(f"   • Discovery rate: {len(positive_results)/len(df_final)*100:.1f}% positive detections")
    
    print("\n" + "=" * 70)
    print("🎉 DEMO COMPLETE - Archaeological discovery pipeline successfully demonstrated!")
    print("=" * 70)
    
    return df_final

if __name__ == "__main__":
    final_results = run_demo_with_mock_llm()