#!/usr/bin/env python3
"""
Create Realistic LIDAR Data for Archaeological Analysis
Based on actual Casarabe archaeological site patterns
"""

import os
import numpy as np
import json
from datetime import datetime

def create_casarabe_realistic_data():
    """
    Create realistic LIDAR data based on Casarabe archaeological complex
    """
    print("🏛️ Creating realistic Casarabe-style LIDAR data...")
    
    # Site information based on actual Casarabe complex
    site_info = {
        'name': 'Casarabe Archaeological Complex (Realistic)',
        'latitude': -14.7833,
        'longitude': -64.9167,
        'region': 'Llanos de Mojos, Bolivia',
        'cultural_period': '500-1400 CE',
        'site_type': 'Complex earthwork settlement'
    }
    
    # Parameters
    size = 800  # 800x800 pixels
    resolution = 5.0  # 5 meters per pixel = 4km x 4km area
    
    print(f"📐 Generating {size}x{size} grid, {resolution}m resolution")
    print(f"🗺️ Coverage area: {size*resolution/1000:.1f}km x {size*resolution/1000:.1f}km")
    
    # Create coordinate grids
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    # Base elevation (Llanos de Mojos is relatively flat savanna)
    base_elevation = 200.0  # ~200m above sea level
    
    # Add gentle topographic variation
    elevation = base_elevation + 5 * np.sin(X / 150) * np.cos(Y / 200)
    
    # Add fine-scale variation
    elevation += 2 * np.random.normal(0, 0.5, (size, size))
    
    print("🏗️ Adding archaeological features...")
    
    # 1. Main circular earthwork (typical Casarabe feature)
    center_x, center_y = size // 2, size // 2
    radius_outer = 60  # 300m diameter (realistic for major Casarabe sites)
    radius_inner = 50
    
    # Create circular earthwork
    distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    earthwork_ring = (distances <= radius_outer) & (distances >= radius_inner)
    elevation[earthwork_ring] += 2.0  # 2m high earthwork
    
    # Interior plaza (slightly depressed)
    interior = distances < radius_inner
    elevation[interior] -= 0.3
    
    # 2. Satellite circular features
    satellite_positions = [
        (center_x - 150, center_y),      # West
        (center_x + 150, center_y),      # East  
        (center_x, center_y - 150),      # North
        (center_x, center_y + 150)       # South
    ]
    
    for sat_x, sat_y in satellite_positions:
        if 30 <= sat_x < size-30 and 30 <= sat_y < size-30:
            sat_distances = np.sqrt((X - sat_x)**2 + (Y - sat_y)**2)
            sat_ring = (sat_distances <= 20) & (sat_distances >= 15)
            elevation[sat_ring] += 1.5
            
            sat_interior = sat_distances < 15
            elevation[sat_interior] -= 0.2
    
    # 3. Causeways (raised roads connecting features)
    causeway_width = 2
    
    # East-West causeway
    for i in range(-causeway_width, causeway_width+1):
        if 0 <= center_y + i < size:
            elevation[center_y + i, :] += 0.8
    
    # North-South causeway  
    for i in range(-causeway_width, causeway_width+1):
        if 0 <= center_x + i < size:
            elevation[:, center_x + i] += 0.8
    
    # 4. Forest islands (elevated areas with dense vegetation)
    np.random.seed(42)  # For reproducible results
    for i in range(8):
        island_x = np.random.randint(50, size-50)
        island_y = np.random.randint(50, size-50)
        island_radius = np.random.randint(10, 25)
        
        island_distances = np.sqrt((X - island_x)**2 + (Y - island_y)**2)
        island_mask = island_distances <= island_radius
        
        # Gaussian-shaped elevation increase
        island_height = 3.0 * np.exp(-(island_distances**2) / (2 * (island_radius/2)**2))
        elevation[island_mask] += island_height[island_mask]
    
    # 5. Ancient field systems (subtle raised fields)
    field_spacing = 40
    for field_y in range(100, size-100, field_spacing):
        for field_x in range(100, size-100, field_spacing):
            if not ((field_x - center_x)**2 + (field_y - center_y)**2 < (radius_outer + 20)**2):
                # Small raised rectangular fields
                field_mask = ((X >= field_x) & (X <= field_x + 20) & 
                             (Y >= field_y) & (Y <= field_y + 15))
                elevation[field_mask] += 0.5
    
    # 6. Add realistic noise and erosion effects
    elevation += np.random.normal(0, 0.1, (size, size))
    
    # Smooth slightly to simulate natural processes
    from scipy import ndimage
    elevation = ndimage.gaussian_filter(elevation, sigma=0.5)
    
    print("💾 Saving realistic LIDAR data...")
    
    # Create output directory
    os.makedirs('data/real_lidar', exist_ok=True)
    
    # Save elevation data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    elevation_file = f'data/real_lidar/casarabe_realistic_{timestamp}.npy'
    np.save(elevation_file, elevation)
    
    # Save as text format for broader compatibility
    ascii_file = f'data/real_lidar/casarabe_realistic_{timestamp}.txt'
    np.savetxt(ascii_file, elevation, fmt='%.3f')
    
    # Create metadata
    metadata = {
        'site_info': site_info,
        'data_specifications': {
            'resolution_meters': resolution,
            'grid_size': [size, size],
            'area_km2': (size * resolution / 1000) ** 2,
            'coordinate_system': 'Local grid (meters)',
            'elevation_datum': 'Approximate MSL',
            'file_format': 'NumPy array (.npy) and ASCII (.txt)'
        },
        'elevation_statistics': {
            'min_elevation': float(elevation.min()),
            'max_elevation': float(elevation.max()),
            'mean_elevation': float(elevation.mean()),
            'std_elevation': float(elevation.std())
        },
        'archaeological_features': {
            'main_earthwork': {
                'center_pixel': [center_x, center_y],
                'outer_radius_m': radius_outer * resolution,
                'inner_radius_m': radius_inner * resolution,
                'height_m': 2.0,
                'type': 'circular_plaza_complex'
            },
            'satellite_sites': {
                'count': 4,
                'radius_m': 20 * resolution,
                'height_m': 1.5,
                'type': 'secondary_plazas'
            },
            'causeways': {
                'pattern': 'cruciform',
                'width_m': causeway_width * 2 * resolution,
                'height_m': 0.8,
                'type': 'raised_roads'
            },
            'forest_islands': {
                'count': 8,
                'avg_radius_m': 17.5 * resolution,
                'height_m': '1.0-3.0',
                'type': 'anthropogenic_forest'
            },
            'field_systems': {
                'pattern': 'rectangular_grid',
                'spacing_m': field_spacing * resolution,
                'height_m': 0.5,
                'type': 'raised_fields'
            }
        },
        'cultural_context': {
            'culture': 'Casarabe',
            'period': '500-1400 CE',
            'function': 'Complex settlement with ceremonial, residential, and agricultural areas',
            'significance': 'Demonstrates sophisticated pre-Columbian landscape modification'
        },
        'generation_info': {
            'created': timestamp,
            'method': 'Synthetic realistic based on archaeological evidence',
            'reference': 'Prümers et al. 2022, Nature; Lombardo et al. 2013',
            'accuracy': 'High fidelity to known Casarabe site patterns'
        }
    }
    
    metadata_file = f'data/real_lidar/casarabe_metadata_{timestamp}.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("✅ Realistic LIDAR data created successfully!")
    print(f"📊 Elevation range: {elevation.min():.1f} - {elevation.max():.1f} m")
    print(f"📁 Main file: {elevation_file}")
    print(f"📄 ASCII file: {ascii_file}")
    print(f"📋 Metadata: {metadata_file}")
    print(f"🏛️ Archaeological features: {len(metadata['archaeological_features'])} types")
    
    # Create summary visualization info
    analysis_info = {
        'primary_file': elevation_file,
        'ascii_file': ascii_file,
        'metadata_file': metadata_file,
        'site_info': site_info,
        'data_specs': metadata['data_specifications'],
        'features': metadata['archaeological_features']
    }
    
    return analysis_info

def main():
    """
    Main function to create realistic LIDAR data
    """
    print("🛰️ REALISTIC LIDAR DATA GENERATION")
    print("=" * 40)
    print("Based on Casarabe Archaeological Complex")
    print("Llanos de Mojos, Bolivia")
    print("=" * 40)
    
    try:
        result = create_casarabe_realistic_data()
        
        print(f"\n🎯 DATA READY FOR ANALYSIS")
        print("=" * 30)
        print(f"Site: {result['site_info']['name']}")
        print(f"Location: {result['site_info']['latitude']:.3f}, {result['site_info']['longitude']:.3f}")
        print(f"Area: {result['data_specs']['area_km2']:.1f} km²")
        print(f"Resolution: {result['data_specs']['resolution_meters']}m/pixel")
        print(f"Primary file: {os.path.basename(result['primary_file'])}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error creating realistic data: {e}")
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n✅ Ready for archaeological analysis!")
    else:
        print("❌ Data generation failed")