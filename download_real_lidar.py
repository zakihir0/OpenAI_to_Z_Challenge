#!/usr/bin/env python3
"""
Real LIDAR Data Downloader for Archaeological Analysis
Downloads actual LIDAR data from public sources
"""

import os
import requests
import numpy as np
from urllib.parse import urlparse
import zipfile
import tarfile
import gzip
import json
from datetime import datetime

def download_usgs_lidar_sample():
    """
    Download sample LIDAR data from USGS or similar public source
    """
    print("🌐 Searching for public LIDAR data sources...")
    
    # Try multiple data sources
    data_sources = [
        {
            'name': 'OpenTopography Sample',
            'url': 'https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/COP30_hh/COP30_hh_03_02.tif',
            'type': 'geotiff',
            'description': 'Copernicus 30m DEM sample'
        },
        {
            'name': 'ASTER GDEM Sample',
            'url': 'https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/ASTGTM_hh/ASTGTM2_N00W067_dem.tif',
            'type': 'geotiff', 
            'description': 'ASTER Global DEM - Amazon region'
        }
    ]
    
    downloaded_files = []
    
    for source in data_sources:
        try:
            print(f"📥 Attempting to download: {source['name']}")
            
            # Create filename from URL
            filename = os.path.basename(urlparse(source['url']).path)
            if not filename:
                filename = f"{source['name'].replace(' ', '_').lower()}.tif"
            
            filepath = os.path.join('data/real_lidar', filename)
            
            # Download with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(source['url'], headers=headers, timeout=60, stream=True)
            
            if response.status_code == 200:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size = os.path.getsize(filepath)
                if file_size > 1000:  # At least 1KB
                    print(f"✅ Downloaded: {filename} ({file_size:,} bytes)")
                    downloaded_files.append({
                        'file': filepath,
                        'source': source,
                        'size': file_size
                    })
                else:
                    print(f"❌ Download failed: {filename} (too small)")
                    os.remove(filepath)
            else:
                print(f"❌ HTTP {response.status_code}: {source['name']}")
                
        except Exception as e:
            print(f"❌ Error downloading {source['name']}: {e}")
            continue
    
    return downloaded_files

def download_synthetic_realistic_data():
    """
    Generate realistic synthetic data based on actual archaeological sites
    """
    print("🔬 Generating realistic synthetic LIDAR data...")
    
    # Based on actual Casarabe archaeological site coordinates
    site_coords = {
        'name': 'Casarabe Archaeological Complex',
        'latitude': -14.7833,
        'longitude': -64.9167,
        'region': 'Llanos de Mojos, Bolivia'
    }
    
    # Generate realistic elevation data
    size = 1024  # Higher resolution
    x_range = np.linspace(0, 5000, size)  # 5km x 5km area
    y_range = np.linspace(0, 5000, size)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Base topography (Llanos de Mojos is relatively flat)
    base_elevation = 200 + 10 * np.sin(X/2000) * np.cos(Y/2500)  # Gentle undulation
    
    # Add realistic noise
    noise = np.random.normal(0, 0.5, (size, size))
    elevation = base_elevation + noise
    
    # Add archaeological features based on real Casarabe sites
    print("🏛️ Adding Casarabe-style archaeological features...")
    
    # Large circular earthwork (typical Casarabe feature)
    center_x, center_y = size//2, size//2
    radius = 80  # ~400m diameter (realistic for Casarabe)
    
    y_coords, x_coords = np.ogrid[:size, :size]
    circle_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
    ring_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2) & \
                ((x_coords - center_x)**2 + (y_coords - center_y)**2 >= (radius-8)**2)
    
    elevation[ring_mask] += 1.5  # Earthwork ring
    elevation[circle_mask] -= 0.3  # Slight depression inside
    
    # Smaller circular features (satellite sites)
    for i in range(4):
        angle = i * np.pi / 2
        sat_x = center_x + int(200 * np.cos(angle))
        sat_y = center_y + int(200 * np.sin(angle))
        
        if 0 <= sat_x < size and 0 <= sat_y < size:
            sat_radius = 25
            sat_mask = (x_coords - sat_x)**2 + (y_coords - sat_y)**2 <= sat_radius**2
            elevation[sat_mask] += 1.0
    
    # Causeway (raised road connecting features)
    # Horizontal causeway
    causeway_y = center_y
    causeway_mask = (y_coords >= causeway_y - 3) & (y_coords <= causeway_y + 3)
    elevation[causeway_mask] += 0.5
    
    # Vertical causeway  
    causeway_x = center_x
    causeway_mask_v = (x_coords >= causeway_x - 3) & (x_coords <= causeway_x + 3)
    elevation[causeway_mask_v] += 0.5
    
    # Forest islands (elevated areas with vegetation)
    for i in range(6):
        island_x = np.random.randint(100, size-100)
        island_y = np.random.randint(100, size-100)
        island_radius = np.random.randint(15, 40)
        
        island_mask = (x_coords - island_x)**2 + (y_coords - island_y)**2 <= island_radius**2
        island_height = 2.0 * np.exp(-((x_coords - island_x)**2 + (y_coords - island_y)**2) / (island_radius**2 / 4))
        elevation[island_mask] += island_height[island_mask]
    
    # Save as realistic format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as numpy array (can be converted to other formats)
    elevation_file = f'data/real_lidar/casarabe_realistic_dem_{timestamp}.npy'
    np.save(elevation_file, elevation)
    
    # Save metadata
    metadata = {
        'site_info': site_coords,
        'data_specs': {
            'resolution_m': 5.0,  # 5m per pixel
            'size_pixels': [size, size],
            'area_km2': 25.0,  # 5km x 5km
            'elevation_range': [float(elevation.min()), float(elevation.max())],
            'coordinate_system': 'WGS84',
            'source': 'Synthetic realistic data based on Casarabe archaeology'
        },
        'archaeological_features': {
            'main_earthwork': {
                'center_pixel': [center_x, center_y],
                'radius_m': 400,
                'type': 'circular_earthwork'
            },
            'satellite_sites': 4,
            'causeways': 'cross_pattern',
            'forest_islands': 6
        },
        'generation_date': timestamp
    }
    
    metadata_file = f'data/real_lidar/casarabe_metadata_{timestamp}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Generated realistic data: {elevation_file}")
    print(f"📄 Metadata: {metadata_file}")
    print(f"📐 Resolution: 5m/pixel, Area: 5km×5km")
    print(f"📊 Elevation range: {elevation.min():.1f} - {elevation.max():.1f}m")
    
    return [{
        'file': elevation_file,
        'metadata_file': metadata_file,
        'source': {
            'name': 'Casarabe Realistic Synthetic',
            'type': 'numpy_array',
            'description': 'Realistic synthetic LIDAR based on Casarabe archaeology'
        },
        'site_info': site_coords,
        'size': os.path.getsize(elevation_file)
    }]

def try_sample_las_download():
    """
    Try to download actual LAS sample files
    """
    print("🔍 Searching for LAS sample files...")
    
    # Sample LAS files from various sources
    las_sources = [
        {
            'name': 'Sample Point Cloud',
            'url': 'https://github.com/PDAL/data/raw/master/las/sample.las',
            'type': 'las'
        }
    ]
    
    downloaded = []
    
    for source in las_sources:
        try:
            print(f"📥 Downloading LAS: {source['name']}")
            
            filename = os.path.basename(urlparse(source['url']).path)
            filepath = os.path.join('data/real_lidar', filename)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            }
            
            response = requests.get(source['url'], headers=headers, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 1000:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                file_size = os.path.getsize(filepath)
                print(f"✅ Downloaded LAS: {filename} ({file_size:,} bytes)")
                
                downloaded.append({
                    'file': filepath,
                    'source': source,
                    'size': file_size
                })
            else:
                print(f"❌ Failed to download: {source['name']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return downloaded

def main():
    """
    Main function to acquire real LIDAR data
    """
    print("🛰️ REAL LIDAR DATA ACQUISITION")
    print("=" * 40)
    
    all_downloads = []
    
    # Try different data sources
    print("\n1. Attempting public DEM downloads...")
    dem_files = download_usgs_lidar_sample()
    all_downloads.extend(dem_files)
    
    print("\n2. Attempting LAS sample downloads...")
    las_files = try_sample_las_download()
    all_downloads.extend(las_files)
    
    print("\n3. Generating realistic synthetic data...")
    synthetic_files = download_synthetic_realistic_data()
    all_downloads.extend(synthetic_files)
    
    # Summary
    print(f"\n📊 DATA ACQUISITION SUMMARY")
    print("=" * 30)
    
    if all_downloads:
        print(f"✅ Total files acquired: {len(all_downloads)}")
        for i, download in enumerate(all_downloads, 1):
            source_name = download['source']['name']
            file_path = download['file']
            file_size = download.get('size', 0)
            print(f"  {i}. {source_name}")
            print(f"     File: {os.path.basename(file_path)}")
            print(f"     Size: {file_size:,} bytes")
            
        # Return the best file for analysis
        best_file = all_downloads[0]  # Use first successful download
        print(f"\n🎯 Selected for analysis: {best_file['source']['name']}")
        return best_file
    else:
        print("❌ No data files acquired")
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n✅ Ready for analysis with: {result['file']}")
    else:
        print("❌ Data acquisition failed")