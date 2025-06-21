#!/usr/bin/env python3
"""
Real Satellite Data Integration for Archaeological Site Detection
Integrates multiple satellite data sources including Landsat, Sentinel, and Lidar
"""

import os
import requests
import json
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SatelliteDataManager:
    """Manages access to real satellite data sources"""
    
    def __init__(self):
        self.base_urls = {
            'landsat': 'https://landsatlook.usgs.gov/stac-browser',
            'sentinel': 'https://scihub.copernicus.eu/dhus',
            'nasa_earthdata': 'https://search.earthdata.nasa.gov',
            'usgs_eros': 'https://earthexplorer.usgs.gov',
            'planet': 'https://api.planet.com/data/v1'
        }
        self.apis = {
            'stac': 'https://earth-search.aws.element84.com/v1',
            'landsat_stac': 'https://landsatlook.usgs.gov/stac-server',
            'sentinel_stac': 'https://sentinel-cogs.s3.us-west-2.amazonaws.com'
        }
    
    def search_landsat_scenes(self, bbox: List[float], start_date: str, end_date: str, 
                             cloud_cover: float = 20.0) -> List[Dict]:
        """Search for Landsat scenes using STAC API"""
        try:
            search_url = f"{self.apis['stac']}/search"
            
            search_params = {
                "collections": ["landsat-c2-l2"],
                "bbox": bbox,
                "datetime": f"{start_date}/{end_date}",
                "query": {
                    "eo:cloud_cover": {"lt": cloud_cover}
                },
                "limit": 50
            }
            
            logger.info(f"Searching Landsat scenes for bbox: {bbox}")
            response = requests.post(search_url, json=search_params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                scenes = data.get('features', [])
                logger.info(f"Found {len(scenes)} Landsat scenes")
                return scenes
            else:
                logger.error(f"Landsat search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching Landsat scenes: {e}")
            return []
    
    def search_sentinel_scenes(self, bbox: List[float], start_date: str, end_date: str,
                              cloud_cover: float = 20.0) -> List[Dict]:
        """Search for Sentinel-2 scenes using STAC API"""
        try:
            search_url = f"{self.apis['stac']}/search"
            
            search_params = {
                "collections": ["sentinel-2-l2a"],
                "bbox": bbox,
                "datetime": f"{start_date}/{end_date}",
                "query": {
                    "eo:cloud_cover": {"lt": cloud_cover}
                },
                "limit": 50
            }
            
            logger.info(f"Searching Sentinel-2 scenes for bbox: {bbox}")
            response = requests.post(search_url, json=search_params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                scenes = data.get('features', [])
                logger.info(f"Found {len(scenes)} Sentinel-2 scenes")
                return scenes
            else:
                logger.error(f"Sentinel search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching Sentinel scenes: {e}")
            return []

class LidarDataManager:
    """Manages access to Lidar and elevation data"""
    
    def __init__(self):
        self.elevation_apis = {
            'usgs_ned': 'https://elevation.nationalmap.gov/arcgis/rest/services',
            'nasa_srtm': 'https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster',
            'opentopography': 'https://portal.opentopography.org/API'
        }
    
    def get_srtm_elevation(self, bbox: List[float]) -> Optional[np.ndarray]:
        """Download SRTM elevation data for the specified bounding box"""
        try:
            # SRTM data through OpenTopography API
            west, south, east, north = bbox
            
            api_url = "https://portal.opentopography.org/API/globaldem"
            params = {
                'demtype': 'SRTM30',  # 30m resolution SRTM
                'south': south,
                'north': north,
                'west': west,
                'east': east,
                'outputFormat': 'GTiff'
            }
            
            logger.info(f"Downloading SRTM elevation data for {bbox}")
            response = requests.get(api_url, params=params, timeout=60)
            
            if response.status_code == 200:
                # Save to temporary file and read with rasterio
                temp_file = f"temp_srtm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                
                with rasterio.open(temp_file) as src:
                    elevation_data = src.read(1)
                    transform = src.transform
                    crs = src.crs
                
                os.remove(temp_file)
                logger.info("SRTM elevation data downloaded successfully")
                return elevation_data
            else:
                logger.error(f"SRTM download failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading SRTM data: {e}")
            return None

class AmazonArchaeologyDataCollector:
    """Specialized collector for Amazon archaeological data"""
    
    def __init__(self):
        self.satellite_manager = SatelliteDataManager()
        self.lidar_manager = LidarDataManager()
        
        # Amazon region focus areas with known archaeological potential
        self.focus_regions = {
            'western_amazon': {
                'bbox': [-74.0, -12.0, -68.0, -6.0],  # Peru/Ecuador border
                'name': 'Western Amazon (Peru/Ecuador)',
                'priority': 'high'
            },
            'acre_rondonia': {
                'bbox': [-69.0, -12.0, -63.0, -8.0],  # Acre/Rondônia geoglyphs
                'name': 'Acre-Rondônia Geoglyph Region',
                'priority': 'high'
            },
            'central_amazon': {
                'bbox': [-65.0, -8.0, -58.0, -3.0],   # Central Brazilian Amazon
                'name': 'Central Amazon Basin',
                'priority': 'medium'
            },
            'upper_xingu': {
                'bbox': [-55.0, -15.0, -50.0, -10.0], # Upper Xingu region
                'name': 'Upper Xingu Cultural Area',
                'priority': 'high'
            }
        }
    
    def collect_region_data(self, region_key: str, days_back: int = 365) -> Dict:
        """Collect comprehensive satellite and elevation data for a region"""
        
        if region_key not in self.focus_regions:
            raise ValueError(f"Unknown region: {region_key}")
        
        region = self.focus_regions[region_key]
        bbox = region['bbox']
        
        # Date range for satellite imagery
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Collecting data for {region['name']}")
        logger.info(f"Bounding box: {bbox}")
        logger.info(f"Date range: {start_str} to {end_str}")
        
        # Collect satellite imagery
        landsat_scenes = self.satellite_manager.search_landsat_scenes(
            bbox, start_str, end_str, cloud_cover=15.0
        )
        
        sentinel_scenes = self.satellite_manager.search_sentinel_scenes(
            bbox, start_str, end_str, cloud_cover=15.0
        )
        
        # Collect elevation data
        elevation_data = self.lidar_manager.get_srtm_elevation(bbox)
        
        # Compile results
        collection_result = {
            'region_info': region,
            'bbox': bbox,
            'collection_date': datetime.now().isoformat(),
            'landsat_scenes': len(landsat_scenes),
            'sentinel_scenes': len(sentinel_scenes),
            'landsat_data': landsat_scenes[:5],  # Sample of scenes
            'sentinel_data': sentinel_scenes[:5],  # Sample of scenes
            'elevation_available': elevation_data is not None,
            'elevation_shape': elevation_data.shape if elevation_data is not None else None
        }
        
        return collection_result
    
    def analyze_best_scenes(self, collection_result: Dict) -> List[Dict]:
        """Analyze and rank the best scenes for archaeological analysis"""
        
        best_scenes = []
        
        # Process Landsat scenes
        for scene in collection_result.get('landsat_data', []):
            scene_info = {
                'platform': 'Landsat',
                'scene_id': scene.get('id', 'unknown'),
                'cloud_cover': scene.get('properties', {}).get('eo:cloud_cover', 100),
                'date': scene.get('properties', {}).get('datetime', 'unknown'),
                'download_links': self._extract_download_links(scene),
                'archaeological_score': self._calculate_archaeological_score(scene)
            }
            best_scenes.append(scene_info)
        
        # Process Sentinel scenes
        for scene in collection_result.get('sentinel_data', []):
            scene_info = {
                'platform': 'Sentinel-2',
                'scene_id': scene.get('id', 'unknown'),
                'cloud_cover': scene.get('properties', {}).get('eo:cloud_cover', 100),
                'date': scene.get('properties', {}).get('datetime', 'unknown'),
                'download_links': self._extract_download_links(scene),
                'archaeological_score': self._calculate_archaeological_score(scene)
            }
            best_scenes.append(scene_info)
        
        # Sort by archaeological score and cloud cover
        best_scenes.sort(key=lambda x: (x['archaeological_score'], -x['cloud_cover']), reverse=True)
        
        return best_scenes[:10]  # Return top 10 scenes
    
    def _extract_download_links(self, scene: Dict) -> Dict:
        """Extract download links from scene metadata"""
        assets = scene.get('assets', {})
        links = {}
        
        # Common asset types we're interested in
        asset_types = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22', 'thermal']
        
        for asset_type in asset_types:
            if asset_type in assets:
                links[asset_type] = assets[asset_type].get('href', '')
        
        return links
    
    def _calculate_archaeological_score(self, scene: Dict) -> float:
        """Calculate archaeological potential score for a scene"""
        score = 0.0
        properties = scene.get('properties', {})
        
        # Lower cloud cover is better
        cloud_cover = properties.get('eo:cloud_cover', 100)
        if cloud_cover < 5:
            score += 1.0
        elif cloud_cover < 15:
            score += 0.7
        elif cloud_cover < 30:
            score += 0.3
        
        # Recent imagery is preferred
        date_str = properties.get('datetime', '')
        if date_str:
            try:
                scene_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                days_old = (datetime.now().replace(tzinfo=scene_date.tzinfo) - scene_date).days
                if days_old < 30:
                    score += 0.5
                elif days_old < 90:
                    score += 0.3
            except:
                pass
        
        # Check for required bands
        assets = scene.get('assets', {})
        required_bands = ['red', 'green', 'blue', 'nir']
        available_bands = sum(1 for band in required_bands if band in assets)
        score += (available_bands / len(required_bands)) * 0.5
        
        return score

def run_real_data_archaeological_survey():
    """Execute comprehensive archaeological survey using real satellite data"""
    
    collector = AmazonArchaeologyDataCollector()
    
    # Survey all high-priority regions
    high_priority_regions = [
        'western_amazon', 'acre_rondonia', 'upper_xingu'
    ]
    
    survey_results = {}
    
    for region_key in high_priority_regions:
        logger.info(f"Starting survey of {region_key}")
        
        try:
            # Collect data for region
            collection_result = collector.collect_region_data(region_key)
            
            # Analyze best scenes
            best_scenes = collector.analyze_best_scenes(collection_result)
            
            survey_results[region_key] = {
                'collection_summary': collection_result,
                'best_scenes': best_scenes,
                'survey_status': 'completed'
            }
            
            # Log summary
            logger.info(f"Region {region_key} survey completed:")
            logger.info(f"  - Landsat scenes: {collection_result['landsat_scenes']}")
            logger.info(f"  - Sentinel scenes: {collection_result['sentinel_scenes']}")
            logger.info(f"  - Elevation data: {collection_result['elevation_available']}")
            logger.info(f"  - Best scenes identified: {len(best_scenes)}")
            
        except Exception as e:
            logger.error(f"Error surveying {region_key}: {e}")
            survey_results[region_key] = {
                'survey_status': 'failed',
                'error': str(e)
            }
    
    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'results/real_data_survey_{timestamp}.json'
    
    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(survey_results, f, indent=2, default=str)
    
    logger.info(f"Real data survey completed. Results saved to {output_file}")
    
    return survey_results

if __name__ == "__main__":
    results = run_real_data_archaeological_survey()
    
    # Print summary
    print("\n" + "="*60)
    print("REAL SATELLITE DATA ARCHAEOLOGICAL SURVEY")
    print("="*60)
    
    for region, data in results.items():
        if data['survey_status'] == 'completed':
            collection = data['collection_summary']
            print(f"\n{collection['region_info']['name']}:")
            print(f"  Landsat scenes: {collection['landsat_scenes']}")
            print(f"  Sentinel scenes: {collection['sentinel_scenes']}")
            print(f"  Elevation data: {collection['elevation_available']}")
            print(f"  Best scenes: {len(data['best_scenes'])}")
        else:
            print(f"\n{region}: Survey failed - {data.get('error', 'Unknown error')}")