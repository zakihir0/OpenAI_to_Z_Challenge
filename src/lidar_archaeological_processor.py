#!/usr/bin/env python3
"""
LIDAR Archaeological Site Detection System
Advanced point cloud processing for archaeological feature detection using LIDAR data

Implements the complete LIDAR archaeological analysis pipeline:
1. Point cloud processing (.las/.laz files)
2. DTM/DSM generation 
3. Hillshade and terrain visualization
4. Automated structure detection
5. Feature classification and analysis
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json

# Geospatial processing
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import pyproj

# LIDAR and point cloud processing
try:
    import laspy
    import open3d as o3d
    LIDAR_AVAILABLE = True
except ImportError:
    LIDAR_AVAILABLE = False
    logging.warning("LIDAR libraries not available - using synthetic data generation")

# Image processing and computer vision
import cv2
from scipy import ndimage
from skimage import filters, feature, segmentation, measure
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LidarArchaeologicalProcessor:
    """Main LIDAR archaeological processing engine"""
    
    def __init__(self, resolution: float = 0.5):
        """
        Initialize LIDAR processor
        
        Args:
            resolution: Grid resolution for DTM/DSM generation (meters)
        """
        self.resolution = resolution
        self.crs = pyproj.CRS.from_epsg(4326)  # WGS84
        self.processing_params = {
            'ground_classification_threshold': 2.0,
            'vegetation_height_threshold': 2.0,
            'structure_min_height': 0.3,
            'structure_max_height': 50.0,
            'noise_removal_radius': 1.0,
            'hillshade_azimuth': 315,
            'hillshade_altitude': 45
        }
        
    def process_lidar_file(self, lidar_path: str, output_dir: str = None) -> Dict:
        """
        Process LIDAR file for archaeological features
        
        Args:
            lidar_path: Path to .las or .laz file
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing analysis results
        """
        if output_dir is None:
            output_dir = os.path.dirname(lidar_path)
            
        logger.info(f"Processing LIDAR file: {lidar_path}")
        
        # Load point cloud
        point_cloud = self._load_point_cloud(lidar_path)
        if point_cloud is None:
            return self._generate_synthetic_analysis(lidar_path)
            
        # Extract DTM and DSM
        dtm, dsm = self._extract_elevation_models(point_cloud)
        
        # Generate hillshade
        hillshade = self._generate_hillshade(dtm)
        
        # Detect archaeological structures
        structures = self._detect_archaeological_structures(dtm, dsm)
        
        # Analyze terrain features
        terrain_analysis = self._analyze_terrain_features(dtm, hillshade)
        
        # Generate visualizations
        visualizations = self._create_visualizations(
            dtm, dsm, hillshade, structures, output_dir
        )
        
        # Compile comprehensive analysis
        analysis_results = {
            'file_info': {
                'input_file': lidar_path,
                'processing_time': datetime.now().isoformat(),
                'resolution': self.resolution,
                'bounds': self._get_bounds(point_cloud) if point_cloud else None
            },
            'elevation_models': {
                'dtm_stats': self._calculate_raster_stats(dtm),
                'dsm_stats': self._calculate_raster_stats(dsm),
                'canopy_height_stats': self._calculate_raster_stats(dsm - dtm)
            },
            'archaeological_structures': structures,
            'terrain_analysis': terrain_analysis,
            'visualizations': visualizations,
            'archaeological_score': self._calculate_archaeological_score(structures, terrain_analysis),
            'recommendations': self._generate_recommendations(structures, terrain_analysis)
        }
        
        # Save results
        self._save_analysis_results(analysis_results, output_dir)
        
        return analysis_results
    
    def _load_point_cloud(self, lidar_path: str) -> Optional[np.ndarray]:
        """Load LIDAR point cloud from file"""
        
        if not LIDAR_AVAILABLE:
            logger.warning("LIDAR libraries not available")
            return None
            
        try:
            # Load LAS/LAZ file
            las_file = laspy.read(lidar_path)
            
            # Extract coordinates
            points = np.vstack([
                las_file.x,
                las_file.y, 
                las_file.z,
                las_file.classification if hasattr(las_file, 'classification') else np.zeros(len(las_file.x)),
                las_file.intensity if hasattr(las_file, 'intensity') else np.zeros(len(las_file.x))
            ]).T
            
            logger.info(f"Loaded {len(points)} points from {lidar_path}")
            return points
            
        except Exception as e:
            logger.error(f"Failed to load LIDAR file {lidar_path}: {e}")
            return None
    
    def _extract_elevation_models(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract DTM (ground) and DSM (surface) from point cloud"""
        
        # Get bounds
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        x_max, y_max = points[:, 0].max(), points[:, 1].max()
        
        # Create grid
        x_cells = int((x_max - x_min) / self.resolution) + 1
        y_cells = int((y_max - y_min) / self.resolution) + 1
        
        # Initialize grids
        dtm = np.full((y_cells, x_cells), np.nan)
        dsm = np.full((y_cells, x_cells), np.nan)
        
        # Grid indices
        x_indices = ((points[:, 0] - x_min) / self.resolution).astype(int)
        y_indices = ((points[:, 1] - y_min) / self.resolution).astype(int)
        
        # Clip to valid range
        valid_mask = (
            (x_indices >= 0) & (x_indices < x_cells) &
            (y_indices >= 0) & (y_indices < y_cells)
        )
        
        if np.any(valid_mask):
            x_indices = x_indices[valid_mask]
            y_indices = y_indices[valid_mask]
            z_values = points[valid_mask, 2]
            classifications = points[valid_mask, 3] if points.shape[1] > 3 else np.zeros(np.sum(valid_mask))
            
            # Generate DSM (maximum elevation per cell)
            for i in range(len(x_indices)):
                y_idx, x_idx = y_indices[i], x_indices[i]
                if np.isnan(dsm[y_idx, x_idx]) or z_values[i] > dsm[y_idx, x_idx]:
                    dsm[y_idx, x_idx] = z_values[i]
            
            # Generate DTM (ground points only)
            ground_mask = (classifications == 2) | (classifications == 0)  # Ground classification
            if np.any(ground_mask):
                ground_x = x_indices[ground_mask]
                ground_y = y_indices[ground_mask]
                ground_z = z_values[ground_mask]
                
                for i in range(len(ground_x)):
                    y_idx, x_idx = ground_y[i], ground_x[i]
                    if np.isnan(dtm[y_idx, x_idx]) or ground_z[i] < dtm[y_idx, x_idx]:
                        dtm[y_idx, x_idx] = ground_z[i]
            else:
                # If no ground classification, use minimum elevation
                logger.warning("No ground classification found, using minimum elevation for DTM")
                for i in range(len(x_indices)):
                    y_idx, x_idx = y_indices[i], x_indices[i]
                    if np.isnan(dtm[y_idx, x_idx]) or z_values[i] < dtm[y_idx, x_idx]:
                        dtm[y_idx, x_idx] = z_values[i]
        
        # Fill gaps using interpolation
        dtm = self._fill_elevation_gaps(dtm)
        dsm = self._fill_elevation_gaps(dsm)
        
        logger.info(f"Generated DTM: {dtm.shape}, DSM: {dsm.shape}")
        return dtm, dsm
    
    def _fill_elevation_gaps(self, elevation_grid: np.ndarray) -> np.ndarray:
        """Fill gaps in elevation grid using interpolation"""
        
        # Create mask of valid data
        valid_mask = ~np.isnan(elevation_grid)
        
        if np.sum(valid_mask) == 0:
            logger.warning("No valid elevation data found")
            return elevation_grid
        
        # Use simple interpolation to fill gaps
        filled_grid = elevation_grid.copy()
        
        # Get coordinates of valid and invalid points
        y_coords, x_coords = np.mgrid[0:elevation_grid.shape[0], 0:elevation_grid.shape[1]]
        
        valid_points = np.column_stack([
            x_coords[valid_mask].ravel(),
            y_coords[valid_mask].ravel()
        ])
        valid_values = elevation_grid[valid_mask]
        
        # Find points that need interpolation
        invalid_mask = np.isnan(elevation_grid)
        if np.any(invalid_mask):
            invalid_points = np.column_stack([
                x_coords[invalid_mask].ravel(),
                y_coords[invalid_mask].ravel()
            ])
            
            # Use nearest neighbor interpolation for simplicity
            from scipy.spatial.distance import cdist
            distances = cdist(invalid_points, valid_points)
            nearest_indices = np.argmin(distances, axis=1)
            filled_grid[invalid_mask] = valid_values[nearest_indices]
        
        return filled_grid
    
    def _generate_hillshade(self, dtm: np.ndarray) -> np.ndarray:
        """Generate hillshade from DTM"""
        
        # Calculate gradients
        dy, dx = np.gradient(dtm)
        
        # Convert to radians
        azimuth_rad = np.radians(self.processing_params['hillshade_azimuth'])
        altitude_rad = np.radians(self.processing_params['hillshade_altitude'])
        
        # Calculate slope and aspect
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)
        
        # Calculate hillshade
        hillshade = np.sin(altitude_rad) * np.sin(slope) + \
                   np.cos(altitude_rad) * np.cos(slope) * \
                   np.cos(azimuth_rad - aspect)
        
        # Normalize to 0-255
        hillshade = ((hillshade + 1) * 127.5).astype(np.uint8)
        
        logger.info(f"Generated hillshade: {hillshade.shape}")
        return hillshade
    
    def _detect_archaeological_structures(self, dtm: np.ndarray, dsm: np.ndarray) -> Dict:
        """Detect potential archaeological structures in elevation data"""
        
        # Calculate canopy height model
        chm = dsm - dtm
        
        # Remove vegetation (areas with high CHM values)
        ground_surface = dtm.copy()
        vegetation_mask = chm > self.processing_params['vegetation_height_threshold']
        
        # Detect earthworks and structures
        structures = {
            'earthworks': self._detect_earthworks(ground_surface),
            'linear_features': self._detect_linear_features(ground_surface),
            'circular_features': self._detect_circular_features(ground_surface),
            'mounds': self._detect_mounds(ground_surface),
            'ditches': self._detect_ditches(ground_surface),
            'platforms': self._detect_platforms(ground_surface)
        }
        
        # Calculate summary statistics
        total_features = sum(len(features) for features in structures.values())
        
        structures['summary'] = {
            'total_features': total_features,
            'earthwork_count': len(structures['earthworks']),
            'linear_feature_count': len(structures['linear_features']),
            'circular_feature_count': len(structures['circular_features']),
            'mound_count': len(structures['mounds']),
            'ditch_count': len(structures['ditches']),
            'platform_count': len(structures['platforms'])
        }
        
        logger.info(f"Detected {total_features} potential archaeological structures")
        return structures
    
    def _detect_earthworks(self, elevation: np.ndarray) -> List[Dict]:
        """Detect earthwork features using edge detection and morphology"""
        
        # Apply Gaussian filter to reduce noise
        filtered = filters.gaussian(elevation, sigma=1.0)
        
        # Calculate local relief
        relief = filtered - ndimage.uniform_filter(filtered, size=20)
        
        # Threshold for significant elevation changes
        threshold = np.std(relief) * 1.5
        earthwork_mask = np.abs(relief) > threshold
        
        # Remove small features
        earthwork_mask = ndimage.binary_opening(earthwork_mask, structure=np.ones((3,3)))
        
        # Find connected components
        labeled_features = measure.label(earthwork_mask)
        
        earthworks = []
        for region in measure.regionprops(labeled_features):
            if region.area > 100:  # Minimum size threshold
                earthworks.append({
                    'centroid': region.centroid,
                    'area': region.area,
                    'bbox': region.bbox,
                    'relief_amplitude': float(np.max(relief[labeled_features == region.label]) - 
                                           np.min(relief[labeled_features == region.label])),
                    'eccentricity': region.eccentricity,
                    'type': 'earthwork'
                })
        
        return earthworks
    
    def _detect_linear_features(self, elevation: np.ndarray) -> List[Dict]:
        """Detect linear archaeological features (roads, canals, etc.)"""
        
        # Apply edge detection
        edges = feature.canny(elevation, sigma=2.0)
        
        # Use Hough line transform to detect linear features
        try:
            from skimage.transform import hough_line, hough_line_peaks
            
            # Get Hough transform
            tested_angles = np.linspace(-np.pi/2, np.pi/2, 180)
            h, theta, d = hough_line(edges, theta=tested_angles)
            
            # Find peaks
            peaks = hough_line_peaks(h, theta, d, min_distance=20, min_angle=10)
            
            linear_features = []
            for _, angle, dist in zip(*peaks):
                linear_features.append({
                    'angle': float(angle),
                    'distance': float(dist),
                    'strength': float(h[peaks[0][len(linear_features)], peaks[1][len(linear_features)]]),
                    'type': 'linear_feature'
                })
            
        except ImportError:
            # Fallback method using morphological operations
            linear_features = self._detect_linear_morphological(edges)
        
        return linear_features
    
    def _detect_linear_morphological(self, edges: np.ndarray) -> List[Dict]:
        """Fallback linear feature detection using morphological operations"""
        
        linear_features = []
        
        # Define linear structuring elements
        horizontal_kernel = np.array([[1, 1, 1, 1, 1]])
        vertical_kernel = np.array([[1], [1], [1], [1], [1]])
        diagonal1_kernel = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        diagonal2_kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        
        kernels = [
            ('horizontal', horizontal_kernel),
            ('vertical', vertical_kernel),
            ('diagonal1', diagonal1_kernel),
            ('diagonal2', diagonal2_kernel)
        ]
        
        for orientation, kernel in kernels:
            # Apply morphological opening
            opened = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            # Find connected components
            labeled = measure.label(opened)
            
            for region in measure.regionprops(labeled):
                if region.area > 50:  # Minimum length
                    linear_features.append({
                        'centroid': region.centroid,
                        'area': region.area,
                        'bbox': region.bbox,
                        'orientation': orientation,
                        'eccentricity': region.eccentricity,
                        'type': 'linear_feature'
                    })
        
        return linear_features
    
    def _detect_circular_features(self, elevation: np.ndarray) -> List[Dict]:
        """Detect circular archaeological features (ceremonial circles, etc.)"""
        
        # Apply Gaussian filter
        filtered = filters.gaussian(elevation, sigma=1.5)
        
        # Calculate local maxima and minima
        local_maxima = feature.peak_local_maxima(filtered, min_distance=20)
        local_minima = feature.peak_local_maxima(-filtered, min_distance=20)
        
        circular_features = []
        
        # Try Hough circle detection on edges
        edges = feature.canny(elevation, sigma=2.0)
        try:
            from skimage.transform import hough_circle, hough_circle_peaks
            
            # Define range of radii to search
            radii = np.arange(5, 50, 2)
            hough_res = hough_circle(edges, radii)
            
            # Find circular features
            accums, cx, cy, radii_found = hough_circle_peaks(
                hough_res, radii, min_xdistance=20, min_ydistance=20
            )
            
            for center_y, center_x, radius, accum in zip(cy, cx, radii_found, accums):
                circular_features.append({
                    'center': (center_y, center_x),
                    'radius': float(radius),
                    'strength': float(accum),
                    'type': 'circular_feature'
                })
                
        except ImportError:
            logger.warning("Hough circle detection not available, using template matching")
            circular_features = self._detect_circular_template_matching(elevation)
        
        return circular_features
    
    def _detect_circular_template_matching(self, elevation: np.ndarray) -> List[Dict]:
        """Fallback circular feature detection using template matching"""
        
        circular_features = []
        
        # Create circular templates of different sizes
        for radius in range(5, 30, 5):
            # Create circular template
            template = np.zeros((radius*2+1, radius*2+1))
            y, x = np.ogrid[:radius*2+1, :radius*2+1]
            center = radius
            mask = (x - center)**2 + (y - center)**2 <= radius**2
            template[mask] = 1
            
            # Template matching
            result = cv2.matchTemplate(elevation.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            
            # Find peaks
            threshold = 0.6
            locations = np.where(result >= threshold)
            
            for y, x in zip(*locations):
                circular_features.append({
                    'center': (y + radius, x + radius),
                    'radius': float(radius),
                    'strength': float(result[y, x]),
                    'type': 'circular_feature'
                })
        
        return circular_features
    
    def _detect_mounds(self, elevation: np.ndarray) -> List[Dict]:
        """Detect mound features"""
        
        # Calculate local relief
        filtered = filters.gaussian(elevation, sigma=2.0)
        local_mean = ndimage.uniform_filter(filtered, size=30)
        relief = filtered - local_mean
        
        # Find positive relief (mounds)
        mound_threshold = np.std(relief) * 2.0
        mound_mask = relief > mound_threshold
        
        # Clean up mask
        mound_mask = ndimage.binary_opening(mound_mask, structure=np.ones((5,5)))
        mound_mask = ndimage.binary_closing(mound_mask, structure=np.ones((3,3)))
        
        # Find mound features
        labeled_mounds = measure.label(mound_mask)
        
        mounds = []
        for region in measure.regionprops(labeled_mounds):
            if region.area > 25:  # Minimum size
                # Calculate height
                mound_pixels = relief[labeled_mounds == region.label]
                height = np.max(mound_pixels)
                
                mounds.append({
                    'centroid': region.centroid,
                    'area': region.area,
                    'height': float(height),
                    'bbox': region.bbox,
                    'eccentricity': region.eccentricity,
                    'type': 'mound'
                })
        
        return mounds
    
    def _detect_ditches(self, elevation: np.ndarray) -> List[Dict]:
        """Detect ditch/canal features"""
        
        # Calculate local relief
        filtered = filters.gaussian(elevation, sigma=2.0)
        local_mean = ndimage.uniform_filter(filtered, size=30)
        relief = filtered - local_mean
        
        # Find negative relief (ditches)
        ditch_threshold = -np.std(relief) * 2.0
        ditch_mask = relief < ditch_threshold
        
        # Clean up mask
        ditch_mask = ndimage.binary_opening(ditch_mask, structure=np.ones((3,3)))
        ditch_mask = ndimage.binary_closing(ditch_mask, structure=np.ones((5,5)))
        
        # Find ditch features
        labeled_ditches = measure.label(ditch_mask)
        
        ditches = []
        for region in measure.regionprops(labeled_ditches):
            if region.area > 25:  # Minimum size
                # Calculate depth
                ditch_pixels = relief[labeled_ditches == region.label]
                depth = abs(np.min(ditch_pixels))
                
                ditches.append({
                    'centroid': region.centroid,
                    'area': region.area,
                    'depth': float(depth),
                    'bbox': region.bbox,
                    'eccentricity': region.eccentricity,
                    'type': 'ditch'
                })
        
        return ditches
    
    def _detect_platforms(self, elevation: np.ndarray) -> List[Dict]:
        """Detect platform/terrace features"""
        
        # Calculate slope
        dy, dx = np.gradient(elevation)
        slope = np.sqrt(dx**2 + dy**2)
        
        # Find flat areas (potential platforms)
        slope_threshold = np.percentile(slope, 20)  # Bottom 20% of slopes
        flat_mask = slope < slope_threshold
        
        # Remove very small flat areas
        flat_mask = ndimage.binary_opening(flat_mask, structure=np.ones((5,5)))
        
        # Find connected flat areas
        labeled_platforms = measure.label(flat_mask)
        
        platforms = []
        for region in measure.regionprops(labeled_platforms):
            if region.area > 100:  # Minimum size for platforms
                # Check if it's elevated relative to surroundings
                platform_pixels = elevation[labeled_platforms == region.label]
                mean_elevation = np.mean(platform_pixels)
                
                # Get surrounding area
                bbox = region.bbox
                y1, x1, y2, x2 = bbox
                
                # Expand bounding box to get surroundings
                buffer = 10
                y1_buf = max(0, y1 - buffer)
                x1_buf = max(0, x1 - buffer)
                y2_buf = min(elevation.shape[0], y2 + buffer)
                x2_buf = min(elevation.shape[1], x2 + buffer)
                
                surrounding_area = elevation[y1_buf:y2_buf, x1_buf:x2_buf]
                surrounding_mean = np.mean(surrounding_area)
                
                elevation_difference = mean_elevation - surrounding_mean
                
                if elevation_difference > 0.5:  # Platform must be elevated
                    platforms.append({
                        'centroid': region.centroid,
                        'area': region.area,
                        'elevation': float(mean_elevation),
                        'elevation_difference': float(elevation_difference),
                        'bbox': region.bbox,
                        'type': 'platform'
                    })
        
        return platforms
    
    def _analyze_terrain_features(self, dtm: np.ndarray, hillshade: np.ndarray) -> Dict:
        """Analyze general terrain characteristics"""
        
        # Calculate slope and aspect
        dy, dx = np.gradient(dtm)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        aspect = np.degrees(np.arctan2(-dx, dy))
        
        # Calculate curvature
        dyy, dyx = np.gradient(dy)
        dxy, dxx = np.gradient(dx)
        curvature = (dxx + dyy) / (1 + dx**2 + dy**2)**(3/2)
        
        # Terrain analysis
        terrain_stats = {
            'elevation_stats': {
                'min': float(np.min(dtm)),
                'max': float(np.max(dtm)),
                'mean': float(np.mean(dtm)),
                'std': float(np.std(dtm)),
                'range': float(np.max(dtm) - np.min(dtm))
            },
            'slope_stats': {
                'mean': float(np.mean(slope)),
                'std': float(np.std(slope)),
                'steep_areas_percent': float(np.sum(slope > 30) / slope.size * 100)
            },
            'aspect_stats': {
                'north_facing_percent': float(np.sum((aspect > 315) | (aspect < 45)) / aspect.size * 100),
                'south_facing_percent': float(np.sum((aspect > 135) & (aspect < 225)) / aspect.size * 100)
            },
            'curvature_stats': {
                'mean': float(np.mean(curvature)),
                'std': float(np.std(curvature)),
                'convex_areas_percent': float(np.sum(curvature > 0) / curvature.size * 100)
            },
            'terrain_complexity': self._calculate_terrain_complexity(dtm, slope, curvature)
        }
        
        return terrain_stats
    
    def _calculate_terrain_complexity(self, dtm: np.ndarray, slope: np.ndarray, curvature: np.ndarray) -> float:
        """Calculate overall terrain complexity score"""
        
        # Combine multiple terrain metrics
        elevation_variability = np.std(dtm) / np.mean(dtm) if np.mean(dtm) > 0 else 0
        slope_variability = np.std(slope) / np.mean(slope) if np.mean(slope) > 0 else 0
        curvature_variability = np.std(curvature)
        
        # Normalize and combine
        complexity_score = (
            elevation_variability * 0.4 +
            slope_variability * 0.4 +
            curvature_variability * 0.2
        )
        
        return float(min(complexity_score, 1.0))
    
    def _calculate_archaeological_score(self, structures: Dict, terrain: Dict) -> float:
        """Calculate overall archaeological potential score"""
        
        score = 0.0
        
        # Structure density scoring (50% of total)
        total_features = structures['summary']['total_features']
        if total_features > 0:
            structure_score = min(total_features / 50.0, 0.5)  # Max 0.5 for structures
            score += structure_score
            
            # Bonus for diverse feature types
            feature_types = sum([
                1 for count in [
                    structures['summary']['earthwork_count'],
                    structures['summary']['linear_feature_count'], 
                    structures['summary']['circular_feature_count'],
                    structures['summary']['mound_count']
                ] if count > 0
            ])
            diversity_bonus = feature_types * 0.05
            score += diversity_bonus
        
        # Terrain characteristics (30% of total)
        complexity = terrain['terrain_complexity']
        terrain_score = complexity * 0.3
        score += terrain_score
        
        # Archaeological indicators (20% of total)
        elevation_range = terrain['elevation_stats']['range']
        if elevation_range > 10:  # Significant topographical variation
            score += 0.1
        
        if terrain['slope_stats']['steep_areas_percent'] < 70:  # Not too steep for habitation
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_recommendations(self, structures: Dict, terrain: Dict) -> List[str]:
        """Generate archaeological investigation recommendations"""
        
        recommendations = []
        total_features = structures['summary']['total_features']
        score = self._calculate_archaeological_score(structures, terrain)
        
        if score > 0.7:
            recommendations.extend([
                "HIGH PRIORITY: Immediate archaeological field survey recommended",
                "Deploy ground-penetrating radar to confirm subsurface features",
                "Coordinate with local archaeological authorities",
                "Consider drone-based high-resolution mapping"
            ])
        elif score > 0.4:
            recommendations.extend([
                "MEDIUM PRIORITY: Detailed remote sensing analysis recommended",
                "Acquire higher resolution LIDAR data if available",
                "Compare with historical aerial photography",
                "Consult regional archaeological databases"
            ])
        else:
            recommendations.extend([
                "Continue monitoring with periodic LIDAR surveys",
                "Investigate similar terrain patterns in region",
                "Review for potential natural geological features"
            ])
        
        # Feature-specific recommendations
        if structures['summary']['circular_feature_count'] > 3:
            recommendations.append("Investigate circular features for ceremonial/ritual significance")
        
        if structures['summary']['linear_feature_count'] > 5:
            recommendations.append("Map linear feature network for ancient transportation routes")
        
        if structures['summary']['mound_count'] > 2:
            recommendations.append("Analyze mound features for settlement or burial significance")
        
        if structures['summary']['platform_count'] > 1:
            recommendations.append("Investigate platform features for habitation or ceremonial use")
        
        return recommendations
    
    def _create_visualizations(self, dtm: np.ndarray, dsm: np.ndarray, 
                             hillshade: np.ndarray, structures: Dict, 
                             output_dir: str) -> Dict:
        """Create comprehensive visualizations"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_files = {}
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Hillshade visualization
            plt.figure(figsize=(12, 10))
            plt.imshow(hillshade, cmap='gray')
            plt.title('Hillshade Analysis - Archaeological Features')
            plt.colorbar(label='Hillshade Intensity')
            
            # Overlay detected features
            self._overlay_structures_on_plot(structures)
            
            hillshade_file = os.path.join(output_dir, f'hillshade_analysis_{timestamp}.png')
            plt.savefig(hillshade_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files['hillshade'] = hillshade_file
            
            # 2. DTM with contours
            plt.figure(figsize=(12, 10))
            plt.imshow(dtm, cmap='terrain')
            plt.contour(dtm, levels=20, colors='black', alpha=0.5, linewidths=0.5)
            plt.title('Digital Terrain Model with Contours')
            plt.colorbar(label='Elevation (m)')
            
            # Overlay structures
            self._overlay_structures_on_plot(structures)
            
            dtm_file = os.path.join(output_dir, f'dtm_contours_{timestamp}.png')
            plt.savefig(dtm_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files['dtm_contours'] = dtm_file
            
            # 3. Canopy Height Model
            chm = dsm - dtm
            plt.figure(figsize=(12, 10))
            plt.imshow(chm, cmap='viridis')
            plt.title('Canopy Height Model')
            plt.colorbar(label='Height Above Ground (m)')
            
            chm_file = os.path.join(output_dir, f'canopy_height_{timestamp}.png')
            plt.savefig(chm_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files['canopy_height'] = chm_file
            
            # 4. Archaeological Features Overview
            plt.figure(figsize=(15, 12))
            plt.imshow(hillshade, cmap='gray', alpha=0.7)
            
            # Plot different feature types with different colors
            colors = {
                'earthworks': 'red',
                'linear_features': 'blue', 
                'circular_features': 'green',
                'mounds': 'orange',
                'ditches': 'purple',
                'platforms': 'yellow'
            }
            
            for feature_type, features in structures.items():
                if feature_type != 'summary' and features:
                    color = colors.get(feature_type, 'black')
                    self._plot_features(features, color, feature_type.replace('_', ' ').title())
            
            plt.title(f'Archaeological Features Overview - {structures["summary"]["total_features"]} Features Detected')
            plt.legend()
            
            overview_file = os.path.join(output_dir, f'archaeological_overview_{timestamp}.png')
            plt.savefig(overview_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files['archaeological_overview'] = overview_file
            
            logger.info(f"Created {len(viz_files)} visualization files")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            viz_files['error'] = str(e)
        
        return viz_files
    
    def _overlay_structures_on_plot(self, structures: Dict):
        """Overlay archaeological structures on current plot"""
        
        colors = {
            'earthworks': 'red',
            'linear_features': 'blue',
            'circular_features': 'green', 
            'mounds': 'orange',
            'ditches': 'purple',
            'platforms': 'yellow'
        }
        
        for feature_type, features in structures.items():
            if feature_type != 'summary' and features:
                color = colors.get(feature_type, 'black')
                for feature in features[:5]:  # Limit to top 5 features per type
                    if 'centroid' in feature:
                        y, x = feature['centroid']
                        plt.plot(x, y, 'o', color=color, markersize=6, alpha=0.8)
                    elif 'center' in feature:
                        y, x = feature['center'] 
                        plt.plot(x, y, 'o', color=color, markersize=6, alpha=0.8)
    
    def _plot_features(self, features: List[Dict], color: str, label: str):
        """Plot features with specific color and label"""
        
        if not features:
            return
            
        x_coords, y_coords = [], []
        
        for feature in features:
            if 'centroid' in feature:
                y, x = feature['centroid']
                x_coords.append(x)
                y_coords.append(y)
            elif 'center' in feature:
                y, x = feature['center']
                x_coords.append(x) 
                y_coords.append(y)
        
        if x_coords and y_coords:
            plt.scatter(x_coords, y_coords, c=color, label=f'{label} ({len(features)})', 
                       s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    def _save_analysis_results(self, results: Dict, output_dir: str):
        """Save analysis results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = os.path.join(output_dir, f'lidar_analysis_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        report_file = os.path.join(output_dir, f'archaeological_report_{timestamp}.txt')
        with open(report_file, 'w') as f:
            self._write_analysis_report(f, results)
        
        logger.info(f"Saved analysis results to {json_file} and {report_file}")
    
    def _write_analysis_report(self, file, results: Dict):
        """Write comprehensive analysis report"""
        
        file.write("LIDAR ARCHAEOLOGICAL ANALYSIS REPORT\n")
        file.write("=" * 50 + "\n\n")
        
        # File information
        file_info = results['file_info']
        file.write(f"Input File: {file_info['input_file']}\n")
        file.write(f"Processing Time: {file_info['processing_time']}\n")
        file.write(f"Resolution: {file_info['resolution']} meters\n\n")
        
        # Archaeological features summary
        summary = results['archaeological_structures']['summary']
        file.write("ARCHAEOLOGICAL FEATURES DETECTED\n")
        file.write("-" * 35 + "\n")
        file.write(f"Total Features: {summary['total_features']}\n")
        file.write(f"Earthworks: {summary['earthwork_count']}\n")
        file.write(f"Linear Features: {summary['linear_feature_count']}\n")
        file.write(f"Circular Features: {summary['circular_feature_count']}\n")
        file.write(f"Mounds: {summary['mound_count']}\n")
        file.write(f"Ditches: {summary['ditch_count']}\n")
        file.write(f"Platforms: {summary['platform_count']}\n\n")
        
        # Archaeological score and recommendations
        score = results['archaeological_score']
        file.write(f"ARCHAEOLOGICAL POTENTIAL SCORE: {score:.3f}\n")
        file.write(f"CONFIDENCE LEVEL: {'HIGH' if score > 0.7 else 'MEDIUM' if score > 0.4 else 'LOW'}\n\n")
        
        file.write("RECOMMENDATIONS\n")
        file.write("-" * 15 + "\n")
        for i, rec in enumerate(results['recommendations'], 1):
            file.write(f"{i}. {rec}\n")
        
        file.write("\n")
        
        # Terrain analysis
        terrain = results['terrain_analysis']
        file.write("TERRAIN CHARACTERISTICS\n")
        file.write("-" * 22 + "\n")
        elev_stats = terrain['elevation_stats']
        file.write(f"Elevation Range: {elev_stats['min']:.1f} - {elev_stats['max']:.1f} m\n")
        file.write(f"Mean Elevation: {elev_stats['mean']:.1f} m\n")
        file.write(f"Terrain Complexity: {terrain['terrain_complexity']:.3f}\n")
        file.write(f"Mean Slope: {terrain['slope_stats']['mean']:.1f}°\n")
        file.write(f"Steep Areas: {terrain['slope_stats']['steep_areas_percent']:.1f}%\n\n")
    
    def _generate_synthetic_analysis(self, lidar_path: str) -> Dict:
        """Generate synthetic analysis when LIDAR libraries are unavailable"""
        
        logger.warning("Generating synthetic LIDAR analysis for demonstration")
        
        # Create synthetic terrain
        size = 512
        x, y = np.meshgrid(np.linspace(0, 100, size), np.linspace(0, 100, size))
        
        # Base terrain with hills and valleys
        terrain = (
            20 * np.sin(x / 10) * np.cos(y / 15) +
            10 * np.sin(x / 5) * np.sin(y / 8) +
            5 * np.random.randn(size, size)
        )
        terrain = ndimage.gaussian_filter(terrain, sigma=2.0)
        
        # Add synthetic archaeological features
        synthetic_structures = self._add_synthetic_archaeological_features(terrain)
        
        # Generate hillshade
        hillshade = self._generate_hillshade(terrain)
        
        # Analyze synthetic terrain
        terrain_analysis = self._analyze_terrain_features(terrain, hillshade)
        
        return {
            'file_info': {
                'input_file': lidar_path,
                'processing_time': datetime.now().isoformat(),
                'resolution': self.resolution,
                'synthetic_data': True
            },
            'elevation_models': {
                'dtm_stats': self._calculate_raster_stats(terrain),
                'dsm_stats': self._calculate_raster_stats(terrain + np.random.rand(*terrain.shape) * 5),
                'canopy_height_stats': self._calculate_raster_stats(np.random.rand(*terrain.shape) * 5)
            },
            'archaeological_structures': synthetic_structures,
            'terrain_analysis': terrain_analysis,
            'archaeological_score': 0.6,  # Synthetic moderate score
            'recommendations': [
                "SYNTHETIC DATA: Acquire real LIDAR data for accurate analysis",
                "Install required LIDAR processing libraries (laspy, pdal, open3d)",
                "This demonstration shows the analysis pipeline capabilities"
            ]
        }
    
    def _add_synthetic_archaeological_features(self, terrain: np.ndarray) -> Dict:
        """Add synthetic archaeological features to terrain"""
        
        # Add circular earthworks
        center_y, center_x = 150, 200
        radius = 30
        y, x = np.ogrid[:terrain.shape[0], :terrain.shape[1]]
        circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        terrain[circle_mask] += 2.0  # Raise circular area
        
        # Add linear feature (ancient road)
        cv2.line(terrain, (100, 50), (400, 350), 1.5, thickness=5)
        
        # Add mound
        mound_y, mound_x = 300, 350
        mound_radius = 20
        mound_mask = (x - mound_x)**2 + (y - mound_y)**2 <= mound_radius**2
        mound_height = 3.0 * np.exp(-((x - mound_x)**2 + (y - mound_y)**2) / (mound_radius**2 / 2))
        terrain[mound_mask] += mound_height[mound_mask]
        
        return {
            'earthworks': [{'centroid': (center_y, center_x), 'area': np.pi * radius**2, 'type': 'earthwork'}],
            'linear_features': [{'angle': 0.7, 'distance': 250, 'type': 'linear_feature'}],
            'circular_features': [{'center': (center_y, center_x), 'radius': radius, 'type': 'circular_feature'}],
            'mounds': [{'centroid': (mound_y, mound_x), 'area': np.pi * mound_radius**2, 'height': 3.0, 'type': 'mound'}],
            'ditches': [],
            'platforms': [],
            'summary': {
                'total_features': 3,
                'earthwork_count': 1,
                'linear_feature_count': 1,
                'circular_feature_count': 1,
                'mound_count': 1,
                'ditch_count': 0,
                'platform_count': 0
            }
        }
    
    def _calculate_raster_stats(self, raster: np.ndarray) -> Dict:
        """Calculate statistics for raster data"""
        
        return {
            'min': float(np.min(raster)),
            'max': float(np.max(raster)),
            'mean': float(np.mean(raster)),
            'std': float(np.std(raster)),
            'percentile_25': float(np.percentile(raster, 25)),
            'percentile_75': float(np.percentile(raster, 75))
        }
    
    def _get_bounds(self, points: np.ndarray) -> Dict:
        """Get bounding box of point cloud"""
        
        return {
            'x_min': float(np.min(points[:, 0])),
            'x_max': float(np.max(points[:, 0])),
            'y_min': float(np.min(points[:, 1])),
            'y_max': float(np.max(points[:, 1])),
            'z_min': float(np.min(points[:, 2])),
            'z_max': float(np.max(points[:, 2]))
        }


def main():
    """Main function for testing LIDAR processor"""
    
    # Initialize processor
    processor = LidarArchaeologicalProcessor(resolution=0.5)
    
    # Example usage
    test_file = "/path/to/lidar_data.las"  # Replace with actual LIDAR file
    
    if os.path.exists(test_file):
        results = processor.process_lidar_file(test_file)
        print(f"Analysis complete. Archaeological score: {results['archaeological_score']:.3f}")
        print(f"Total features detected: {results['archaeological_structures']['summary']['total_features']}")
    else:
        print("No LIDAR file found. Generating synthetic analysis...")
        results = processor._generate_synthetic_analysis("synthetic_data.las")
        print(f"Synthetic analysis complete. Score: {results['archaeological_score']:.3f}")


if __name__ == "__main__":
    main()