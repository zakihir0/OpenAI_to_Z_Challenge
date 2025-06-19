# Satellite Data Processing for Archaeological Site Detection
# Implementation for the OpenAI to Z Challenge

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
import os

# Geospatial and remote sensing libraries
try:
    import rasterio
    from rasterio.plot import show
    from rasterio.mask import mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, box
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# Image processing libraries
from scipy import ndimage
from scipy.signal import convolve2d
from skimage import feature, measure, morphology, filters, segmentation
from skimage.util import img_as_float
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Machine learning for classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class SatelliteImageProcessor:
    """Process satellite imagery for archaeological feature detection"""
    
    def __init__(self):
        self.bands = {}
        self.metadata = {}
        self.indices = {}
        
    def load_image_data(self, file_path: Optional[str] = None, mock_data: bool = True) -> Dict[str, np.ndarray]:
        """Load satellite image data from file or generate mock data"""
        
        if mock_data or not RASTERIO_AVAILABLE:
            # Generate realistic mock satellite data
            height, width = 1000, 1000
            
            # Simulate different spectral bands
            bands = {
                'blue': self._generate_mock_band(height, width, base_value=0.1),
                'green': self._generate_mock_band(height, width, base_value=0.15),
                'red': self._generate_mock_band(height, width, base_value=0.2),
                'nir': self._generate_mock_band(height, width, base_value=0.4),
                'swir1': self._generate_mock_band(height, width, base_value=0.25),
                'swir2': self._generate_mock_band(height, width, base_value=0.15)
            }
            
            # Add some archaeological features to mock data
            bands = self._add_mock_archaeological_features(bands)
            
            self.bands = bands
            self.metadata = {
                'height': height,
                'width': width,
                'crs': 'EPSG:4326',
                'transform': None,
                'acquisition_date': '2024-01-15',
                'satellite': 'Mock Satellite',
                'resolution': 30  # meters
            }
            
            return bands
        
        else:
            # Load actual satellite data using rasterio
            try:
                with rasterio.open(file_path) as src:
                    bands = {}
                    for i in range(1, src.count + 1):
                        bands[f'band_{i}'] = src.read(i)
                    
                    self.metadata = {
                        'height': src.height,
                        'width': src.width,
                        'crs': str(src.crs),
                        'transform': src.transform,
                        'bounds': src.bounds
                    }
                    
                    self.bands = bands
                    return bands
            except Exception as e:
                print(f"Error loading satellite data: {e}")
                return self.load_image_data(mock_data=True)
    
    def _generate_mock_band(self, height: int, width: int, base_value: float) -> np.ndarray:
        """Generate realistic mock spectral band data"""
        # Base reflectance with natural variation
        band = np.random.normal(base_value, base_value * 0.2, (height, width))
        
        # Add some spatial correlation (smooth variations)
        band = ndimage.gaussian_filter(band, sigma=2)
        
        # Add some random noise
        noise = np.random.normal(0, base_value * 0.05, (height, width))
        band += noise
        
        # Clip to valid reflectance range
        band = np.clip(band, 0, 1)
        
        return band
    
    def _add_mock_archaeological_features(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Add mock archaeological features to simulate real site signatures"""
        height, width = bands['red'].shape
        
        # Add circular earthwork features
        for _ in range(3):
            center_x = np.random.randint(200, width - 200)
            center_y = np.random.randint(200, height - 200)
            radius = np.random.randint(50, 150)
            
            # Create circular mask
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Modify spectral signatures to simulate bare soil/archaeological features
            bands['red'][mask] *= 1.2  # Increase red reflectance
            bands['nir'][mask] *= 0.8  # Decrease NIR (less vegetation)
            bands['swir1'][mask] *= 1.1  # Increase SWIR
        
        # Add linear features (ancient roads/causeways)
        for _ in range(2):
            start_x = np.random.randint(0, width)
            start_y = np.random.randint(0, height)
            length = np.random.randint(200, 500)
            angle = np.random.uniform(0, 2 * np.pi)
            
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            # Ensure within bounds
            end_x = np.clip(end_x, 0, width - 1)
            end_y = np.clip(end_y, 0, height - 1)
            
            # Create line mask
            rr, cc = self._line_coordinates(start_y, start_x, end_y, end_x)
            
            # Apply within bounds
            valid_mask = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            rr, cc = rr[valid_mask], cc[valid_mask]
            
            # Widen the line
            for dr in [-2, -1, 0, 1, 2]:
                for dc in [-2, -1, 0, 1, 2]:
                    rr_wide = np.clip(rr + dr, 0, height - 1)
                    cc_wide = np.clip(cc + dc, 0, width - 1)
                    
                    bands['red'][rr_wide, cc_wide] *= 1.15
                    bands['nir'][rr_wide, cc_wide] *= 0.85
        
        return bands
    
    def _line_coordinates(self, r0: int, c0: int, r1: int, c1: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinates for a line between two points (Bresenham's algorithm)"""
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        
        r_step = 1 if r0 < r1 else -1
        c_step = 1 if c0 < c1 else -1
        
        err = dr - dc
        
        rr, cc = [r0], [c0]
        r, c = r0, c0
        
        while r != r1 or c != c1:
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += r_step
            if e2 < dr:
                err += dr
                c += c_step
            rr.append(r)
            cc.append(c)
        
        return np.array(rr), np.array(cc)
    
    def calculate_spectral_indices(self) -> Dict[str, np.ndarray]:
        """Calculate vegetation and soil indices for archaeological analysis"""
        
        indices = {}
        
        if 'red' in self.bands and 'nir' in self.bands:
            # Normalized Difference Vegetation Index (NDVI)
            red = self.bands['red']
            nir = self.bands['nir']
            indices['ndvi'] = (nir - red) / (nir + red + 1e-8)
            
            # Enhanced Vegetation Index (EVI)
            if 'blue' in self.bands:
                blue = self.bands['blue']
                indices['evi'] = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
        
        if 'swir1' in self.bands and 'nir' in self.bands:
            # Normalized Difference Water Index (NDWI)
            swir1 = self.bands['swir1']
            nir = self.bands['nir']
            indices['ndwi'] = (nir - swir1) / (nir + swir1 + 1e-8)
        
        if 'swir1' in self.bands and 'swir2' in self.bands:
            # Normalized Burn Ratio (NBR) - useful for detecting cleared areas
            swir1 = self.bands['swir1']
            swir2 = self.bands['swir2']
            indices['nbr'] = (nir - swir2) / (nir + swir2 + 1e-8)
        
        if 'red' in self.bands and 'swir1' in self.bands:
            # Iron Oxide Ratio - useful for detecting exposed soils
            red = self.bands['red']
            swir1 = self.bands['swir1']
            indices['iron_oxide'] = red / (swir1 + 1e-8)
        
        # Soil Brightness Index
        if all(band in self.bands for band in ['red', 'green', 'blue']):
            indices['brightness'] = (self.bands['red'] + self.bands['green'] + self.bands['blue']) / 3
        
        self.indices = indices
        return indices
    
    def detect_geometric_patterns(self, method: str = 'edge_detection') -> List[Dict[str, Any]]:
        """Detect geometric patterns that might indicate archaeological features"""
        
        if method == 'edge_detection':
            return self._detect_edges_and_shapes()
        elif method == 'template_matching':
            return self._template_matching()
        elif method == 'machine_learning':
            return self._ml_pattern_detection()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_edges_and_shapes(self) -> List[Dict[str, Any]]:
        """Detect geometric patterns using edge detection and shape analysis"""
        
        patterns = []
        
        # Use brightness or NDVI for pattern detection
        if 'brightness' in self.indices:
            image = self.indices['brightness']
        elif 'ndvi' in self.indices:
            image = self.indices['ndvi']
        else:
            image = self.bands['red']  # Fallback
        
        # Edge detection
        edges = feature.canny(image, sigma=2, low_threshold=0.1, high_threshold=0.2)
        
        # Find contours
        contours = measure.find_contours(edges, 0.8)
        
        for i, contour in enumerate(contours):
            if len(contour) < 20:  # Filter very small contours
                continue
            
            # Calculate geometric properties
            props = self._analyze_contour_geometry(contour)
            
            if props['area'] > 100:  # Filter based on minimum area
                patterns.append({
                    'id': f'pattern_{i}',
                    'type': 'geometric_contour',
                    'contour': contour.tolist(),
                    'properties': props,
                    'detection_method': 'edge_detection'
                })
        
        return patterns
    
    def _analyze_contour_geometry(self, contour: np.ndarray) -> Dict[str, float]:
        """Analyze geometric properties of a contour"""
        
        # Convert contour to integer coordinates for region analysis
        contour_int = contour.astype(int)
        
        # Create binary mask for the contour
        mask = np.zeros((self.metadata['height'], self.metadata['width']), dtype=bool)
        
        # Fill the contour
        rr, cc = contour_int[:, 0], contour_int[:, 1]
        valid_mask = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
        mask[rr[valid_mask], cc[valid_mask]] = True
        
        # Calculate properties
        area = np.sum(mask)
        perimeter = len(contour)
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Centroid
        if area > 0:
            moments = measure.moments(mask.astype(float))
            centroid = (moments[1, 0] / moments[0, 0], moments[0, 1] / moments[0, 0])
        else:
            centroid = (0, 0)
        
        # Regularity (based on distance variation from centroid)
        if len(contour) > 3:
            distances = np.sqrt(np.sum((contour - centroid) ** 2, axis=1))
            regularity = 1.0 / (1.0 + np.std(distances))
        else:
            regularity = 0.0
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'centroid': centroid,
            'regularity': float(regularity),
            'aspect_ratio': self._calculate_aspect_ratio(contour)
        }
    
    def _calculate_aspect_ratio(self, contour: np.ndarray) -> float:
        """Calculate aspect ratio of contour bounding box"""
        if len(contour) < 2:
            return 1.0
        
        min_r, max_r = contour[:, 0].min(), contour[:, 0].max()
        min_c, max_c = contour[:, 1].min(), contour[:, 1].max()
        
        height = max_r - min_r
        width = max_c - min_c
        
        if height == 0 or width == 0:
            return 1.0
        
        return max(height, width) / min(height, width)
    
    def _template_matching(self) -> List[Dict[str, Any]]:
        """Detect patterns using template matching for common archaeological shapes"""
        
        patterns = []
        
        # Use NDVI or brightness for template matching
        if 'ndvi' in self.indices:
            image = self.indices['ndvi']
        else:
            image = self.bands['red']
        
        # Normalize image
        image_norm = (image - image.min()) / (image.max() - image.min())
        
        # Create circular templates of different sizes
        circle_templates = []
        for radius in [20, 35, 50, 75]:
            template = self._create_circular_template(radius)
            circle_templates.append((template, radius))
        
        # Match templates
        for template, radius in circle_templates:
            # Template matching using normalized cross-correlation
            result = feature.match_template(image_norm, template, pad_input=True)
            
            # Find peaks in the result
            peak_coords = feature.peak_local_maxima(result, min_distance=radius, threshold_abs=0.3)
            
            for coord in zip(*peak_coords):
                confidence = result[coord]
                patterns.append({
                    'id': f'circle_r{radius}_{coord[0]}_{coord[1]}',
                    'type': 'circular_template',
                    'center': coord,
                    'radius': radius,
                    'confidence': float(confidence),
                    'detection_method': 'template_matching'
                })
        
        return patterns
    
    def _create_circular_template(self, radius: int) -> np.ndarray:
        """Create circular template for template matching"""
        size = radius * 2 + 1
        template = np.zeros((size, size))
        
        center = radius
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        
        template[mask] = 1
        return template
    
    def _ml_pattern_detection(self) -> List[Dict[str, Any]]:
        """Use machine learning for pattern detection"""
        
        # Create feature vectors for each pixel
        features = self._extract_pixel_features()
        
        # Use unsupervised clustering to find anomalous patterns
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # DBSCAN clustering to find anomalous regions
        clustering = DBSCAN(eps=0.5, min_samples=10)
        cluster_labels = clustering.fit_predict(features_scaled)
        
        patterns = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            # Get pixels belonging to this cluster
            cluster_mask = cluster_labels == label
            cluster_pixels = np.where(cluster_mask.reshape(self.metadata['height'], self.metadata['width']))
            
            if len(cluster_pixels[0]) > 50:  # Minimum cluster size
                # Calculate cluster properties
                centroid = (np.mean(cluster_pixels[0]), np.mean(cluster_pixels[1]))
                area = len(cluster_pixels[0])
                
                # Calculate bounding box
                min_r, max_r = cluster_pixels[0].min(), cluster_pixels[0].max()
                min_c, max_c = cluster_pixels[1].min(), cluster_pixels[1].max()
                
                patterns.append({
                    'id': f'cluster_{label}',
                    'type': 'ml_cluster',
                    'centroid': centroid,
                    'area': area,
                    'bounding_box': {
                        'min_row': int(min_r),
                        'max_row': int(max_r),
                        'min_col': int(min_c),
                        'max_col': int(max_c)
                    },
                    'detection_method': 'machine_learning'
                })
        
        return patterns
    
    def _extract_pixel_features(self) -> np.ndarray:
        """Extract features for each pixel for ML analysis"""
        
        height, width = self.metadata['height'], self.metadata['width']
        features = []
        
        # Spectral features
        for band_name, band_data in self.bands.items():
            features.append(band_data.flatten())
        
        # Index features
        for index_name, index_data in self.indices.items():
            features.append(index_data.flatten())
        
        # Texture features (local standard deviation)
        if 'red' in self.bands:
            texture = ndimage.generic_filter(self.bands['red'], np.std, size=5)
            features.append(texture.flatten())
        
        # Edge features
        if 'brightness' in self.indices:
            edges = feature.canny(self.indices['brightness'])
            features.append(edges.flatten().astype(float))
        
        return np.column_stack(features)
    
    def analyze_vegetation_anomalies(self) -> Dict[str, Any]:
        """Analyze vegetation patterns for archaeological indicators"""
        
        if 'ndvi' not in self.indices:
            self.calculate_spectral_indices()
        
        ndvi = self.indices['ndvi']
        
        # Calculate vegetation anomalies
        ndvi_mean = np.mean(ndvi)
        ndvi_std = np.std(ndvi)
        
        # Areas with significantly different vegetation
        low_veg_mask = ndvi < (ndvi_mean - 2 * ndvi_std)  # Sparse vegetation
        high_veg_mask = ndvi > (ndvi_mean + 2 * ndvi_std)  # Dense vegetation
        
        # Find connected components
        low_veg_labels = measure.label(low_veg_mask)
        high_veg_labels = measure.label(high_veg_mask)
        
        anomalies = {
            'low_vegetation_areas': [],
            'high_vegetation_areas': [],
            'statistics': {
                'ndvi_mean': float(ndvi_mean),
                'ndvi_std': float(ndvi_std),
                'low_veg_threshold': float(ndvi_mean - 2 * ndvi_std),
                'high_veg_threshold': float(ndvi_mean + 2 * ndvi_std)
            }
        }
        
        # Analyze low vegetation areas (potential archaeological sites)
        for region in measure.regionprops(low_veg_labels):
            if region.area > 100:  # Minimum size threshold
                anomalies['low_vegetation_areas'].append({
                    'centroid': region.centroid,
                    'area': region.area,
                    'bounding_box': region.bbox,
                    'mean_ndvi': float(np.mean(ndvi[low_veg_labels == region.label]))
                })
        
        # Analyze high vegetation areas (forest islands)
        for region in measure.regionprops(high_veg_labels):
            if region.area > 100:
                anomalies['high_vegetation_areas'].append({
                    'centroid': region.centroid,
                    'area': region.area,
                    'bounding_box': region.bbox,
                    'mean_ndvi': float(np.mean(ndvi[high_veg_labels == region.label]))
                })
        
        return anomalies
    
    def create_composite_analysis(self) -> Dict[str, Any]:
        """Create comprehensive analysis combining multiple detection methods"""
        
        # Calculate spectral indices
        self.calculate_spectral_indices()
        
        # Detect patterns using different methods
        edge_patterns = self.detect_geometric_patterns('edge_detection')
        template_patterns = self.detect_geometric_patterns('template_matching')
        ml_patterns = self.detect_geometric_patterns('machine_learning')
        
        # Analyze vegetation anomalies
        vegetation_anomalies = self.analyze_vegetation_anomalies()
        
        # Combine and rank potential archaeological features
        all_patterns = edge_patterns + template_patterns + ml_patterns
        
        # Score each pattern based on multiple criteria
        scored_patterns = []
        for pattern in all_patterns:
            score = self._calculate_archaeological_score(pattern, vegetation_anomalies)
            pattern['archaeological_score'] = score
            scored_patterns.append(pattern)
        
        # Sort by score
        scored_patterns.sort(key=lambda x: x['archaeological_score'], reverse=True)
        
        return {
            'total_patterns': len(all_patterns),
            'high_potential_sites': [p for p in scored_patterns if p['archaeological_score'] > 0.7],
            'medium_potential_sites': [p for p in scored_patterns if 0.4 <= p['archaeological_score'] <= 0.7],
            'all_patterns': scored_patterns,
            'vegetation_anomalies': vegetation_anomalies,
            'analysis_timestamp': datetime.now().isoformat(),
            'metadata': self.metadata
        }
    
    def _calculate_archaeological_score(self, pattern: Dict[str, Any], vegetation_anomalies: Dict[str, Any]) -> float:
        """Calculate archaeological potential score for a detected pattern"""
        
        score = 0.0
        
        # Base score from detection method
        method_scores = {
            'edge_detection': 0.3,
            'template_matching': 0.5,
            'machine_learning': 0.4
        }
        score += method_scores.get(pattern.get('detection_method', ''), 0.0)
        
        # Geometric regularity score
        if 'properties' in pattern:
            props = pattern['properties']
            if 'regularity' in props:
                score += props['regularity'] * 0.3
            if 'circularity' in props:
                score += props['circularity'] * 0.2
        
        # Template matching confidence
        if 'confidence' in pattern:
            score += pattern['confidence'] * 0.3
        
        # Size-based scoring (prefer medium-sized features)
        if 'area' in pattern:
            area = pattern['area']
            if 500 <= area <= 5000:  # Optimal size range for archaeological features
                score += 0.2
            elif 100 <= area <= 10000:  # Acceptable range
                score += 0.1
        
        # Vegetation anomaly correlation
        if pattern.get('type') == 'circular_template' and 'center' in pattern:
            center = pattern['center']
            # Check if pattern coincides with vegetation anomaly
            for anomaly in vegetation_anomalies['low_vegetation_areas']:
                anom_center = anomaly['centroid']
                distance = np.sqrt((center[0] - anom_center[0])**2 + (center[1] - anom_center[1])**2)
                if distance < 50:  # Within 50 pixels
                    score += 0.2
                    break
        
        return min(score, 1.0)  # Cap at 1.0
    
    def visualize_results(self, analysis_results: Dict[str, Any], save_path: str = None) -> None:
        """Create visualizations of the analysis results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Archaeological Site Detection Analysis', fontsize=16)
        
        # 1. RGB Composite
        if all(band in self.bands for band in ['red', 'green', 'blue']):
            rgb = np.dstack([self.bands['red'], self.bands['green'], self.bands['blue']])
            rgb = np.clip(rgb * 3, 0, 1)  # Enhance brightness
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title('RGB Composite')
            axes[0, 0].axis('off')
        
        # 2. NDVI
        if 'ndvi' in self.indices:
            im1 = axes[0, 1].imshow(self.indices['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
            axes[0, 1].set_title('NDVI (Vegetation Index)')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1])
        
        # 3. Detected Patterns
        if 'brightness' in self.indices:
            base_image = self.indices['brightness']
        else:
            base_image = self.bands['red']
        
        axes[0, 2].imshow(base_image, cmap='gray')
        
        # Overlay high potential sites
        for site in analysis_results['high_potential_sites']:
            if 'center' in site:
                center = site['center']
                circle = plt.Circle(center[::-1], radius=20, fill=False, color='red', linewidth=2)
                axes[0, 2].add_patch(circle)
            elif 'centroid' in site:
                centroid = site['centroid']
                axes[0, 2].plot(centroid[1], centroid[0], 'ro', markersize=8)
        
        axes[0, 2].set_title(f'High Potential Sites ({len(analysis_results["high_potential_sites"])})')
        axes[0, 2].axis('off')
        
        # 4. Vegetation Anomalies
        if 'ndvi' in self.indices:
            veg_display = self.indices['ndvi'].copy()
            
            # Highlight anomalies
            for anomaly in analysis_results['vegetation_anomalies']['low_vegetation_areas']:
                bbox = anomaly['bounding_box']
                veg_display[bbox[0]:bbox[2], bbox[1]:bbox[3]] = -1  # Mark in red
            
            for anomaly in analysis_results['vegetation_anomalies']['high_vegetation_areas']:
                bbox = anomaly['bounding_box']
                veg_display[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1  # Mark in green
            
            im2 = axes[1, 0].imshow(veg_display, cmap='RdYlGn', vmin=-1, vmax=1)
            axes[1, 0].set_title('Vegetation Anomalies')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0])
        
        # 5. Score Distribution
        scores = [pattern['archaeological_score'] for pattern in analysis_results['all_patterns']]
        axes[1, 1].hist(scores, bins=20, alpha=0.7, color='blue')
        axes[1, 1].set_title('Archaeological Score Distribution')
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_ylabel('Frequency')
        
        # 6. Site Statistics
        stats_text = f"""
        Total Patterns Detected: {analysis_results['total_patterns']}
        High Potential Sites: {len(analysis_results['high_potential_sites'])}
        Medium Potential Sites: {len(analysis_results['medium_potential_sites'])}
        
        Vegetation Anomalies:
        - Low Vegetation Areas: {len(analysis_results['vegetation_anomalies']['low_vegetation_areas'])}
        - High Vegetation Areas: {len(analysis_results['vegetation_anomalies']['high_vegetation_areas'])}
        
        Analysis Date: {analysis_results['analysis_timestamp']}
        """
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                        verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Demonstrate satellite data processing for archaeological site detection"""
    
    print("Starting Satellite Data Processing for Archaeological Detection...")
    
    # Initialize processor
    processor = SatelliteImageProcessor()
    
    # Load data (using mock data for demonstration)
    print("Loading satellite imagery...")
    bands = processor.load_image_data(mock_data=True)
    print(f"Loaded {len(bands)} spectral bands")
    
    # Perform comprehensive analysis
    print("Performing comprehensive archaeological analysis...")
    results = processor.create_composite_analysis()
    
    # Print summary
    print(f"\nAnalysis Complete!")
    print(f"Total patterns detected: {results['total_patterns']}")
    print(f"High potential archaeological sites: {len(results['high_potential_sites'])}")
    print(f"Medium potential sites: {len(results['medium_potential_sites'])}")
    
    # Display top sites
    print("\nTop 5 High-Potential Sites:")
    for i, site in enumerate(results['high_potential_sites'][:5]):
        print(f"{i+1}. Score: {site['archaeological_score']:.3f}, Type: {site['type']}, Method: {site['detection_method']}")
    
    # Create visualization
    print("\nCreating visualization...")
    viz_path = '/home/myuser/OpenAI_to_Z_Challenge/satellite_analysis_visualization.png'
    processor.visualize_results(results, save_path=viz_path)
    
    # Save detailed results
    output_path = '/home/myuser/OpenAI_to_Z_Challenge/satellite_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    results = main()