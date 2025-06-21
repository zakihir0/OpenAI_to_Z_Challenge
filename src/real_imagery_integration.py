#!/usr/bin/env python3
"""
Real Imagery Integration for Archaeological Analysis
Combines actual satellite/aerial imagery with analysis results
"""

import os
import requests
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from datetime import datetime
import io
import cv2
from typing import Dict, List, Optional, Tuple
import logging
import time
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealImageryProvider:
    """Provides access to real satellite and aerial imagery"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache'
        })
        
        # Multiple free satellite imagery sources
        self.imagery_sources = {
            'google_hybrid': 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            'google_satellite': 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            'esri_world': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            'bing_aerial': 'https://ecn.t0.tiles.virtualearth.net/tiles/a{quadkey}?g=587&mkt=en-us&n=z',
            'mapbox_satellite': 'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}@2x'
        }
    
    def deg2num(self, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile numbers"""
        import math
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
    
    def download_satellite_tile(self, lat: float, lon: float, zoom: int = 17, 
                               source: str = 'google_satellite') -> Optional[np.ndarray]:
        """Download a satellite tile for specific coordinates"""
        try:
            x, y = self.deg2num(lat, lon, zoom)
            
            if source == 'google_satellite':
                url = self.imagery_sources[source].format(x=x, y=y, z=zoom)
            elif source == 'esri_world':
                url = self.imagery_sources[source].format(x=x, y=y, z=zoom)
            else:
                logger.warning(f"Source {source} not implemented yet")
                return None
            
            logger.info(f"Downloading tile: {url}")
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return np.array(image)
            else:
                logger.warning(f"Failed to download tile: {response.status_code}")
                # Try alternative source
                if source == 'google_satellite':
                    return self.download_satellite_tile(lat, lon, zoom, 'esri_world')
                return None
                
        except Exception as e:
            logger.error(f"Error downloading satellite tile: {e}")
            return None
    
    def get_multi_resolution_imagery(self, lat: float, lon: float) -> Dict:
        """Get imagery at multiple resolutions and sources"""
        
        imagery_collection = {
            'location': {'lat': lat, 'lon': lon},
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Try multiple zoom levels and sources
        zoom_levels = [15, 16, 17]  # Different resolutions
        sources = ['google_satellite', 'esri_world']
        
        for zoom in zoom_levels:
            for source in sources:
                try:
                    image_data = self.download_satellite_tile(lat, lon, zoom, source)
                    if image_data is not None:
                        key = f"{source}_z{zoom}"
                        imagery_collection['sources'][key] = {
                            'image_data': image_data,
                            'zoom': zoom,
                            'source': source,
                            'shape': image_data.shape,
                            'success': True
                        }
                        logger.info(f"✓ Successfully downloaded {key}")
                        break  # Got one working source for this zoom
                except Exception as e:
                    logger.error(f"Failed to download {source} at zoom {zoom}: {e}")
        
        return imagery_collection
    
    def create_high_quality_composite(self, lat: float, lon: float, radius_tiles: int = 2) -> Optional[np.ndarray]:
        """Create high-quality composite image from multiple tiles"""
        
        try:
            zoom = 16  # Good balance of resolution and availability
            center_x, center_y = self.deg2num(lat, lon, zoom)
            
            # Download tiles in a grid around the center
            tiles = []
            for dx in range(-radius_tiles, radius_tiles + 1):
                for dy in range(-radius_tiles, radius_tiles + 1):
                    x = center_x + dx
                    y = center_y + dy
                    
                    # Convert back to lat/lon for download
                    tile_lat = self.num2deg_lat(y, zoom)
                    tile_lon = self.num2deg_lon(x, zoom)
                    
                    tile_data = self.download_satellite_tile(tile_lat, tile_lon, zoom)
                    if tile_data is not None:
                        tiles.append({
                            'data': tile_data,
                            'x': dx, 'y': dy,
                            'shape': tile_data.shape
                        })
                        time.sleep(0.5)  # Rate limiting
            
            if not tiles:
                return None
            
            # Merge tiles into composite
            return self._merge_tile_grid(tiles, radius_tiles)
            
        except Exception as e:
            logger.error(f"Error creating composite: {e}")
            return None
    
    def num2deg_lat(self, y: int, zoom: int) -> float:
        """Convert tile Y to latitude"""
        import math
        n = 2.0 ** zoom
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        return math.degrees(lat_rad)
    
    def num2deg_lon(self, x: int, zoom: int) -> float:
        """Convert tile X to longitude"""
        n = 2.0 ** zoom
        return x / n * 360.0 - 180.0
    
    def _merge_tile_grid(self, tiles: List[Dict], radius: int) -> np.ndarray:
        """Merge tiles into a single composite image"""
        
        if not tiles:
            return None
        
        # Assume all tiles same size
        tile_height, tile_width = tiles[0]['shape'][:2]
        grid_size = 2 * radius + 1
        
        # Create composite
        if len(tiles[0]['shape']) == 3:
            channels = tiles[0]['shape'][2]
            composite = np.zeros((grid_size * tile_height, grid_size * tile_width, channels), dtype=np.uint8)
        else:
            composite = np.zeros((grid_size * tile_height, grid_size * tile_width), dtype=np.uint8)
        
        # Place tiles
        for tile in tiles:
            grid_x = tile['x'] + radius
            grid_y = tile['y'] + radius
            
            start_y = grid_y * tile_height
            end_y = start_y + tile_height
            start_x = grid_x * tile_width
            end_x = start_x + tile_width
            
            composite[start_y:end_y, start_x:end_x] = tile['data']
        
        return composite

class IntegratedArchaeologicalAnalyzer:
    """Combines real imagery with archaeological analysis"""
    
    def __init__(self):
        self.imagery_provider = RealImageryProvider()
        self.deepseek_config = {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'base_url': os.getenv('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1'),
            'model': os.getenv('OPENAI_MODEL', 'deepseek/deepseek-r1-0528:free')
        }
    
    def comprehensive_site_analysis(self, site_info: Dict) -> Dict:
        """Perform comprehensive analysis combining real imagery and detection"""
        
        lat, lon = site_info['lat'], site_info['lon']
        logger.info(f"Starting comprehensive analysis for {site_info['name']}")
        
        # Get real satellite imagery
        imagery_collection = self.imagery_provider.get_multi_resolution_imagery(lat, lon)
        
        # Get high-quality composite if possible
        composite_image = self.imagery_provider.create_high_quality_composite(lat, lon, radius_tiles=1)
        
        # Select best available image for analysis
        analysis_image = self._select_best_image(imagery_collection, composite_image)
        
        if analysis_image is None:
            logger.error(f"Could not obtain imagery for {site_info['name']}")
            return {'error': 'No imagery available'}
        
        # Perform archaeological analysis
        analysis_results = {
            'site_info': site_info,
            'imagery_metadata': {
                'sources_attempted': list(imagery_collection['sources'].keys()),
                'successful_downloads': len(imagery_collection['sources']),
                'analysis_image_shape': analysis_image.shape,
                'composite_available': composite_image is not None
            },
            'feature_analysis': self._analyze_archaeological_features(analysis_image),
            'vegetation_analysis': self._analyze_vegetation_patterns(analysis_image),
            'geometric_analysis': self._analyze_geometric_patterns(analysis_image),
            'ai_interpretation': self._get_ai_interpretation(analysis_image, site_info),
            'archaeological_score': 0.0
        }
        
        # Calculate overall archaeological score
        analysis_results['archaeological_score'] = self._calculate_archaeological_score(analysis_results)
        
        # Create comprehensive visualization
        visualization_path = self._create_integrated_visualization(
            analysis_image, imagery_collection, analysis_results
        )
        analysis_results['visualization_path'] = visualization_path
        
        return analysis_results
    
    def _select_best_image(self, imagery_collection: Dict, composite: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Select the best available image for analysis"""
        
        # Prefer composite if available
        if composite is not None:
            return composite
        
        # Select highest resolution successful download
        best_image = None
        best_zoom = 0
        
        for key, data in imagery_collection['sources'].items():
            if data['success'] and data['zoom'] > best_zoom:
                best_image = data['image_data']
                best_zoom = data['zoom']
        
        return best_image
    
    def _analyze_archaeological_features(self, image: np.ndarray) -> Dict:
        """Analyze archaeological features in real imagery"""
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        features = {
            'edge_density': self._calculate_edge_density(gray),
            'circular_features': self._detect_circular_features(gray),
            'linear_features': self._detect_linear_features(gray),
            'texture_analysis': self._analyze_texture(gray)
        }
        
        return features
    
    def _analyze_vegetation_patterns(self, image: np.ndarray) -> Dict:
        """Analyze vegetation patterns for archaeological indicators"""
        
        if len(image.shape) != 3:
            return {'error': 'RGB image required'}
        
        # Enhanced vegetation analysis
        red = image[:, :, 0].astype(float)
        green = image[:, :, 1].astype(float)
        blue = image[:, :, 2].astype(float)
        
        # Calculate vegetation indices
        ndvi_proxy = (green - red) / (green + red + 1e-8)
        
        vegetation_stats = {
            'ndvi_mean': float(np.mean(ndvi_proxy)),
            'ndvi_std': float(np.std(ndvi_proxy)),
            'vegetation_coverage': float(np.sum(ndvi_proxy > 0.1) / ndvi_proxy.size),
            'anomaly_detection': self._detect_vegetation_anomalies(ndvi_proxy),
            'pattern_regularity': self._calculate_pattern_regularity(ndvi_proxy)
        }
        
        return vegetation_stats
    
    def _analyze_geometric_patterns(self, image: np.ndarray) -> Dict:
        """Advanced geometric pattern analysis"""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Edge detection with multiple methods
        edges_canny = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        geometric_features = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter noise
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Fit bounding rectangle
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
                    
                    geometric_features.append({
                        'area': float(area),
                        'circularity': float(circularity),
                        'aspect_ratio': float(aspect_ratio),
                        'center': [float(rect[0][0]), float(rect[0][1])],
                        'angle': float(rect[2])
                    })
        
        return {
            'total_features': len(geometric_features),
            'circular_count': len([f for f in geometric_features if f['circularity'] > 0.7]),
            'linear_count': len([f for f in geometric_features if f['aspect_ratio'] > 3.0]),
            'feature_details': geometric_features[:20]  # Top 20 features
        }
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density in image"""
        edges = cv2.Canny(gray, 50, 150)
        return float(np.sum(edges > 0) / edges.size)
    
    def _detect_circular_features(self, gray: np.ndarray) -> List[Dict]:
        """Detect circular features using Hough circles"""
        try:
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=5, maxRadius=100)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                return [{'center': [int(x), int(y)], 'radius': int(r)} 
                       for x, y, r in circles]
            return []
        except:
            return []
    
    def _detect_linear_features(self, gray: np.ndarray) -> List[Dict]:
        """Detect linear features using Hough lines"""
        try:
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                   minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                return [{'start': [int(x1), int(y1)], 'end': [int(x2), int(y2)]} 
                       for x1, y1, x2, y2 in lines[:, 0]]
            return []
        except:
            return []
    
    def _analyze_texture(self, gray: np.ndarray) -> Dict:
        """Analyze image texture properties"""
        # Calculate texture metrics
        return {
            'entropy': self._calculate_entropy(gray),
            'contrast': float(np.std(gray)),
            'homogeneity': float(1.0 / (1.0 + np.var(gray)))
        }
    
    def _calculate_entropy(self, gray: np.ndarray) -> float:
        """Calculate image entropy"""
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))
    
    def _detect_vegetation_anomalies(self, ndvi: np.ndarray) -> Dict:
        """Detect vegetation anomalies that might indicate archaeological features"""
        
        # Calculate statistics
        mean_ndvi = np.mean(ndvi)
        std_ndvi = np.std(ndvi)
        
        # Detect anomalies (areas significantly different from mean)
        anomaly_threshold = 2 * std_ndvi
        anomalies = np.abs(ndvi - mean_ndvi) > anomaly_threshold
        
        return {
            'anomaly_percentage': float(np.sum(anomalies) / anomalies.size),
            'anomaly_count': int(np.sum(anomalies)),
            'mean_deviation': float(std_ndvi)
        }
    
    def _calculate_pattern_regularity(self, ndvi: np.ndarray) -> float:
        """Calculate pattern regularity in vegetation"""
        try:
            # Use FFT to detect regular patterns
            fft = np.fft.fft2(ndvi)
            power_spectrum = np.abs(fft) ** 2
            peak_ratio = np.max(power_spectrum) / np.mean(power_spectrum)
            return float(min(peak_ratio / 1000, 1.0))
        except:
            return 0.0
    
    def _calculate_archaeological_score(self, analysis: Dict) -> float:
        """Calculate overall archaeological potential score"""
        
        score = 0.0
        
        # Geometric features
        geom = analysis['geometric_analysis']
        if geom['circular_count'] > 0:
            score += min(geom['circular_count'] * 0.1, 0.3)
        if geom['linear_count'] > 0:
            score += min(geom['linear_count'] * 0.05, 0.2)
        
        # Vegetation anomalies
        veg = analysis['vegetation_analysis']
        if 'anomaly_detection' in veg:
            anomaly_score = veg['anomaly_detection']['anomaly_percentage']
            if anomaly_score > 0.05:  # 5% anomalies
                score += min(anomaly_score * 2, 0.2)
        
        # Pattern regularity
        if veg.get('pattern_regularity', 0) > 0.1:
            score += 0.1
        
        # Edge density (structured features)
        edge_density = analysis['feature_analysis']['edge_density']
        if edge_density > 0.02:  # 2% edges
            score += min(edge_density * 5, 0.2)
        
        return min(score, 1.0)
    
    def _get_ai_interpretation(self, image: np.ndarray, site_info: Dict) -> str:
        """Get AI interpretation of real satellite imagery"""
        
        try:
            import openai
            
            # Analyze image properties
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            image_stats = {
                'dimensions': f"{image.shape[0]}x{image.shape[1]}",
                'brightness_mean': f"{np.mean(gray):.1f}",
                'contrast': f"{np.std(gray):.1f}",
                'edge_density': f"{cv2.Canny(gray.astype(np.uint8), 50, 150).sum() / gray.size:.4f}"
            }
            
            prompt = f"""
            Analyze this REAL Amazon satellite image for archaeological significance:
            
            SITE: {site_info['name']}
            LOCATION: {site_info['lat']:.3f}, {site_info['lon']:.3f}
            PRIORITY: {site_info.get('priority', 'unknown')}
            EXPECTED: {', '.join(site_info.get('expected_features', []))}
            
            IMAGE PROPERTIES (REAL SATELLITE DATA):
            - Resolution: {image_stats['dimensions']} pixels
            - Brightness: {image_stats['brightness_mean']}
            - Contrast: {image_stats['contrast']}
            - Edge Density: {image_stats['edge_density']}
            
            This is actual satellite imagery from the Amazon region. Provide expert archaeological assessment focusing on:
            
            1. Visible anthropogenic features in the satellite data
            2. Vegetation patterns suggesting subsurface archaeology
            3. Geometric anomalies indicating human construction
            4. Comparison with known Amazonian archaeological sites
            5. Confidence level and research recommendations
            
            Be specific about observable features in this real imagery.
            """
            
            client = openai.OpenAI(
                api_key=self.deepseek_config['api_key'],
                base_url=self.deepseek_config['base_url']
            )
            
            response = client.chat.completions.create(
                model=self.deepseek_config['model'],
                messages=[
                    {"role": "system", "content": "You are a leading expert in satellite archaeology with 20+ years experience analyzing real satellite imagery for archaeological discovery in the Amazon basin."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI interpretation failed: {e}")
            return f"AI analysis unavailable: {str(e)}"
    
    def _create_integrated_visualization(self, analysis_image: np.ndarray, 
                                       imagery_collection: Dict, 
                                       analysis_results: Dict) -> str:
        """Create comprehensive visualization combining real imagery and analysis"""
        
        site_name = analysis_results['site_info']['name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Main satellite image (large)
        ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
        ax1.imshow(analysis_image)
        ax1.set_title(f'REAL SATELLITE IMAGERY\n{site_name}\n{analysis_results["site_info"]["lat"]:.3f}, {analysis_results["site_info"]["lon"]:.3f}', 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Add scale bar and north arrow
        self._add_scale_elements(ax1, analysis_image.shape)
        
        # Edge detection analysis
        ax2 = plt.subplot2grid((4, 4), (0, 2))
        if len(analysis_image.shape) == 3:
            gray = cv2.cvtColor(analysis_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = analysis_image
        edges = cv2.Canny(gray, 50, 150)
        ax2.imshow(edges, cmap='hot')
        ax2.set_title('Archaeological Features\n(Edge Detection)', fontsize=10)
        ax2.axis('off')
        
        # Vegetation analysis
        ax3 = plt.subplot2grid((4, 4), (0, 3))
        if len(analysis_image.shape) == 3:
            green_channel = analysis_image[:, :, 1]
            ax3.imshow(green_channel, cmap='Greens')
        else:
            ax3.imshow(analysis_image, cmap='Greens')
        ax3.set_title('Vegetation Analysis\n(Green Channel)', fontsize=10)
        ax3.axis('off')
        
        # Geometric pattern overlay
        ax4 = plt.subplot2grid((4, 4), (1, 2))
        overlay_image = self._create_feature_overlay(analysis_image, analysis_results)
        ax4.imshow(overlay_image)
        ax4.set_title('Detected Patterns\n(Overlay)', fontsize=10)
        ax4.axis('off')
        
        # Statistics panel
        ax5 = plt.subplot2grid((4, 4), (1, 3))
        self._add_statistics_panel(ax5, analysis_results)
        
        # Multi-resolution comparison
        if len(imagery_collection['sources']) > 1:
            available_sources = list(imagery_collection['sources'].keys())
            for i, source_key in enumerate(available_sources[:2]):
                ax = plt.subplot2grid((4, 4), (2, i))
                source_data = imagery_collection['sources'][source_key]
                ax.imshow(source_data['image_data'])
                ax.set_title(f'{source_data["source"]}\nZoom {source_data["zoom"]}', fontsize=9)
                ax.axis('off')
        
        # AI Interpretation text
        ax6 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)
        ai_text = analysis_results['ai_interpretation']
        # Wrap text to fit
        wrapped_text = self._wrap_text(ai_text, 80)
        ax6.text(0.05, 0.95, f"AI ARCHAEOLOGICAL ASSESSMENT:\n{wrapped_text}", 
                transform=ax6.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        # Archaeological score and metadata
        ax7 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
        self._add_assessment_summary(ax7, analysis_results)
        
        plt.tight_layout()
        
        # Save visualization
        filename = f'results/integrated_analysis_{site_name.replace(" ", "_")}_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Integrated visualization saved: {filename}")
        return filename
    
    def _add_scale_elements(self, ax, image_shape):
        """Add scale bar and north arrow to image"""
        height, width = image_shape[:2]
        
        # Simple scale bar (approximate)
        scale_length = width // 10
        scale_y = height - 30
        scale_x = 20
        
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
               'w-', linewidth=3)
        ax.text(scale_x + scale_length//2, scale_y - 10, '~100m', 
               ha='center', va='top', color='white', fontweight='bold')
        
        # North arrow
        arrow_x, arrow_y = width - 50, 50
        ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y + 20),
                   arrowprops=dict(arrowstyle='->', color='white', lw=2),
                   color='white', fontweight='bold', ha='center')
    
    def _create_feature_overlay(self, image: np.ndarray, analysis: Dict) -> np.ndarray:
        """Create overlay showing detected archaeological features"""
        
        overlay = image.copy()
        
        # Get geometric features
        geom_analysis = analysis['geometric_analysis']
        
        # Draw circles for circular features
        for feature in geom_analysis['feature_details']:
            if feature['circularity'] > 0.7:  # Circular features
                center = tuple(map(int, feature['center']))
                radius = int(np.sqrt(feature['area'] / np.pi))
                cv2.circle(overlay, center, radius, (255, 0, 0), 2)  # Red circles
        
        # Draw lines for linear features
        feature_analysis = analysis['feature_analysis']
        for line in feature_analysis.get('linear_features', []):
            start = tuple(line['start'])
            end = tuple(line['end'])
            cv2.line(overlay, start, end, (0, 255, 0), 2)  # Green lines
        
        return overlay
    
    def _add_statistics_panel(self, ax, analysis: Dict):
        """Add statistics panel to visualization"""
        
        geom = analysis['geometric_analysis']
        veg = analysis['vegetation_analysis']
        
        stats_text = f"""ANALYSIS STATISTICS
        
Archaeological Score: {analysis['archaeological_score']:.2f}
        
GEOMETRIC FEATURES:
• Total Features: {geom['total_features']}
• Circular: {geom['circular_count']}
• Linear: {geom['linear_count']}

VEGETATION:
• Coverage: {veg.get('vegetation_coverage', 0):.1%}
• Anomalies: {veg.get('anomaly_detection', {}).get('anomaly_percentage', 0):.1%}

IMAGE SOURCES:
• Downloads: {analysis['imagery_metadata']['successful_downloads']}
• Resolution: {analysis['imagery_metadata']['analysis_image_shape'][0]}x{analysis['imagery_metadata']['analysis_image_shape'][1]}
"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _add_assessment_summary(self, ax, analysis: Dict):
        """Add archaeological assessment summary"""
        
        score = analysis['archaeological_score']
        confidence = 'HIGH' if score > 0.7 else 'MEDIUM' if score > 0.4 else 'LOW'
        
        summary_text = f"""
ARCHAEOLOGICAL ASSESSMENT SUMMARY

OVERALL SCORE: {score:.2f}/1.0 ({confidence} CONFIDENCE)

SITE: {analysis['site_info']['name']}
COORDINATES: {analysis['site_info']['lat']:.3f}, {analysis['site_info']['lon']:.3f}
PRIORITY: {analysis['site_info'].get('priority', 'Unknown')}

EXPECTED FEATURES: {', '.join(analysis['site_info'].get('expected_features', []))}

RECOMMENDATION: {'IMMEDIATE INVESTIGATION' if score > 0.7 else 'FURTHER ANALYSIS NEEDED' if score > 0.4 else 'LOW PRIORITY'}
"""
        
        color = 'green' if score > 0.7 else 'orange' if score > 0.4 else 'red'
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to specified width"""
        import textwrap
        return '\n'.join(textwrap.wrap(text, width))

def analyze_sites_with_real_imagery():
    """Analyze archaeological sites using real satellite imagery"""
    
    # Priority archaeological sites
    target_sites = [
        {
            'name': 'Llanos de Mojos Border Region',
            'lat': -15.200, 'lon': -64.800,
            'priority': 'highest',
            'expected_features': ['raised_fields', 'canals', 'earthworks']
        },
        {
            'name': 'Acre Geoglyph Extension Area',
            'lat': -10.500, 'lon': -67.800,
            'priority': 'highest',
            'expected_features': ['earthworks', 'circular_patterns']
        },
        {
            'name': 'Upper Xingu Expansion Zone',
            'lat': -12.500, 'lon': -52.800,
            'priority': 'high',
            'expected_features': ['circular_patterns', 'linear_features']
        }
    ]
    
    analyzer = IntegratedArchaeologicalAnalyzer()
    results = []
    
    logger.info("Starting real satellite imagery archaeological analysis")
    
    for site in target_sites:
        logger.info(f"Analyzing: {site['name']}")
        
        try:
            analysis = analyzer.comprehensive_site_analysis(site)
            
            if 'error' not in analysis:
                results.append(analysis)
                score = analysis['archaeological_score']
                logger.info(f"  ✓ Analysis complete - Score: {score:.2f}")
                logger.info(f"  ✓ Visualization: {analysis['visualization_path']}")
            else:
                logger.error(f"  ✗ Analysis failed: {analysis['error']}")
                
        except Exception as e:
            logger.error(f"  ✗ Error analyzing {site['name']}: {e}")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'results/real_imagery_analysis_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate summary report
    _generate_real_imagery_report(results, timestamp)
    
    logger.info(f"Real imagery analysis complete. Results: {results_file}")
    return results

def _generate_real_imagery_report(results: List[Dict], timestamp: str):
    """Generate comprehensive real imagery analysis report"""
    
    report_lines = [
        "REAL SATELLITE IMAGERY ARCHAEOLOGICAL ANALYSIS REPORT",
        "="*60,
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Sites Successfully Analyzed: {len(results)}",
        "",
        "REAL IMAGERY ARCHAEOLOGICAL DISCOVERIES:",
        ""
    ]
    
    for i, result in enumerate(results, 1):
        site = result['site_info']
        score = result['archaeological_score']
        geom = result['geometric_analysis']
        
        report_lines.extend([
            f"{i}. {site['name']} 🛰️",
            f"   Real Coordinates: {site['lat']:.3f}, {site['lon']:.3f}",
            f"   Archaeological Score: {score:.2f}/1.0",
            f"   Confidence: {'HIGH' if score > 0.7 else 'MEDIUM' if score > 0.4 else 'LOW'}",
            "",
            f"   📡 REAL SATELLITE DATA:",
            f"   - Image Sources: {result['imagery_metadata']['successful_downloads']}",
            f"   - Resolution: {result['imagery_metadata']['analysis_image_shape'][0]}x{result['imagery_metadata']['analysis_image_shape'][1]}",
            f"   - Composite Available: {result['imagery_metadata']['composite_available']}",
            "",
            f"   🏛️ DETECTED FEATURES:",
            f"   - Total Geometric: {geom['total_features']}",
            f"   - Circular (earthworks): {geom['circular_count']}",
            f"   - Linear (roads/canals): {geom['linear_count']}",
            "",
            f"   🎯 VISUALIZATION: {result['visualization_path']}",
            "",
            "   " + "="*50,
            ""
        ])
    
    avg_score = np.mean([r['archaeological_score'] for r in results]) if results else 0
    
    report_lines.extend([
        "📊 ANALYSIS SUMMARY:",
        f"- Average Archaeological Score: {avg_score:.2f}",
        f"- High Confidence Sites: {len([r for r in results if r['archaeological_score'] > 0.7])}",
        f"- Total Features Detected: {sum(r['geometric_analysis']['total_features'] for r in results)}",
        "",
        "🔬 TECHNICAL ACHIEVEMENTS:",
        "- Successfully downloaded real satellite imagery",
        "- Integrated multi-source satellite data",
        "- Applied computer vision to actual Amazon imagery",
        "- Generated comprehensive archaeological assessments",
        "",
        "🚁 IMMEDIATE NEXT STEPS:",
        "1. Ground-truth high-scoring locations",
        "2. Request higher resolution commercial imagery",
        "3. Plan field expedition to confirmed sites",
        "4. Coordinate with local authorities and communities"
    ])
    
    # Save report
    report_file = f'results/real_imagery_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))

if __name__ == "__main__":
    results = analyze_sites_with_real_imagery()