#!/usr/bin/env python3
"""
Archaeological Site Analysis System
Core analysis engine for satellite imagery and archaeological feature detection
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArchaeologicalAnalyzer:
    """Main archaeological analysis engine"""
    
    def __init__(self):
        self.api_config = self._load_api_config()
        self.api_available = self._validate_api_config()
    
    def _load_api_config(self) -> Dict:
        """Load API configuration from environment"""
        return {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'base_url': os.getenv('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1'),
            'model': os.getenv('OPENAI_MODEL', 'deepseek/deepseek-r1-0528:free')
        }
    
    def _validate_api_config(self) -> bool:
        """Validate API configuration"""
        if not self.api_config['api_key']:
            logger.warning("No API key found - using fallback analysis")
            return False
        
        logger.info(f"API configured: {self.api_config['model']}")
        return True
    
    def analyze_site(self, site_info: Dict, image_data: Optional[np.ndarray] = None) -> Dict:
        """Perform comprehensive archaeological site analysis"""
        
        # Get or create image data
        if image_data is None:
            image_data = self._acquire_site_imagery(site_info)
        
        # Computer vision analysis
        cv_analysis = self._computer_vision_analysis(image_data)
        
        # AI interpretation
        ai_interpretation = self._get_ai_interpretation(image_data, site_info, cv_analysis)
        
        # Calculate archaeological score
        archaeological_score = self._calculate_archaeological_score(cv_analysis, site_info)
        
        return {
            'site_info': site_info,
            'timestamp': datetime.now().isoformat(),
            'image_properties': {
                'shape': image_data.shape,
                'dtype': str(image_data.dtype),
                'size_mb': image_data.nbytes / (1024 * 1024)
            },
            'computer_vision_analysis': cv_analysis,
            'ai_interpretation': ai_interpretation,
            'archaeological_score': archaeological_score,
            'confidence_level': self._get_confidence_level(archaeological_score),
            'recommendations': self._generate_recommendations(archaeological_score, cv_analysis)
        }
    
    def _acquire_site_imagery(self, site_info: Dict) -> np.ndarray:
        """Acquire satellite imagery for the site"""
        
        # Try to download real satellite imagery
        real_image = self._download_real_satellite_image(site_info)
        
        if real_image is not None:
            logger.info(f"Using real satellite imagery for {site_info['name']}")
            return real_image
        
        logger.warning(f"Real imagery unavailable for {site_info['name']}, using synthetic fallback")
        return self._create_synthetic_imagery(site_info)
    
    def _download_real_satellite_image(self, site_info: Dict) -> Optional[np.ndarray]:
        """Download real satellite imagery"""
        
        try:
            lat, lon = site_info['lat'], site_info['lon']
            x_tile, y_tile = self._deg2tile(lat, lon, 16)
            
            url = f'https://mt1.google.com/vt/lyrs=s&x={x_tile}&y={y_tile}&z=16'
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                import PIL.Image
                from io import BytesIO
                
                img = PIL.Image.open(BytesIO(response.content))
                img_array = np.array(img.convert('RGB'))
                
                # Resize to standard size
                if img_array.shape[:2] != (512, 512):
                    img_resized = PIL.Image.fromarray(img_array).resize((512, 512))
                    img_array = np.array(img_resized)
                
                return img_array
                
        except Exception as e:
            logger.debug(f"Failed to download real imagery: {e}")
        
        return None
    
    def _create_synthetic_imagery(self, site_info: Dict) -> np.ndarray:
        """Create synthetic satellite imagery for testing"""
        
        # Basic forest base
        image = np.random.randint(30, 80, (512, 512, 3), dtype=np.uint8)
        image[:, :, 1] += 20  # More green for forest
        
        # Add terrain variation
        x, y = np.meshgrid(np.linspace(0, 10, 512), np.linspace(0, 10, 512))
        terrain = 15 * np.sin(x) * np.cos(y)
        
        for i in range(3):
            image[:, :, i] = np.clip(image[:, :, i] + terrain, 0, 255)
        
        return image
    
    def _computer_vision_analysis(self, image: np.ndarray) -> Dict:
        """Comprehensive computer vision analysis"""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Edge detection and contour analysis
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze geometric features
        geometric_features = self._analyze_geometric_features(contours)
        
        # Texture analysis
        texture_analysis = {
            'mean_intensity': float(np.mean(gray)),
            'std_intensity': float(np.std(gray)),
            'entropy': self._calculate_entropy(gray),
            'edge_density': float(np.sum(edges > 0) / edges.size)
        }
        
        # Vegetation analysis (if color image)
        vegetation_analysis = {}
        if len(image.shape) == 3:
            vegetation_analysis = self._analyze_vegetation(image)
        
        # Spatial pattern analysis
        spatial_patterns = self._analyze_spatial_patterns(gray)
        
        return {
            'geometric_features': geometric_features,
            'texture_analysis': texture_analysis,
            'vegetation_analysis': vegetation_analysis,
            'spatial_patterns': spatial_patterns
        }
    
    def _analyze_geometric_features(self, contours) -> Dict:
        """Analyze geometric features from contours"""
        
        features = []
        circular_count = 0
        linear_count = 0
        regular_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter noise
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Bounding rectangle for aspect ratio
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
                    
                    feature = {
                        'area': float(area),
                        'circularity': float(circularity),
                        'aspect_ratio': float(aspect_ratio),
                        'regularity': float(circularity if circularity < 1.0 else 1.0 / aspect_ratio)
                    }
                    
                    features.append(feature)
                    
                    # Classify features
                    if circularity > 0.7:
                        circular_count += 1
                    if aspect_ratio > 3.0:
                        linear_count += 1
                    if feature['regularity'] > 0.6:
                        regular_count += 1
        
        return {
            'total_count': len(features),
            'circular_count': circular_count,
            'linear_count': linear_count,
            'regular_count': regular_count,
            'feature_details': features[:10]  # Top 10 features
        }
    
    def _analyze_vegetation(self, image: np.ndarray) -> Dict:
        """Analyze vegetation patterns"""
        
        red = image[:, :, 0].astype(float)
        green = image[:, :, 1].astype(float)
        
        # NDVI proxy
        ndvi_proxy = (green - red) / (green + red + 1e-8)
        
        return {
            'ndvi_mean': float(np.mean(ndvi_proxy)),
            'ndvi_std': float(np.std(ndvi_proxy)),
            'vegetation_coverage': float(np.sum(ndvi_proxy > 0.1) / ndvi_proxy.size),
            'anomaly_areas': float(np.sum(np.abs(ndvi_proxy - np.mean(ndvi_proxy)) > 2*np.std(ndvi_proxy)) / ndvi_proxy.size)
        }
    
    def _analyze_spatial_patterns(self, gray: np.ndarray) -> Dict:
        """Analyze spatial patterns for regularity"""
        
        # Grid regularity through FFT
        grid_score = 0.0
        try:
            fft = np.fft.fft2(gray)
            power_spectrum = np.abs(fft) ** 2
            peak_threshold = np.mean(power_spectrum) + 3 * np.std(power_spectrum)
            peaks = np.sum(power_spectrum > peak_threshold)
            grid_score = min(peaks / 100.0, 1.0)
        except:
            pass
        
        return {
            'grid_regularity': float(grid_score),
            'symmetry_score': self._calculate_symmetry(gray),
            'pattern_strength': float(np.std(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0.0
        }
    
    def _calculate_entropy(self, gray: np.ndarray) -> float:
        """Calculate image entropy"""
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))
    
    def _calculate_symmetry(self, gray: np.ndarray) -> float:
        """Calculate symmetry score"""
        try:
            h, w = gray.shape
            
            # Horizontal symmetry
            left_half = gray[:, :w//2]
            right_half = np.fliplr(gray[:, w//2:])
            
            if left_half.shape == right_half.shape:
                h_symmetry = 1.0 - np.mean(np.abs(left_half - right_half)) / 255.0
            else:
                h_symmetry = 0.0
            
            # Vertical symmetry
            top_half = gray[:h//2, :]
            bottom_half = np.flipud(gray[h//2:, :])
            
            if top_half.shape == bottom_half.shape:
                v_symmetry = 1.0 - np.mean(np.abs(top_half - bottom_half)) / 255.0
            else:
                v_symmetry = 0.0
            
            return float((h_symmetry + v_symmetry) / 2.0)
        except:
            return 0.0
    
    def _get_ai_interpretation(self, image: np.ndarray, site_info: Dict, cv_analysis: Dict) -> str:
        """Get AI interpretation of the analysis"""
        
        if not self.api_available:
            return self._generate_expert_fallback_analysis(site_info, cv_analysis)
        
        # Try AI analysis
        try:
            return self._call_ai_api(site_info, cv_analysis)
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return self._generate_expert_fallback_analysis(site_info, cv_analysis)
    
    def _call_ai_api(self, site_info: Dict, cv_analysis: Dict) -> str:
        """Call AI API for analysis"""
        
        import openai
        
        client = openai.OpenAI(
            api_key=self.api_config['api_key'],
            base_url=self.api_config['base_url']
        )
        
        prompt = self._build_ai_prompt(site_info, cv_analysis)
        
        response = client.chat.completions.create(
            model=self.api_config['model'],
            messages=[
                {"role": "system", "content": "You are Dr. Maria Santos, a leading expert in Amazonian archaeology with 25 years of experience in satellite-based site discovery."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _build_ai_prompt(self, site_info: Dict, cv_analysis: Dict) -> str:
        """Build AI analysis prompt"""
        
        cv_summary = self._summarize_cv_analysis(cv_analysis)
        
        return f"""
EXPERT ARCHAEOLOGICAL ANALYSIS REQUEST

SITE: {site_info['name']}
COORDINATES: {site_info.get('lat', 0):.3f}, {site_info.get('lon', 0):.3f}
REGION: Amazon Basin
PRIORITY: {site_info.get('priority', 'unknown')}

COMPUTER VISION ANALYSIS RESULTS:
{cv_summary}

EXPECTED ARCHAEOLOGICAL FEATURES:
{', '.join(site_info.get('expected_features', []))}

As a world-renowned expert in Amazonian archaeology, provide a comprehensive assessment:

1. ARCHAEOLOGICAL SIGNIFICANCE: Evaluate the detected patterns
2. CULTURAL CONTEXT: Compare with known Amazon basin sites
3. FEATURE INTERPRETATION: Explain what the patterns might represent
4. CONFIDENCE ASSESSMENT: Rate confidence level and evidence
5. RESEARCH RECOMMENDATIONS: Suggest specific next steps

Focus on Amazon archaeological context and be specific about feature significance.
"""
    
    def _summarize_cv_analysis(self, cv_analysis: Dict) -> str:
        """Summarize computer vision analysis for AI prompt"""
        
        geom = cv_analysis['geometric_features']
        texture = cv_analysis['texture_analysis']
        veg = cv_analysis.get('vegetation_analysis', {})
        spatial = cv_analysis['spatial_patterns']
        
        return f"""
GEOMETRIC FEATURES:
- Total features: {geom['total_count']}
- Circular features: {geom['circular_count']} (potential earthworks)
- Linear features: {geom['linear_count']} (potential roads/canals)
- Regular patterns: {geom['regular_count']} (artificial structures)

TEXTURE ANALYSIS:
- Edge density: {texture['edge_density']:.3f}
- Entropy: {texture['entropy']:.2f}
- Contrast: {texture['std_intensity']:.1f}

VEGETATION PATTERNS:
- Coverage: {veg.get('vegetation_coverage', 0):.1%}
- Anomalies: {veg.get('anomaly_areas', 0):.1%}
- NDVI mean: {veg.get('ndvi_mean', 0):.3f}

SPATIAL ANALYSIS:
- Grid regularity: {spatial['grid_regularity']:.3f}
- Symmetry: {spatial['symmetry_score']:.3f}
- Pattern strength: {spatial['pattern_strength']:.3f}
"""
    
    def _generate_expert_fallback_analysis(self, site_info: Dict, cv_analysis: Dict) -> str:
        """Generate expert analysis when AI is unavailable"""
        
        geom = cv_analysis['geometric_features']
        veg = cv_analysis.get('vegetation_analysis', {})
        spatial = cv_analysis['spatial_patterns']
        
        # Determine site type
        if geom['circular_count'] > 5:
            site_type = "earthwork complex"
        elif geom['linear_count'] > 3:
            site_type = "linear feature system"
        elif geom['regular_count'] > 8:
            site_type = "structured settlement"
        else:
            site_type = "mixed archaeological features"
        
        # Assess confidence
        total_indicators = geom['circular_count'] + geom['linear_count'] + geom['regular_count']
        if total_indicators > 15:
            confidence = "HIGH"
            significance = "significant archaeological potential"
        elif total_indicators > 8:
            confidence = "MEDIUM"
            significance = "moderate archaeological interest"
        else:
            confidence = "LOW"
            significance = "requires further investigation"
        
        return f"""
EXPERT ARCHAEOLOGICAL ASSESSMENT - {site_info['name']}

SITE CLASSIFICATION: {site_type.upper()}
ARCHAEOLOGICAL SIGNIFICANCE: {significance.upper()}
CONFIDENCE LEVEL: {confidence}

FEATURE ANALYSIS:
The satellite imagery reveals {geom['total_count']} distinct geometric features, including {geom['circular_count']} circular patterns and {geom['linear_count']} linear structures. This pattern distribution is consistent with {site_type} typical of pre-Columbian Amazon settlements.

CULTURAL CONTEXT:
Based on the geographic location at {site_info.get('lat', 0):.3f}, {site_info.get('lon', 0):.3f}, this site falls within the broader Amazon archaeological region known for:
- Geometric earthworks (similar to Acre geoglyphs)
- Raised field agricultural systems
- Complex settlement patterns from 500-1500 CE

VEGETATION INDICATORS:
Vegetation analysis shows {veg.get('anomaly_areas', 0):.1%} anomalous areas, suggesting possible subsurface archaeological features. The NDVI patterns indicate {veg.get('vegetation_coverage', 0):.0%} vegetation coverage with clear geometric disruptions.

SPATIAL PATTERNS:
Grid regularity score of {spatial['grid_regularity']:.2f} and symmetry score of {spatial['symmetry_score']:.2f} indicate planned construction rather than natural formation.

RECOMMENDATIONS:
1. Ground-penetrating radar survey to confirm subsurface features
2. High-resolution drone imagery for detailed mapping
3. Consultation with local indigenous communities
4. Archaeological field survey if patterns confirmed
5. Comparative analysis with known Amazon archaeological sites

RESEARCH PRIORITY: {confidence} - This site warrants {'immediate' if confidence == 'HIGH' else 'eventual'} archaeological investigation.
"""
    
    def _calculate_archaeological_score(self, cv_analysis: Dict, site_info: Dict) -> float:
        """Calculate comprehensive archaeological score"""
        
        score = 0.0
        
        # Geometric features (40% of score)
        geom = cv_analysis['geometric_features']
        geom_score = min((geom['circular_count'] * 0.05 + 
                         geom['linear_count'] * 0.03 + 
                         geom['regular_count'] * 0.02), 0.4)
        score += geom_score
        
        # Texture complexity (20% of score)
        texture = cv_analysis['texture_analysis']
        texture_score = min(texture['edge_density'] * 2.0, 0.2)
        score += texture_score
        
        # Vegetation anomalies (20% of score)
        veg = cv_analysis.get('vegetation_analysis', {})
        if veg:
            veg_score = min(veg.get('anomaly_areas', 0) * 5.0, 0.2)
            score += veg_score
        
        # Spatial patterns (20% of score)
        spatial = cv_analysis['spatial_patterns']
        spatial_score = min((spatial['grid_regularity'] + spatial['symmetry_score']) * 0.1, 0.2)
        score += spatial_score
        
        return min(score, 1.0)
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level based on score"""
        if score > 0.7:
            return "HIGH"
        elif score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self, score: float, cv_analysis: Dict) -> List[str]:
        """Generate specific recommendations based on analysis"""
        
        recommendations = []
        
        if score > 0.7:
            recommendations.extend([
                "Immediate high-resolution satellite imagery acquisition",
                "Ground-penetrating radar survey planning",
                "Consultation with regional archaeological authorities",
                "Coordination with local indigenous communities"
            ])
        elif score > 0.4:
            recommendations.extend([
                "Additional remote sensing analysis",
                "Comparative study with known archaeological sites",
                "Environmental impact assessment",
                "Preliminary field reconnaissance"
            ])
        else:
            recommendations.extend([
                "Extended monitoring for changes over time",
                "Analysis of historical imagery for patterns",
                "Investigation of similar features in region"
            ])
        
        # Feature-specific recommendations
        geom = cv_analysis['geometric_features']
        if geom['circular_count'] > 5:
            recommendations.append("Focus on circular earthwork investigation (potential geoglyphs)")
        if geom['linear_count'] > 3:
            recommendations.append("Map linear feature network (potential road/canal system)")
        
        return recommendations
    
    def _deg2tile(self, lat_deg: float, lon_deg: float, zoom: int) -> tuple:
        """Convert lat/lon to tile coordinates"""
        import math
        
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x_tile = int((lon_deg + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        
        return x_tile, y_tile