"""
OpenAI to Z Challenge - Amazon Archaeological Site Detection
Multi-modal approach combining computer vision and OpenAI models
"""

import os
import numpy as np
import cv2
from PIL import Image
import json
import requests
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArchaeologicalSite:
    """Dataclass for archaeological site detections"""
    latitude: float
    longitude: float
    confidence: float
    site_type: str
    description: str
    evidence_type: List[str]

class SatelliteImageProcessor:
    """Handles satellite image processing and feature extraction"""
    
    def __init__(self):
        self.vegetation_threshold = 0.3
        self.structure_threshold = 0.7
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess satellite image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def calculate_vegetation_index(self, image: np.ndarray) -> np.ndarray:
        """Calculate normalized difference vegetation index (NDVI) approximation"""
        # Convert to float for calculations
        img_float = image.astype(np.float32) / 255.0
        
        # Approximate NDVI using visible bands
        # NDVI = (NIR - Red) / (NIR + Red)
        # Using Green as NIR approximation
        green = img_float[:, :, 1]
        red = img_float[:, :, 0]
        
        # Avoid division by zero
        denominator = green + red
        denominator[denominator == 0] = 1e-6
        
        ndvi = (green - red) / denominator
        return ndvi
    
    def detect_geometric_patterns(self, image: np.ndarray) -> List[Dict]:
        """Detect geometric patterns that might indicate human structures"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection with multiple scales
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        patterns = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Lower threshold for smaller features
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check for various geometric patterns
                if len(approx) == 4:  # Rectangular
                    patterns.append({
                        'type': 'rectangular',
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'bbox': (x, y, w, h),
                        'confidence': min(1.0, area / 5000)  # Lower area threshold
                    })
                elif len(approx) > 6:  # Circular/elliptical
                    patterns.append({
                        'type': 'circular',
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'bbox': (x, y, w, h),
                        'confidence': min(1.0, area / 8000)
                    })
                elif len(approx) == 3:  # Triangular
                    patterns.append({
                        'type': 'triangular',
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'bbox': (x, y, w, h),
                        'confidence': min(1.0, area / 6000)
                    })
        
        return patterns
    
    def analyze_soil_marks(self, image: np.ndarray) -> np.ndarray:
        """Analyze soil color variations that might indicate buried structures"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for different soil types
        # Disturbed soil often appears different from natural soil
        brown_lower = np.array([10, 50, 50])
        brown_upper = np.array([20, 255, 200])
        
        soil_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_CLOSE, kernel)
        soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_OPEN, kernel)
        
        return soil_mask

class OpenAIAnalyzer:
    """Integrates with OpenAI API for advanced image analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not found. Some features will be limited.")
    
    def analyze_image_with_gpt4v(self, image_path: str, context: str = "") -> Dict:
        """Analyze image using GPT-4 Vision"""
        if not self.api_key:
            return {"error": "OpenAI API key not available"}
        
        # This would require actual OpenAI API integration
        # For now, return a mock response
        mock_response = {
            "analysis": "Potential archaeological features detected based on geometric patterns and vegetation anomalies.",
            "confidence": 0.75,
            "features": [
                "Linear structures possibly indicating ancient pathways",
                "Circular patterns suggesting settlement areas",
                "Vegetation anomalies indicating subsurface structures"
            ]
        }
        return mock_response
    
    def generate_site_description(self, features: List[Dict]) -> str:
        """Generate detailed description of potential archaeological site"""
        if not features:
            return "No significant archaeological features detected."
        
        descriptions = []
        for feature in features:
            if feature['type'] == 'rectangular':
                descriptions.append(f"Rectangular structure ({feature['area']:.0f} sq pixels) with aspect ratio {feature['aspect_ratio']:.2f}")
        
        return "; ".join(descriptions)

class ArchaeologicalDetector:
    """Main class for archaeological site detection"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.image_processor = SatelliteImageProcessor()
        self.openai_analyzer = OpenAIAnalyzer(openai_api_key)
        self.detection_threshold = 0.1  # Lower threshold for better detection
    
    def process_image(self, image_path: str, coordinates: Tuple[float, float]) -> List[ArchaeologicalSite]:
        """Process a single satellite image for archaeological features"""
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = self.image_processor.load_image(image_path)
        if image is None:
            return []
        
        sites = []
        
        # Computer vision analysis
        patterns = self.image_processor.detect_geometric_patterns(image)
        vegetation_index = self.image_processor.calculate_vegetation_index(image)
        soil_marks = self.image_processor.analyze_soil_marks(image)
        
        # OpenAI analysis
        ai_analysis = self.openai_analyzer.analyze_image_with_gpt4v(image_path)
        
        # Combine results
        for pattern in patterns:
            if pattern['confidence'] > self.detection_threshold:
                # Calculate coordinates (simplified)
                lat, lon = coordinates  # This would need proper georeferencing
                
                site = ArchaeologicalSite(
                    latitude=lat,
                    longitude=lon,
                    confidence=pattern['confidence'],
                    site_type="Structure",
                    description=self.openai_analyzer.generate_site_description([pattern]),
                    evidence_type=["geometric_pattern", "soil_variation"]
                )
                sites.append(site)
        
        return sites
    
    def process_region(self, image_directory: str, coordinates_file: str) -> List[ArchaeologicalSite]:
        """Process multiple images in a region"""
        all_sites = []
        
        # Load coordinates if available
        coordinates = {}
        if os.path.exists(coordinates_file):
            try:
                with open(coordinates_file, 'r') as f:
                    coordinates = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load coordinates file: {e}")
        
        # Process each image
        for filename in os.listdir(image_directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_path = os.path.join(image_directory, filename)
                coord = coordinates.get(filename, (0.0, 0.0))  # Default coordinates
                
                sites = self.process_image(image_path, coord)
                all_sites.extend(sites)
        
        return all_sites
    
    def export_results(self, sites: List[ArchaeologicalSite], output_file: str):
        """Export detection results to file"""
        results = []
        for site in sites:
            results.append({
                'latitude': site.latitude,
                'longitude': site.longitude,
                'confidence': site.confidence,
                'site_type': site.site_type,
                'description': site.description,
                'evidence_type': site.evidence_type
            })
        
        # Export to JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also export to CSV for easier analysis
        df = pd.DataFrame(results)
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results exported to {output_file} and {csv_file}")

def main():
    """Main execution function"""
    # Initialize detector
    detector = ArchaeologicalDetector()
    
    # Create sample data directories
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Process images (if any exist)
    image_dir = 'data/images'
    coordinates_file = 'data/coordinates.json'
    
    if os.path.exists(image_dir) and os.listdir(image_dir):
        logger.info("Processing satellite images...")
        sites = detector.process_region(image_dir, coordinates_file)
        
        # Export results
        output_file = 'results/archaeological_sites.json'
        detector.export_results(sites, output_file)
        
        logger.info(f"Detected {len(sites)} potential archaeological sites")
        
        # Print summary
        for i, site in enumerate(sites[:5]):  # Show first 5 sites
            print(f"Site {i+1}: {site.description} (Confidence: {site.confidence:.2f})")
    else:
        logger.info("No images found in data directory. Creating sample structure...")
        
        # Create sample coordinates file
        sample_coordinates = {
            "sample_image_1.jpg": [-3.4653, -62.2159],  # Amazon region
            "sample_image_2.jpg": [-2.5297, -60.0261]
        }
        
        with open(coordinates_file, 'w') as f:
            json.dump(sample_coordinates, f, indent=2)
        
        print("Sample structure created. Add satellite images to data/images/ directory.")
        print("Update data/coordinates.json with actual image coordinates.")

if __name__ == "__main__":
    main()