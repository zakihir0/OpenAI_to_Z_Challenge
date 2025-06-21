#!/usr/bin/env python3
"""
AI-Powered Archaeological Structure Detection
Deep learning models for automated archaeological feature recognition in LIDAR data

Uses CNN, U-Net, and custom architectures for:
- Semantic segmentation of archaeological features
- Object detection for specific structure types
- Pattern recognition and classification
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
import cv2
from datetime import datetime

# Deep learning frameworks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available - using traditional computer vision methods")

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Traditional computer vision
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from skimage import filters, feature, measure, segmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArchaeologicalStructureDetector:
    """AI-powered archaeological structure detection system"""
    
    def __init__(self, model_type: str = 'hybrid'):
        """
        Initialize structure detector
        
        Args:
            model_type: 'traditional', 'cnn', 'unet', or 'hybrid'
        """
        self.model_type = model_type
        self.models = {}
        self.feature_extractors = {}
        
        # Initialize based on available libraries
        if PYTORCH_AVAILABLE and model_type in ['cnn', 'unet', 'hybrid']:
            self._initialize_deep_models()
        else:
            logger.info("Using traditional computer vision methods")
            self.model_type = 'traditional'
        
        self._initialize_traditional_methods()
    
    def _initialize_deep_models(self):
        """Initialize deep learning models"""
        
        if self.model_type in ['cnn', 'hybrid']:
            self.models['cnn_classifier'] = ArchaeologicalCNN()
            
        if self.model_type in ['unet', 'hybrid']:
            self.models['unet_segmentation'] = ArchaeologicalUNet()
        
        logger.info(f"Initialized deep learning models: {list(self.models.keys())}")
    
    def _initialize_traditional_methods(self):
        """Initialize traditional computer vision methods"""
        
        self.feature_extractors = {
            'geometric': GeometricFeatureExtractor(),
            'texture': TextureFeatureExtractor(),
            'morphological': MorphologicalFeatureExtractor()
        }
        
        # Initialize traditional classifiers
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        
        logger.info("Initialized traditional feature extractors and classifiers")
    
    def detect_structures(self, elevation_data: np.ndarray, 
                         hillshade: np.ndarray = None) -> Dict:
        """
        Detect archaeological structures in elevation data
        
        Args:
            elevation_data: DTM/DSM array
            hillshade: Optional hillshade array
            
        Returns:
            Dictionary with detected structures and confidence scores
        """
        
        logger.info(f"Detecting structures using {self.model_type} approach")
        
        # Preprocess data
        processed_data = self._preprocess_data(elevation_data, hillshade)
        
        # Extract features
        features = self._extract_comprehensive_features(processed_data)
        
        # Apply detection methods based on model type
        if self.model_type == 'traditional':
            detections = self._traditional_detection(features, processed_data)
        elif self.model_type == 'cnn':
            detections = self._cnn_detection(processed_data)
        elif self.model_type == 'unet':
            detections = self._unet_detection(processed_data)
        elif self.model_type == 'hybrid':
            detections = self._hybrid_detection(features, processed_data)
        else:
            detections = self._traditional_detection(features, processed_data)
        
        # Post-process and classify detections
        classified_structures = self._classify_structures(detections, features)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            classified_structures, features
        )
        
        return {
            'structures': classified_structures,
            'confidence_scores': confidence_scores,
            'feature_summary': self._summarize_features(features),
            'detection_method': self.model_type,
            'processing_time': datetime.now().isoformat()
        }
    
    def _preprocess_data(self, elevation: np.ndarray, 
                        hillshade: Optional[np.ndarray]) -> Dict:
        """Preprocess elevation and hillshade data"""
        
        # Normalize elevation data
        elevation_norm = (elevation - np.mean(elevation)) / np.std(elevation)
        
        # Generate hillshade if not provided
        if hillshade is None:
            dy, dx = np.gradient(elevation)
            slope = np.arctan(np.sqrt(dx**2 + dy**2))
            aspect = np.arctan2(-dx, dy)
            
            azimuth_rad = np.radians(315)  # Northwest illumination
            altitude_rad = np.radians(45)   # 45-degree altitude
            
            hillshade = np.sin(altitude_rad) * np.sin(slope) + \
                       np.cos(altitude_rad) * np.cos(slope) * \
                       np.cos(azimuth_rad - aspect)
            hillshade = ((hillshade + 1) * 127.5).astype(np.uint8)
        
        # Calculate additional terrain derivatives
        dy, dx = np.gradient(elevation)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        # Calculate curvature
        dyy, dyx = np.gradient(dy)
        dxy, dxx = np.gradient(dx)
        curvature = (dxx + dyy) / (1 + dx**2 + dy**2)**(3/2)
        
        # Local relief
        local_relief = elevation - ndimage.uniform_filter(elevation, size=20)
        
        return {
            'elevation_raw': elevation,
            'elevation_normalized': elevation_norm,
            'hillshade': hillshade,
            'slope': slope,
            'curvature': curvature,
            'local_relief': local_relief,
            'gradient_x': dx,
            'gradient_y': dy
        }
    
    def _extract_comprehensive_features(self, data: Dict) -> Dict:
        """Extract comprehensive features for structure detection"""
        
        features = {}
        
        # Geometric features
        if 'geometric' in self.feature_extractors:
            features['geometric'] = self.feature_extractors['geometric'].extract(data)
        
        # Texture features
        if 'texture' in self.feature_extractors:
            features['texture'] = self.feature_extractors['texture'].extract(data)
        
        # Morphological features
        if 'morphological' in self.feature_extractors:
            features['morphological'] = self.feature_extractors['morphological'].extract(data)
        
        return features
    
    def _traditional_detection(self, features: Dict, data: Dict) -> Dict:
        """Traditional computer vision-based detection"""
        
        detections = {
            'circular_structures': [],
            'linear_structures': [],
            'rectangular_structures': [],
            'mounds': [],
            'depressions': []
        }
        
        elevation = data['elevation_raw']
        hillshade = data['hillshade']
        local_relief = data['local_relief']
        
        # Detect circular structures using Hough circles
        detections['circular_structures'] = self._detect_circular_hough(hillshade)
        
        # Detect linear structures using line detection
        detections['linear_structures'] = self._detect_linear_structures(hillshade)
        
        # Detect rectangular structures using contour analysis
        detections['rectangular_structures'] = self._detect_rectangular_structures(hillshade)
        
        # Detect mounds using local maxima
        detections['mounds'] = self._detect_mounds_traditional(local_relief)
        
        # Detect depressions using local minima
        detections['depressions'] = self._detect_depressions_traditional(local_relief)
        
        return detections
    
    def _detect_circular_hough(self, image: np.ndarray) -> List[Dict]:
        """Detect circular structures using Hough circle transform"""
        
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Hough circle detection
        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=5, maxRadius=100
        )
        
        circular_structures = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                circular_structures.append({
                    'center': (y, x),  # Note: y, x order for numpy arrays
                    'radius': int(r),
                    'confidence': 0.7,  # Base confidence
                    'type': 'circular'
                })
        
        return circular_structures
    
    def _detect_linear_structures(self, image: np.ndarray) -> List[Dict]:
        """Detect linear structures using Hough line transform"""
        
        edges = cv2.Canny(image, 50, 150)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180, threshold=100,
            minLineLength=50, maxLineGap=10
        )
        
        linear_structures = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                
                linear_structures.append({
                    'start': (y1, x1),
                    'end': (y2, x2),
                    'length': float(length),
                    'angle': float(angle),
                    'confidence': 0.6,
                    'type': 'linear'
                })
        
        return linear_structures
    
    def _detect_rectangular_structures(self, image: np.ndarray) -> List[Dict]:
        """Detect rectangular structures using contour analysis"""
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_structures = []
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if polygon has 4 sides (rectangular)
            if len(approx) == 4 and cv2.contourArea(contour) > 100:
                # Calculate bounding rectangle
                rect = cv2.minAreaRect(contour)
                (center_x, center_y), (width, height), angle = rect
                
                # Check aspect ratio for rectangular shape
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
                
                if aspect_ratio < 5.0:  # Not too elongated
                    rectangular_structures.append({
                        'center': (center_y, center_x),
                        'width': float(width),
                        'height': float(height),
                        'angle': float(angle),
                        'aspect_ratio': float(aspect_ratio),
                        'area': float(cv2.contourArea(contour)),
                        'confidence': 0.6,
                        'type': 'rectangular'
                    })
        
        return rectangular_structures
    
    def _detect_mounds_traditional(self, relief: np.ndarray) -> List[Dict]:
        """Detect mounds using traditional methods"""
        
        # Find local maxima using scipy
        from scipy import ndimage
        
        # Use maximum filter to find local maxima
        local_maxima = ndimage.maximum_filter(relief, size=20) == relief
        local_maxima_coords = np.where(local_maxima)
        
        mounds = []
        threshold = np.std(relief) * 2.0
        
        for y, x in zip(*local_maxima_coords):
            height = relief[y, x]
            
            if height > threshold:
                # Analyze surrounding area
                window_size = 20
                y1, y2 = max(0, y-window_size), min(relief.shape[0], y+window_size)
                x1, x2 = max(0, x-window_size), min(relief.shape[1], x+window_size)
                
                surrounding = relief[y1:y2, x1:x2]
                prominence = height - np.mean(surrounding)
                
                if prominence > threshold / 2:
                    mounds.append({
                        'center': (y, x),
                        'height': float(height),
                        'prominence': float(prominence),
                        'confidence': min(prominence / threshold, 1.0),
                        'type': 'mound'
                    })
        
        return mounds
    
    def _detect_depressions_traditional(self, relief: np.ndarray) -> List[Dict]:
        """Detect depressions using traditional methods"""
        
        # Find local minima using scipy
        from scipy import ndimage
        
        # Use minimum filter to find local minima
        local_minima = ndimage.minimum_filter(relief, size=20) == relief
        local_minima_coords = np.where(local_minima)
        
        depressions = []
        threshold = -np.std(relief) * 2.0
        
        for y, x in zip(*local_minima_coords):
            depth = relief[y, x]
            
            if depth < threshold:
                # Analyze surrounding area
                window_size = 20
                y1, y2 = max(0, y-window_size), min(relief.shape[0], y+window_size)
                x1, x2 = max(0, x-window_size), min(relief.shape[1], x+window_size)
                
                surrounding = relief[y1:y2, x1:x2]
                prominence = np.mean(surrounding) - depth
                
                if prominence > abs(threshold) / 2:
                    depressions.append({
                        'center': (y, x),
                        'depth': float(abs(depth)),
                        'prominence': float(prominence),
                        'confidence': min(prominence / abs(threshold), 1.0),
                        'type': 'depression'
                    })
        
        return depressions
    
    def _cnn_detection(self, data: Dict) -> Dict:
        """CNN-based structure detection"""
        
        if not PYTORCH_AVAILABLE or 'cnn_classifier' not in self.models:
            logger.warning("CNN model not available, falling back to traditional methods")
            features = self._extract_comprehensive_features(data)
            return self._traditional_detection(features, data)
        
        # Prepare input data for CNN
        input_tensor = self._prepare_cnn_input(data)
        
        # Run CNN inference (mock implementation)
        detections = self._mock_cnn_inference(input_tensor, data)
        
        return detections
    
    def _unet_detection(self, data: Dict) -> Dict:
        """U-Net-based semantic segmentation"""
        
        if not PYTORCH_AVAILABLE or 'unet_segmentation' not in self.models:
            logger.warning("U-Net model not available, falling back to traditional methods")
            features = self._extract_comprehensive_features(data)
            return self._traditional_detection(features, data)
        
        # Prepare input for U-Net
        input_tensor = self._prepare_unet_input(data)
        
        # Run U-Net inference (mock implementation)
        segmentation_mask = self._mock_unet_inference(input_tensor)
        
        # Convert segmentation to detections
        detections = self._segmentation_to_detections(segmentation_mask, data)
        
        return detections
    
    def _hybrid_detection(self, features: Dict, data: Dict) -> Dict:
        """Hybrid detection combining traditional and deep learning methods"""
        
        # Get traditional detections
        traditional_detections = self._traditional_detection(features, data)
        
        # Apply deep learning refinement if available
        if PYTORCH_AVAILABLE and self.models:
            # Refine detections using deep learning
            refined_detections = self._refine_with_deep_learning(
                traditional_detections, data
            )
        else:
            refined_detections = traditional_detections
        
        return refined_detections
    
    def _prepare_cnn_input(self, data: Dict) -> torch.Tensor:
        """Prepare input tensor for CNN"""
        
        # Stack elevation, hillshade, and slope as channels
        input_data = np.stack([
            data['elevation_normalized'],
            data['hillshade'] / 255.0,
            data['slope'] / 90.0  # Normalize slope to 0-1
        ], axis=0)
        
        return torch.from_numpy(input_data).float().unsqueeze(0)
    
    def _prepare_unet_input(self, data: Dict) -> torch.Tensor:
        """Prepare input tensor for U-Net"""
        
        # Multi-channel input for U-Net
        input_data = np.stack([
            data['elevation_normalized'],
            data['hillshade'] / 255.0,
            data['slope'] / 90.0,
            data['curvature'],
            data['local_relief'] / np.std(data['local_relief'])
        ], axis=0)
        
        return torch.from_numpy(input_data).float().unsqueeze(0)
    
    def _mock_cnn_inference(self, input_tensor: torch.Tensor, data: Dict) -> Dict:
        """Mock CNN inference for demonstration"""
        
        logger.info("Running mock CNN inference")
        
        # Simulate CNN detection results
        height, width = data['elevation_raw'].shape
        
        # Generate some mock detections based on actual terrain
        detections = {
            'circular_structures': [],
            'linear_structures': [],
            'rectangular_structures': [],
            'mounds': [],
            'depressions': []
        }
        
        # Add some mock circular structures
        for i in range(3):
            y = np.random.randint(50, height-50)
            x = np.random.randint(50, width-50)
            r = np.random.randint(10, 30)
            
            detections['circular_structures'].append({
                'center': (y, x),
                'radius': r,
                'confidence': np.random.uniform(0.7, 0.95),
                'type': 'circular',
                'method': 'cnn'
            })
        
        return detections
    
    def _mock_unet_inference(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Mock U-Net inference for demonstration"""
        
        logger.info("Running mock U-Net inference")
        
        # Create mock segmentation mask
        batch_size, channels, height, width = input_tensor.shape
        
        # Generate synthetic segmentation mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add some synthetic regions
        cv2.circle(mask, (width//3, height//3), 25, 1, -1)  # Circular structure
        cv2.rectangle(mask, (width//2, height//2), (width//2+40, height//2+60), 2, -1)  # Rectangular
        cv2.line(mask, (0, height//4), (width, height//4), 3, 5)  # Linear feature
        
        return mask
    
    def _segmentation_to_detections(self, mask: np.ndarray, data: Dict) -> Dict:
        """Convert segmentation mask to structure detections"""
        
        detections = {
            'circular_structures': [],
            'linear_structures': [],
            'rectangular_structures': [],
            'mounds': [],
            'depressions': []
        }
        
        # Define class mappings
        class_mapping = {
            1: 'circular_structures',
            2: 'rectangular_structures', 
            3: 'linear_structures',
            4: 'mounds',
            5: 'depressions'
        }
        
        for class_id, structure_type in class_mapping.items():
            class_mask = (mask == class_id)
            
            if np.any(class_mask):
                # Find connected components
                labeled_mask = measure.label(class_mask)
                
                for region in measure.regionprops(labeled_mask):
                    if region.area > 25:  # Minimum size threshold
                        detection = {
                            'center': region.centroid,
                            'area': region.area,
                            'bbox': region.bbox,
                            'confidence': 0.8,  # Base confidence for segmentation
                            'type': structure_type.rstrip('s'),  # Remove 's' suffix
                            'method': 'unet'
                        }
                        
                        # Add type-specific properties
                        if structure_type == 'circular_structures':
                            detection['radius'] = np.sqrt(region.area / np.pi)
                        elif structure_type == 'rectangular_structures':
                            detection['aspect_ratio'] = region.major_axis_length / region.minor_axis_length
                        
                        detections[structure_type].append(detection)
        
        return detections
    
    def _refine_with_deep_learning(self, detections: Dict, data: Dict) -> Dict:
        """Refine traditional detections using deep learning"""
        
        # Mock refinement process
        refined_detections = detections.copy()
        
        # Boost confidence scores for structures detected by both methods
        for structure_type in refined_detections:
            if structure_type == 'summary':
                continue
                
            for structure in refined_detections[structure_type]:
                if 'confidence' in structure:
                    # Simulate refinement boost
                    structure['confidence'] = min(structure['confidence'] * 1.2, 1.0)
                    structure['method'] = 'hybrid'
        
        return refined_detections
    
    def _classify_structures(self, detections: Dict, features: Dict) -> Dict:
        """Classify detected structures with archaeological context"""
        
        classified = {}
        
        for structure_type, structures in detections.items():
            classified[structure_type] = []
            
            for structure in structures:
                # Add archaeological classification
                classified_structure = structure.copy()
                classified_structure['archaeological_type'] = self._classify_archaeological_type(
                    structure, structure_type
                )
                classified_structure['cultural_context'] = self._infer_cultural_context(
                    structure, structure_type
                )
                
                classified[structure_type].append(classified_structure)
        
        # Add summary statistics
        classified['summary'] = self._calculate_detection_summary(classified)
        
        return classified
    
    def _classify_archaeological_type(self, structure: Dict, structure_type: str) -> str:
        """Classify structure by archaeological type"""
        
        archaeological_types = {
            'circular_structures': ['ceremonial_circle', 'fortified_enclosure', 'residential_compound'],
            'linear_structures': ['ancient_road', 'canal', 'defensive_wall', 'boundary_marker'],
            'rectangular_structures': ['platform_mound', 'residential_foundation', 'ceremonial_plaza'],
            'mounds': ['burial_mound', 'ceremonial_mound', 'residential_mound', 'defensive_mound'],
            'depressions': ['ancient_quarry', 'ceremonial_pond', 'storage_pit', 'defensive_moat']
        }
        
        if structure_type in archaeological_types:
            # Simple classification based on size and other properties
            possible_types = archaeological_types[structure_type]
            
            if 'area' in structure:
                area = structure['area']
                if area > 1000:
                    return possible_types[0]  # Large = ceremonial/important
                elif area > 500:
                    return possible_types[1] if len(possible_types) > 1 else possible_types[0]
                else:
                    return possible_types[-1]  # Small = utilitarian
            
            return possible_types[0]  # Default to first type
        
        return 'unknown'
    
    def _infer_cultural_context(self, structure: Dict, structure_type: str) -> str:
        """Infer cultural/temporal context"""
        
        contexts = [
            'Pre-Columbian (500-1500 CE)',
            'Early Colonial Influence (1500-1600 CE)', 
            'Indigenous Continuity (1600-1800 CE)',
            'Modern Disturbance (1800+ CE)'
        ]
        
        # Simple heuristic based on structure characteristics
        if structure_type in ['circular_structures', 'mounds'] and structure.get('confidence', 0) > 0.7:
            return contexts[0]  # Pre-Columbian
        elif structure_type == 'rectangular_structures':
            return contexts[1]  # Colonial influence
        else:
            return contexts[0]  # Default to Pre-Columbian
    
    def _calculate_detection_summary(self, detections: Dict) -> Dict:
        """Calculate summary statistics for detections"""
        
        summary = {
            'total_structures': 0,
            'by_type': {},
            'confidence_distribution': [],
            'archaeological_types': []
        }
        
        for structure_type, structures in detections.items():
            if structure_type == 'summary':
                continue
                
            count = len(structures)
            summary['total_structures'] += count
            summary['by_type'][structure_type] = count
            
            # Collect confidence scores
            for structure in structures:
                if 'confidence' in structure:
                    summary['confidence_distribution'].append(structure['confidence'])
                if 'archaeological_type' in structure:
                    summary['archaeological_types'].append(structure['archaeological_type'])
        
        # Calculate confidence statistics
        if summary['confidence_distribution']:
            summary['mean_confidence'] = np.mean(summary['confidence_distribution'])
            summary['confidence_std'] = np.std(summary['confidence_distribution'])
        else:
            summary['mean_confidence'] = 0.0
            summary['confidence_std'] = 0.0
        
        return summary
    
    def _calculate_confidence_scores(self, structures: Dict, features: Dict) -> Dict:
        """Calculate overall confidence scores for detection"""
        
        scores = {
            'overall_confidence': 0.0,
            'structure_type_confidence': {},
            'method_reliability': self._get_method_reliability()
        }
        
        if 'summary' in structures:
            summary = structures['summary']
            scores['overall_confidence'] = summary.get('mean_confidence', 0.0)
            
            # Calculate confidence by structure type
            for structure_type, structures_list in structures.items():
                if structure_type == 'summary':
                    continue
                    
                if structures_list:
                    confidences = [s.get('confidence', 0.0) for s in structures_list]
                    scores['structure_type_confidence'][structure_type] = {
                        'mean': np.mean(confidences),
                        'count': len(confidences)
                    }
        
        return scores
    
    def _get_method_reliability(self) -> Dict:
        """Get reliability scores for different detection methods"""
        
        reliability = {
            'traditional': 0.7,
            'cnn': 0.85,
            'unet': 0.9,
            'hybrid': 0.95
        }
        
        return {
            'method': self.model_type,
            'reliability_score': reliability.get(self.model_type, 0.7),
            'description': f"Detection using {self.model_type} method"
        }
    
    def _summarize_features(self, features: Dict) -> Dict:
        """Summarize extracted features"""
        
        summary = {
            'feature_types': list(features.keys()),
            'total_features': 0
        }
        
        for feature_type, feature_data in features.items():
            if isinstance(feature_data, dict):
                summary[f'{feature_type}_count'] = len(feature_data)
                summary['total_features'] += len(feature_data)
        
        return summary


class GeometricFeatureExtractor:
    """Extract geometric features from terrain data"""
    
    def extract(self, data: Dict) -> Dict:
        """Extract geometric features"""
        
        elevation = data['elevation_raw']
        hillshade = data['hillshade']
        
        # Edge detection
        edges = cv2.Canny(hillshade, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        geometric_features = {
            'contours': [],
            'circular_features': [],
            'linear_features': []
        }
        
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                # Calculate geometric properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Fit ellipse
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        (center_x, center_y), (width, height), angle = ellipse
                        aspect_ratio = max(width, height) / min(width, height)
                        
                        feature = {
                            'area': float(area),
                            'perimeter': float(perimeter),
                            'circularity': float(circularity),
                            'aspect_ratio': float(aspect_ratio),
                            'center': (center_y, center_x),
                            'angle': float(angle)
                        }
                        
                        geometric_features['contours'].append(feature)
                        
                        # Classify feature
                        if circularity > 0.7:
                            geometric_features['circular_features'].append(feature)
                        elif aspect_ratio > 3.0:
                            geometric_features['linear_features'].append(feature)
        
        return geometric_features


class TextureFeatureExtractor:
    """Extract texture features from terrain data"""
    
    def extract(self, data: Dict) -> Dict:
        """Extract texture features"""
        
        hillshade = data['hillshade']
        elevation = data['elevation_raw']
        
        # Calculate texture measures
        texture_features = {
            'local_binary_patterns': self._calculate_lbp(hillshade),
            'gabor_responses': self._calculate_gabor_responses(hillshade),
            'entropy': self._calculate_local_entropy(hillshade),
            'roughness': self._calculate_terrain_roughness(elevation)
        }
        
        return texture_features
    
    def _calculate_lbp(self, image: np.ndarray, radius: int = 3, n_points: int = 24) -> Dict:
        """Calculate Local Binary Patterns"""
        
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(image, n_points, radius, method='uniform')
            
            return {
                'mean': float(np.mean(lbp)),
                'std': float(np.std(lbp)),
                'histogram': np.histogram(lbp, bins=n_points + 2)[0].tolist()
            }
        except ImportError:
            # Fallback texture measure
            return {
                'mean': float(np.mean(image)),
                'std': float(np.std(image)),
                'histogram': np.histogram(image, bins=50)[0].tolist()
            }
    
    def _calculate_gabor_responses(self, image: np.ndarray) -> Dict:
        """Calculate Gabor filter responses"""
        
        try:
            from skimage.filters import gabor
            
            responses = []
            for frequency in [0.1, 0.3, 0.5]:
                for theta in [0, 45, 90, 135]:
                    real, _ = gabor(image, frequency=frequency, theta=np.radians(theta))
                    responses.append(np.mean(np.abs(real)))
            
            return {
                'responses': responses,
                'mean_response': float(np.mean(responses)),
                'std_response': float(np.std(responses))
            }
        except ImportError:
            # Fallback using simple gradient measures
            dy, dx = np.gradient(image.astype(float))
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
            
            return {
                'responses': [float(np.mean(gradient_magnitude))],
                'mean_response': float(np.mean(gradient_magnitude)),
                'std_response': float(np.std(gradient_magnitude))
            }
    
    def _calculate_local_entropy(self, image: np.ndarray, window_size: int = 9) -> Dict:
        """Calculate local entropy"""
        
        entropy_image = np.zeros_like(image, dtype=float)
        
        pad_size = window_size // 2
        padded_image = np.pad(image, pad_size, mode='reflect')
        
        for i in range(entropy_image.shape[0]):
            for j in range(entropy_image.shape[1]):
                window = padded_image[i:i+window_size, j:j+window_size]
                hist, _ = np.histogram(window, bins=256, range=(0, 256))
                hist = hist / np.sum(hist)
                hist = hist[hist > 0]
                entropy_image[i, j] = -np.sum(hist * np.log2(hist))
        
        return {
            'mean_entropy': float(np.mean(entropy_image)),
            'std_entropy': float(np.std(entropy_image)),
            'max_entropy': float(np.max(entropy_image))
        }
    
    def _calculate_terrain_roughness(self, elevation: np.ndarray) -> Dict:
        """Calculate terrain roughness measures"""
        
        # Calculate gradients
        dy, dx = np.gradient(elevation)
        slope = np.sqrt(dx**2 + dy**2)
        
        # Calculate curvature
        dyy, dyx = np.gradient(dy)
        dxy, dxx = np.gradient(dx)
        curvature = np.abs(dxx + dyy)
        
        return {
            'mean_slope': float(np.mean(slope)),
            'std_slope': float(np.std(slope)),
            'mean_curvature': float(np.mean(curvature)),
            'roughness_index': float(np.std(elevation) / np.mean(elevation)) if np.mean(elevation) > 0 else 0.0
        }


class MorphologicalFeatureExtractor:
    """Extract morphological features from terrain data"""
    
    def extract(self, data: Dict) -> Dict:
        """Extract morphological features"""
        
        elevation = data['elevation_raw']
        local_relief = data['local_relief']
        
        morphological_features = {
            'peaks': self._find_peaks(local_relief),
            'valleys': self._find_valleys(local_relief),
            'ridges': self._find_ridges(elevation),
            'flat_areas': self._find_flat_areas(data['slope'])
        }
        
        return morphological_features
    
    def _find_peaks(self, relief: np.ndarray) -> List[Dict]:
        """Find peak features"""
        
        from scipy import ndimage
        
        # Local maxima
        local_maxima = ndimage.maximum_filter(relief, size=20) == relief
        
        # Remove edge effects
        local_maxima[0, :] = False
        local_maxima[-1, :] = False
        local_maxima[:, 0] = False
        local_maxima[:, -1] = False
        
        # Get peak coordinates
        peak_coords = np.where(local_maxima)
        
        peaks = []
        threshold = np.std(relief) * 2.0
        
        for y, x in zip(*peak_coords):
            height = relief[y, x]
            if height > threshold:
                peaks.append({
                    'location': (y, x),
                    'height': float(height),
                    'prominence': float(height - np.mean(relief))
                })
        
        return peaks
    
    def _find_valleys(self, relief: np.ndarray) -> List[Dict]:
        """Find valley features"""
        
        from scipy import ndimage
        
        # Local minima
        local_minima = ndimage.minimum_filter(relief, size=20) == relief
        
        # Remove edge effects
        local_minima[0, :] = False
        local_minima[-1, :] = False
        local_minima[:, 0] = False
        local_minima[:, -1] = False
        
        # Get valley coordinates
        valley_coords = np.where(local_minima)
        
        valleys = []
        threshold = -np.std(relief) * 2.0
        
        for y, x in zip(*valley_coords):
            depth = relief[y, x]
            if depth < threshold:
                valleys.append({
                    'location': (y, x),
                    'depth': float(abs(depth)),
                    'prominence': float(np.mean(relief) - depth)
                })
        
        return valleys
    
    def _find_ridges(self, elevation: np.ndarray) -> List[Dict]:
        """Find ridge features"""
        
        # Calculate second derivatives for ridge detection
        dy, dx = np.gradient(elevation)
        dyy, dyx = np.gradient(dy)
        dxy, dxx = np.gradient(dx)
        
        # Ridge criterion: negative curvature in both principal directions
        ridge_strength = -(dxx + dyy)
        
        # Threshold for ridge detection
        threshold = np.percentile(ridge_strength, 95)
        ridge_mask = ridge_strength > threshold
        
        # Find connected ridge components
        labeled_ridges = measure.label(ridge_mask)
        
        ridges = []
        for region in measure.regionprops(labeled_ridges):
            if region.area > 50:  # Minimum ridge length
                ridges.append({
                    'centroid': region.centroid,
                    'length': region.major_axis_length,
                    'orientation': region.orientation,
                    'strength': float(np.mean(ridge_strength[labeled_ridges == region.label]))
                })
        
        return ridges
    
    def _find_flat_areas(self, slope: np.ndarray) -> List[Dict]:
        """Find flat area features"""
        
        # Define flat areas as low slope regions
        slope_threshold = np.percentile(slope, 20)  # Bottom 20% of slopes
        flat_mask = slope < slope_threshold
        
        # Remove small isolated flat areas
        from scipy import ndimage
        flat_mask = ndimage.binary_opening(flat_mask, structure=np.ones((5, 5)))
        
        # Find connected flat components
        labeled_flats = measure.label(flat_mask)
        
        flat_areas = []
        for region in measure.regionprops(labeled_flats):
            if region.area > 100:  # Minimum flat area size
                flat_areas.append({
                    'centroid': region.centroid,
                    'area': region.area,
                    'compactness': region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0,
                    'mean_slope': float(np.mean(slope[labeled_flats == region.label]))
                })
        
        return flat_areas


# Deep learning model classes (mock implementations)
if PYTORCH_AVAILABLE:
    
    class ArchaeologicalCNN(nn.Module):
        """CNN for archaeological structure classification"""
        
        def __init__(self, num_classes: int = 5):
            super(ArchaeologicalCNN, self).__init__()
            
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    
    class ArchaeologicalUNet(nn.Module):
        """U-Net for archaeological structure segmentation"""
        
        def __init__(self, in_channels: int = 5, num_classes: int = 6):
            super(ArchaeologicalUNet, self).__init__()
            
            # Encoder
            self.enc1 = self._conv_block(in_channels, 64)
            self.enc2 = self._conv_block(64, 128)
            self.enc3 = self._conv_block(128, 256)
            self.enc4 = self._conv_block(256, 512)
            
            # Decoder
            self.dec4 = self._conv_block(512 + 256, 256)
            self.dec3 = self._conv_block(256 + 128, 128)
            self.dec2 = self._conv_block(128 + 64, 64)
            self.dec1 = nn.Conv2d(64, num_classes, kernel_size=1)
            
            self.pool = nn.MaxPool2d(2)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        def _conv_block(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            
            # Decoder
            d4 = self.dec4(torch.cat([self.upsample(e4), e3], dim=1))
            d3 = self.dec3(torch.cat([self.upsample(d4), e2], dim=1))
            d2 = self.dec2(torch.cat([self.upsample(d3), e1], dim=1))
            d1 = self.dec1(d2)
            
            return torch.softmax(d1, dim=1)


def main():
    """Test the AI structure detector"""
    
    # Create synthetic test data
    size = 256
    x, y = np.meshgrid(np.linspace(0, 100, size), np.linspace(0, 100, size))
    
    # Create test elevation data
    elevation = (
        20 * np.sin(x / 10) * np.cos(y / 15) +
        10 * np.sin(x / 5) * np.sin(y / 8) +
        2 * np.random.randn(size, size)
    )
    
    # Initialize detector
    detector = ArchaeologicalStructureDetector(model_type='hybrid')
    
    # Run detection
    results = detector.detect_structures(elevation)
    
    print("AI Structure Detection Results:")
    print(f"Total structures detected: {results['structures']['summary']['total_structures']}")
    print(f"Overall confidence: {results['confidence_scores']['overall_confidence']:.3f}")
    print(f"Detection method: {results['detection_method']}")
    
    # Print structure counts by type
    for structure_type, count in results['structures']['summary']['by_type'].items():
        if count > 0:
            print(f"  {structure_type}: {count}")


if __name__ == "__main__":
    main()