# OpenAI to Z Challenge: Archaeological Site Detection Implementation
# Based on research from Kaggle notebooks and archaeological remote sensing techniques

import os
import openai
import pandas as pd
import numpy as np
import rasterio
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
from typing import List, Dict, Any
import json
from datetime import datetime

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    AMAZON_BOUNDS = {
        'min_lat': -20.0,  # Southern boundary
        'max_lat': 10.0,   # Northern boundary  
        'min_lon': -80.0,  # Western boundary
        'max_lon': -45.0   # Eastern boundary
    }
    TARGET_COUNTRIES = ['Brazil', 'Bolivia', 'Colombia', 'Ecuador', 'Guyana', 'Peru', 'Suriname', 'Venezuela', 'French Guiana']

class OpenAIIntegration:
    """Enhanced OpenAI API integration for archaeological analysis"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
    def analyze_satellite_features(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze satellite imagery features using OpenAI's vision models"""
        
        prompt = """
        You are an expert archaeologist analyzing satellite imagery for potential archaeological sites in the Amazon basin.
        
        Analyze the provided satellite data for:
        1. Geometric patterns that could indicate human settlement
        2. Vegetation anomalies that might suggest buried structures
        3. Topographical features consistent with ancient construction
        4. River proximity and accessibility factors
        5. Likelihood of undiscovered archaeological significance
        
        Provide analysis in JSON format with confidence scores (0-1) for each factor.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert archaeological analyst specializing in Amazon basin site detection."},
                    {"role": "user", "content": f"{prompt}\n\nSatellite data: {json.dumps(image_data)}"}
                ],
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return {"error": str(e)}
    
    def generate_site_hypothesis(self, coordinates: tuple, features: Dict[str, Any]) -> str:
        """Generate archaeological hypothesis for potential sites"""
        
        prompt = f"""
        Based on the following archaeological indicators at coordinates {coordinates}:
        {json.dumps(features, indent=2)}
        
        Generate a detailed hypothesis about the potential archaeological significance of this location.
        Include:
        - Likely time period and cultural context
        - Type of settlement or structure
        - Historical significance within Amazon basin archaeology
        - Recommended investigation methods
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a leading expert in pre-Columbian Amazonian archaeology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating hypothesis: {e}"

class SatelliteDataProcessor:
    """Satellite imagery processing for archaeological feature detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def calculate_vegetation_indices(self, red_band: np.ndarray, nir_band: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate vegetation indices for archaeological anomaly detection"""
        
        # Normalized Difference Vegetation Index (NDVI)
        ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
        
        # Enhanced Vegetation Index (EVI)
        evi = 2.5 * ((nir_band - red_band) / (nir_band + 6 * red_band - 7.5 * red_band + 1))
        
        return {
            'ndvi': ndvi,
            'evi': evi
        }
    
    def detect_geometric_patterns(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Detect geometric patterns that might indicate human construction"""
        
        # Edge detection and pattern analysis
        from scipy import ndimage
        from skimage import feature, measure
        
        # Edge detection
        edges = feature.canny(image_array)
        
        # Find contours and analyze shapes
        contours = measure.find_contours(edges, 0.8)
        
        patterns = []
        for i, contour in enumerate(contours):
            if len(contour) > 10:  # Filter small noise
                # Calculate basic geometric properties
                area = measure.regionprops(measure.label(edges))[0].area if measure.regionprops(measure.label(edges)) else 0
                perimeter = len(contour)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                patterns.append({
                    'contour_id': i,
                    'area': area,
                    'perimeter': perimeter,
                    'circularity': circularity,
                    'coordinates': contour.tolist(),
                    'regularity_score': self._calculate_regularity(contour)
                })
        
        return patterns
    
    def _calculate_regularity(self, contour: np.ndarray) -> float:
        """Calculate regularity score for geometric patterns"""
        if len(contour) < 4:
            return 0.0
            
        # Simple regularity measure based on angle consistency
        angles = []
        for i in range(len(contour)):
            p1 = contour[i-1]
            p2 = contour[i]
            p3 = contour[(i+1) % len(contour)]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
            angles.append(angle)
        
        # Higher regularity = lower standard deviation of angles
        return 1.0 / (1.0 + np.std(angles))

class KnowledgeBaseBuilder:
    """Create and manage knowledge base for archaeological analysis"""
    
    def __init__(self):
        self.embeddings_model = "text-embedding-3-large"
        self.knowledge_base = []
        
    def build_archaeological_knowledge_base(self) -> Dict[str, Any]:
        """Build knowledge base from archaeological literature and data"""
        
        # Sample archaeological knowledge (in practice, this would be loaded from databases)
        archaeological_contexts = [
            {
                "site_type": "Fortified Settlement",
                "time_period": "1000-1500 CE",
                "characteristics": "Earthworks, defensive structures, elevated positions",
                "amazon_examples": "Monte Alegre, Santarém culture sites",
                "detection_features": "Geometric earthworks, elevated platforms, water access"
            },
            {
                "site_type": "Ceremonial Complex",
                "time_period": "500-1500 CE", 
                "characteristics": "Large earthen mounds, plazas, ritual spaces",
                "amazon_examples": "Marajoara culture sites, Acre geoglyphs",
                "detection_features": "Circular or rectangular earthworks, central plazas"
            },
            {
                "site_type": "Agricultural Terraces",
                "time_period": "1000-1500 CE",
                "characteristics": "Raised fields, drainage systems, forest islands",
                "amazon_examples": "Llanos de Mojos, raised field systems",
                "detection_features": "Linear patterns, water management features"
            }
        ]
        
        return {
            "contexts": archaeological_contexts,
            "total_sites": len(archaeological_contexts),
            "last_updated": datetime.now().isoformat()
        }
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for knowledge base texts"""
        # Placeholder for actual OpenAI embeddings API call
        # In practice, you would use openai.embeddings.create()
        return [[0.1] * 1536 for _ in texts]  # Mock embeddings

class RAGSystem:
    """Retrieval-Augmented Generation system for archaeological analysis"""
    
    def __init__(self, knowledge_base: KnowledgeBaseBuilder, openai_client: OpenAIIntegration):
        self.kb = knowledge_base
        self.openai = openai_client
        self.vector_store = {}
        
    def setup_vector_store(self):
        """Initialize vector store with archaeological knowledge"""
        kb_data = self.kb.build_archaeological_knowledge_base()
        
        for context in kb_data["contexts"]:
            # Create searchable text
            text = f"{context['site_type']} {context['characteristics']} {context['detection_features']}"
            
            # In practice, you would store actual embeddings
            self.vector_store[context['site_type']] = {
                'text': text,
                'metadata': context,
                'embedding': [0.1] * 1536  # Mock embedding
            }
    
    def retrieve_relevant_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant archaeological context for analysis"""
        # Simple keyword matching (in practice, use vector similarity)
        relevant_contexts = []
        query_lower = query.lower()
        
        for site_type, data in self.vector_store.items():
            if any(word in data['text'].lower() for word in query_lower.split()):
                relevant_contexts.append(data['metadata'])
        
        return relevant_contexts
    
    def generate_enhanced_analysis(self, site_data: Dict[str, Any], query: str) -> str:
        """Generate analysis enhanced with archaeological knowledge"""
        
        relevant_contexts = self.retrieve_relevant_context(query)
        
        context_text = "\n".join([
            f"Site Type: {ctx['site_type']}\nCharacteristics: {ctx['characteristics']}\nDetection Features: {ctx['detection_features']}"
            for ctx in relevant_contexts
        ])
        
        enhanced_prompt = f"""
        Based on the following archaeological context:
        {context_text}
        
        And the following site data:
        {json.dumps(site_data, indent=2)}
        
        Provide a comprehensive archaeological analysis including:
        1. Site classification and cultural context
        2. Comparison with known Amazon basin sites
        3. Confidence assessment
        4. Recommended investigation priorities
        """
        
        try:
            response = self.openai.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in Amazon basin archaeology with access to comprehensive site databases."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.5
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in enhanced analysis: {e}"

class ArchaeologicalSiteDetector:
    """Main class orchestrating archaeological site detection"""
    
    def __init__(self, openai_api_key: str):
        self.openai = OpenAIIntegration(openai_api_key)
        self.satellite_processor = SatelliteDataProcessor()
        self.kb_builder = KnowledgeBaseBuilder()
        self.rag_system = RAGSystem(self.kb_builder, self.openai)
        self.rag_system.setup_vector_store()
        
    def analyze_potential_site(self, coordinates: tuple, satellite_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete analysis pipeline for potential archaeological sites"""
        
        # Step 1: Process satellite data
        geometric_patterns = []
        if 'image_array' in satellite_data:
            geometric_patterns = self.satellite_processor.detect_geometric_patterns(satellite_data['image_array'])
        
        # Step 2: Calculate vegetation indices if spectral data available
        vegetation_indices = {}
        if 'red_band' in satellite_data and 'nir_band' in satellite_data:
            vegetation_indices = self.satellite_processor.calculate_vegetation_indices(
                satellite_data['red_band'], 
                satellite_data['nir_band']
            )
        
        # Step 3: OpenAI feature analysis
        feature_analysis = self.openai.analyze_satellite_features({
            'coordinates': coordinates,
            'geometric_patterns': geometric_patterns,
            'vegetation_indices': list(vegetation_indices.keys()),
            'metadata': satellite_data.get('metadata', {})
        })
        
        # Step 4: Generate enhanced analysis using RAG
        enhanced_analysis = self.rag_system.generate_enhanced_analysis(
            {
                'coordinates': coordinates,
                'features': feature_analysis,
                'patterns': geometric_patterns
            },
            f"archaeological site analysis at {coordinates}"
        )
        
        # Step 5: Generate site hypothesis
        hypothesis = self.openai.generate_site_hypothesis(coordinates, feature_analysis)
        
        return {
            'coordinates': coordinates,
            'analysis_timestamp': datetime.now().isoformat(),
            'geometric_patterns': geometric_patterns,
            'vegetation_indices': vegetation_indices,
            'openai_analysis': feature_analysis,
            'enhanced_analysis': enhanced_analysis,
            'site_hypothesis': hypothesis,
            'confidence_score': self._calculate_overall_confidence(feature_analysis, geometric_patterns)
        }
    
    def _calculate_overall_confidence(self, feature_analysis: Dict[str, Any], patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for site detection"""
        
        # Basic confidence calculation
        confidence_factors = []
        
        # Feature analysis confidence
        if 'confidence_scores' in feature_analysis:
            confidence_factors.extend(feature_analysis['confidence_scores'].values())
        
        # Pattern regularity confidence
        if patterns:
            pattern_scores = [p.get('regularity_score', 0) for p in patterns]
            confidence_factors.extend(pattern_scores)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0
    
    def batch_analyze_region(self, region_bounds: Dict[str, float], grid_size: float = 0.01) -> List[Dict[str, Any]]:
        """Analyze entire region using grid-based approach"""
        
        results = []
        
        # Generate grid coordinates
        lat_range = np.arange(region_bounds['min_lat'], region_bounds['max_lat'], grid_size)
        lon_range = np.arange(region_bounds['min_lon'], region_bounds['max_lon'], grid_size)
        
        for lat in lat_range[:10]:  # Limit for demonstration
            for lon in lon_range[:10]:
                coordinates = (lat, lon)
                
                # Mock satellite data (in practice, fetch from satellite data APIs)
                mock_satellite_data = {
                    'metadata': {
                        'acquisition_date': '2024-01-01',
                        'resolution': '30cm',
                        'cloud_cover': 0.1
                    },
                    'image_array': np.random.rand(100, 100),  # Mock image data
                    'red_band': np.random.rand(100, 100),
                    'nir_band': np.random.rand(100, 100)
                }
                
                analysis = self.analyze_potential_site(coordinates, mock_satellite_data)
                
                # Only keep high-confidence results
                if analysis['confidence_score'] > 0.7:
                    results.append(analysis)
        
        return results

def main():
    """Main execution function"""
    
    # Initialize the system
    if not Config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    detector = ArchaeologicalSiteDetector(Config.OPENAI_API_KEY)
    
    # Example: Analyze a specific region in the Amazon
    test_region = {
        'min_lat': -10.0,
        'max_lat': -9.0,
        'min_lon': -65.0,
        'max_lon': -64.0
    }
    
    print("Starting archaeological site analysis...")
    print(f"Region: {test_region}")
    
    # Batch analysis
    results = detector.batch_analyze_region(test_region)
    
    print(f"\nAnalysis complete. Found {len(results)} high-confidence potential sites.")
    
    # Display top results
    for i, result in enumerate(results[:5]):
        print(f"\n--- Potential Site {i+1} ---")
        print(f"Coordinates: {result['coordinates']}")
        print(f"Confidence Score: {result['confidence_score']:.3f}")
        print(f"Geometric Patterns: {len(result['geometric_patterns'])}")
        print(f"Site Hypothesis: {result['site_hypothesis'][:200]}...")
    
    # Save results
    output_file = '/home/myuser/OpenAI_to_Z_Challenge/analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()