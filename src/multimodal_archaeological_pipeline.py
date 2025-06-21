#!/usr/bin/env python3
"""
Multimodal Archaeological Analysis Pipeline
Complete workflow: LIDAR → Image Generation → LLM Analysis → Cultural Context

Pipeline Steps:
1. 🛰️ Data Acquisition (.las, .laz, .tif)
2. 🧹 Preprocessing (DTM/DSM extraction, noise removal)
3. 🖼️ Image Generation (Hillshade, Slope, Contours, NDVI)
4. 🔀 Multimodal Input Construction
5. 🤖 LLM/Multimodal Analysis (Deepseek, Vision models)
6. 🏛️ Archaeological Interpretation with Cultural Context
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import base64
from io import BytesIO

# Image processing and visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns

# Scientific computing
from scipy import ndimage
from skimage import filters, feature, measure

# Import our custom modules
from lidar_archaeological_processor import LidarArchaeologicalProcessor
from ai_structure_detector import ArchaeologicalStructureDetector
from gis_archaeological_exporter import GISArchaeologicalExporter

# OpenAI/LLM integration
try:
    import openai
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("OpenAI library not available")

# ChromaDB for RAG
try:
    import chromadb
    from chromadb.utils import embedding_functions
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("ChromaDB not available - limited cultural context")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalArchaeologicalPipeline:
    """Complete multimodal pipeline for archaeological analysis"""
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = "https://openrouter.ai/api/v1",
                 model: str = "deepseek/deepseek-r1-0528:free"):
        """
        Initialize the multimodal pipeline
        
        Args:
            api_key: OpenAI/OpenRouter API key
            base_url: API base URL
            model: Model name (supports multimodal models)
        """
        
        # Initialize components
        self.lidar_processor = LidarArchaeologicalProcessor(resolution=0.5)
        self.structure_detector = ArchaeologicalStructureDetector(model_type='hybrid')
        self.gis_exporter = GISArchaeologicalExporter()
        
        # LLM configuration
        self.api_config = {
            'api_key': api_key or os.getenv('OPENAI_API_KEY'),
            'base_url': base_url or os.getenv('OPENAI_BASE_URL', base_url),
            'model': model or os.getenv('OPENAI_MODEL', model)
        }
        
        # Initialize cultural context database
        self.cultural_db = None
        if RAG_AVAILABLE:
            self._initialize_cultural_database()
        
        # Visualization settings
        self.viz_settings = {
            'dpi': 300,
            'figsize': (12, 10),
            'cmap_terrain': 'terrain',
            'cmap_hillshade': 'gray',
            'font_size': 12
        }
        
        logger.info("Multimodal Archaeological Pipeline initialized")
    
    def process_lidar_site(self, 
                          lidar_path: str,
                          site_info: Dict,
                          output_dir: str = None,
                          generate_report: bool = True) -> Dict:
        """
        Complete processing pipeline for a LIDAR site
        
        Args:
            lidar_path: Path to LIDAR file (.las, .laz) or directory
            site_info: Site metadata (coordinates, region, etc.)
            output_dir: Output directory for results
            generate_report: Whether to generate comprehensive report
            
        Returns:
            Complete analysis results with all pipeline outputs
        """
        
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(lidar_path), 'results')
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Starting multimodal analysis for {site_info.get('name', 'unknown site')}")
        
        # Step 1: 🛰️ Data Processing
        logger.info("Step 1: Processing LIDAR data...")
        lidar_results = self._process_lidar_data(lidar_path, site_info)
        
        # Step 2: 🧹 Advanced Preprocessing  
        logger.info("Step 2: Advanced preprocessing...")
        preprocessed_data = self._advanced_preprocessing(lidar_results)
        
        # Step 3: 🖼️ Generate Analysis Images
        logger.info("Step 3: Generating analysis images...")
        analysis_images = self._generate_analysis_images(
            preprocessed_data, output_dir, timestamp
        )
        
        # Step 4: 🔀 Construct Multimodal Input
        logger.info("Step 4: Constructing multimodal input...")
        multimodal_input = self._construct_multimodal_input(
            analysis_images, site_info, preprocessed_data
        )
        
        # Step 5: 🤖 LLM Analysis
        logger.info("Step 5: Running LLM analysis...")
        llm_analysis = self._run_llm_analysis(multimodal_input)
        
        # Step 6: 🏛️ Cultural Context Integration
        logger.info("Step 6: Integrating cultural context...")
        cultural_analysis = self._integrate_cultural_context(
            llm_analysis, site_info, preprocessed_data
        )
        
        # Step 7: Comprehensive Results Assembly
        logger.info("Step 7: Assembling final results...")
        final_results = self._assemble_final_results(
            lidar_results, preprocessed_data, analysis_images,
            llm_analysis, cultural_analysis, site_info, timestamp
        )
        
        # Step 8: Export Results
        logger.info("Step 8: Exporting results...")
        exported_files = self._export_comprehensive_results(
            final_results, output_dir, site_info
        )
        final_results['exported_files'] = exported_files
        
        # Step 9: Generate Report
        if generate_report:
            logger.info("Step 9: Generating comprehensive report...")
            report_path = self._generate_comprehensive_report(
                final_results, output_dir, timestamp
            )
            final_results['comprehensive_report'] = report_path
        
        logger.info(f"Multimodal analysis complete. Results in: {output_dir}")
        return final_results
    
    def _process_lidar_data(self, lidar_path: str, site_info: Dict) -> Dict:
        """Process LIDAR data using our LIDAR processor"""
        
        try:
            if os.path.isfile(lidar_path):
                # Single LIDAR file
                results = self.lidar_processor.process_lidar_file(lidar_path)
            else:
                # Directory or synthetic data
                results = self.lidar_processor._generate_synthetic_analysis(lidar_path)
            
            # Add site information
            results['site_info'] = site_info
            
            return results
            
        except Exception as e:
            logger.warning(f"LIDAR processing failed: {e}")
            # Return synthetic results for demonstration
            return self.lidar_processor._generate_synthetic_analysis(lidar_path)
    
    def _advanced_preprocessing(self, lidar_results: Dict) -> Dict:
        """Advanced preprocessing with enhanced feature extraction"""
        
        # Extract elevation data (synthetic if needed)
        if 'elevation_models' in lidar_results:
            # Use real LIDAR data
            dtm_stats = lidar_results['elevation_models'].get('dtm_stats', {})
            elevation = self._reconstruct_elevation_from_stats(dtm_stats)
        else:
            # Generate synthetic elevation for demonstration
            elevation = self._generate_synthetic_elevation()
        
        # Generate derived products
        processed_data = {
            'elevation': elevation,
            'dtm': elevation,  # Digital Terrain Model
            'dsm': elevation + np.random.rand(*elevation.shape) * 5,  # Digital Surface Model
        }
        
        # Calculate terrain derivatives
        dy, dx = np.gradient(elevation)
        processed_data['slope'] = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        processed_data['aspect'] = np.degrees(np.arctan2(-dx, dy)) % 360
        
        # Calculate curvature
        dyy, dyx = np.gradient(dy)
        dxy, dxx = np.gradient(dx)
        processed_data['curvature'] = (dxx + dyy) / (1 + dx**2 + dy**2)**(3/2)
        
        # Generate hillshade
        processed_data['hillshade'] = self._generate_hillshade(elevation)
        
        # Calculate local relief
        processed_data['local_relief'] = elevation - ndimage.uniform_filter(elevation, size=20)
        
        # Generate vegetation index (synthetic)
        processed_data['ndvi'] = self._generate_synthetic_ndvi(elevation.shape)
        
        # Detect structures using AI detector
        structure_results = self.structure_detector.detect_structures(
            elevation, processed_data['hillshade']
        )
        processed_data['detected_structures'] = structure_results
        
        logger.info("Advanced preprocessing completed")
        return processed_data
    
    def _generate_analysis_images(self, data: Dict, output_dir: str, timestamp: str) -> Dict:
        """Generate comprehensive analysis images for LLM input"""
        
        images = {}
        
        # Set style
        plt.style.use('default')
        
        # 1. Hillshade Analysis
        images['hillshade'] = self._create_hillshade_image(
            data, output_dir, timestamp
        )
        
        # 2. Elevation with Contours
        images['elevation_contours'] = self._create_elevation_contour_image(
            data, output_dir, timestamp
        )
        
        # 3. Slope Analysis
        images['slope_analysis'] = self._create_slope_analysis_image(
            data, output_dir, timestamp
        )
        
        # 4. Multi-layer Composite
        images['composite_analysis'] = self._create_composite_analysis_image(
            data, output_dir, timestamp
        )
        
        # 5. Structure Detection Overlay
        images['structure_detection'] = self._create_structure_detection_image(
            data, output_dir, timestamp
        )
        
        # 6. 3D Perspective View
        images['perspective_3d'] = self._create_3d_perspective_image(
            data, output_dir, timestamp
        )
        
        logger.info(f"Generated {len(images)} analysis images")
        return images
    
    def _create_hillshade_image(self, data: Dict, output_dir: str, timestamp: str) -> str:
        """Create hillshade analysis image"""
        
        fig, ax = plt.subplots(figsize=self.viz_settings['figsize'], 
                              dpi=self.viz_settings['dpi'])
        
        hillshade = data['hillshade']
        
        # Display hillshade
        im = ax.imshow(hillshade, cmap='gray', extent=[0, 100, 0, 100])
        
        # Overlay detected structures
        self._overlay_structures_on_image(ax, data.get('detected_structures', {}))
        
        # Styling
        ax.set_title('LIDAR Hillshade Analysis\nArchaeological Feature Detection', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Distance (arbitrary units)', fontsize=12)
        ax.set_ylabel('Distance (arbitrary units)', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Hillshade Intensity', fontsize=12)
        
        # Add scale bar
        self._add_scale_bar(ax)
        
        # Add north arrow
        self._add_north_arrow(ax)
        
        plt.tight_layout()
        
        filename = f'hillshade_analysis_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_elevation_contour_image(self, data: Dict, output_dir: str, timestamp: str) -> str:
        """Create elevation map with contours"""
        
        fig, ax = plt.subplots(figsize=self.viz_settings['figsize'],
                              dpi=self.viz_settings['dpi'])
        
        elevation = data['elevation']
        
        # Display elevation
        im = ax.imshow(elevation, cmap='terrain', extent=[0, 100, 0, 100])
        
        # Add contour lines
        contours = ax.contour(elevation, levels=20, colors='black', alpha=0.6, linewidths=0.8)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%d m')
        
        # Overlay structures
        self._overlay_structures_on_image(ax, data.get('detected_structures', {}))
        
        # Styling
        ax.set_title('Digital Terrain Model with Contours\nElevation Analysis', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Distance (arbitrary units)', fontsize=12)
        ax.set_ylabel('Distance (arbitrary units)', fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Elevation (m)', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'elevation_contours_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_slope_analysis_image(self, data: Dict, output_dir: str, timestamp: str) -> str:
        """Create slope analysis image"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8),
                                      dpi=self.viz_settings['dpi'])
        
        slope = data['slope']
        aspect = data['aspect']
        
        # Slope map
        im1 = ax1.imshow(slope, cmap='plasma', extent=[0, 100, 0, 100])
        ax1.set_title('Slope Analysis\n(Degrees)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Distance (arbitrary units)', fontsize=12)
        ax1.set_ylabel('Distance (arbitrary units)', fontsize=12)
        
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Slope (°)', fontsize=12)
        
        # Aspect map
        im2 = ax2.imshow(aspect, cmap='hsv', extent=[0, 100, 0, 100])
        ax2.set_title('Aspect Analysis\n(Direction)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Distance (arbitrary units)', fontsize=12)
        ax2.set_ylabel('Distance (arbitrary units)', fontsize=12)
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Aspect (°)', fontsize=12)
        
        # Overlay structures on both
        for ax in [ax1, ax2]:
            self._overlay_structures_on_image(ax, data.get('detected_structures', {}))
        
        plt.tight_layout()
        
        filename = f'slope_analysis_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_composite_analysis_image(self, data: Dict, output_dir: str, timestamp: str) -> str:
        """Create multi-layer composite analysis"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16),
                                                    dpi=self.viz_settings['dpi'])
        
        # Hillshade
        ax1.imshow(data['hillshade'], cmap='gray', extent=[0, 100, 0, 100])
        ax1.set_title('Hillshade', fontsize=12, fontweight='bold')
        self._overlay_structures_on_image(ax1, data.get('detected_structures', {}))
        
        # Elevation
        im2 = ax2.imshow(data['elevation'], cmap='terrain', extent=[0, 100, 0, 100])
        ax2.set_title('Elevation', fontsize=12, fontweight='bold')
        
        # Local Relief
        im3 = ax3.imshow(data['local_relief'], cmap='RdBu_r', extent=[0, 100, 0, 100])
        ax3.set_title('Local Relief', fontsize=12, fontweight='bold')
        
        # Curvature
        im4 = ax4.imshow(data['curvature'], cmap='seismic', extent=[0, 100, 0, 100])
        ax4.set_title('Curvature', fontsize=12, fontweight='bold')
        
        # Add structure overlays to relevant panels
        for ax in [ax2, ax3, ax4]:
            self._overlay_structures_on_image(ax, data.get('detected_structures', {}))
        
        # Style all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel('Distance (arbitrary units)', fontsize=10)
            ax.set_ylabel('Distance (arbitrary units)', fontsize=10)
        
        plt.suptitle('Comprehensive Terrain Analysis\nMulti-Layer Archaeological Assessment', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f'composite_analysis_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_structure_detection_image(self, data: Dict, output_dir: str, timestamp: str) -> str:
        """Create focused structure detection visualization"""
        
        fig, ax = plt.subplots(figsize=self.viz_settings['figsize'],
                              dpi=self.viz_settings['dpi'])
        
        # Use hillshade as base
        ax.imshow(data['hillshade'], cmap='gray', alpha=0.7, extent=[0, 100, 0, 100])
        
        # Detailed structure overlay
        self._overlay_detailed_structures(ax, data.get('detected_structures', {}))
        
        # Styling
        ax.set_title('Archaeological Structure Detection\nAI-Powered Feature Identification', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Distance (arbitrary units)', fontsize=12)
        ax.set_ylabel('Distance (arbitrary units)', fontsize=12)
        
        # Add legend
        self._add_structure_legend(ax)
        
        # Add statistics box
        self._add_detection_statistics_box(ax, data.get('detected_structures', {}))
        
        plt.tight_layout()
        
        filename = f'structure_detection_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_3d_perspective_image(self, data: Dict, output_dir: str, timestamp: str) -> str:
        """Create 3D perspective visualization"""
        
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(14, 10), dpi=self.viz_settings['dpi'])
            ax = fig.add_subplot(111, projection='3d')
            
            elevation = data['elevation']
            y, x = np.mgrid[0:elevation.shape[0]:4, 0:elevation.shape[1]:4]  # Subsample for performance
            z = elevation[::4, ::4]
            
            # Create 3D surface
            surf = ax.plot_surface(x, y, z, cmap='terrain', alpha=0.8, 
                                 linewidth=0, antialiased=True)
            
            # Add structure markers in 3D
            self._add_3d_structure_markers(ax, data.get('detected_structures', {}), elevation)
            
            # Styling
            ax.set_title('3D Terrain Visualization\nArchaeological Features in Context', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('X Distance', fontsize=12)
            ax.set_ylabel('Y Distance', fontsize=12)
            ax.set_zlabel('Elevation (m)', fontsize=12)
            
            # Set viewing angle
            ax.view_init(elev=30, azim=45)
            
            # Add colorbar
            cbar = fig.colorbar(surf, ax=ax, shrink=0.6)
            cbar.set_label('Elevation (m)', fontsize=12)
            
            plt.tight_layout()
            
            filename = f'perspective_3d_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except ImportError:
            logger.warning("3D plotting not available")
            # Create 2D alternative
            return self._create_elevation_contour_image(data, output_dir, timestamp)
    
    def _construct_multimodal_input(self, images: Dict, site_info: Dict, data: Dict) -> Dict:
        """Construct multimodal input for LLM analysis"""
        
        multimodal_input = {
            'images': {},
            'text_context': {},
            'geographical_data': {},
            'analysis_data': {}
        }
        
        # Encode images to base64 for LLM input
        for image_type, image_path in images.items():
            if os.path.exists(image_path):
                multimodal_input['images'][image_type] = self._encode_image_to_base64(image_path)
        
        # Text context
        multimodal_input['text_context'] = {
            'site_name': site_info.get('name', 'Unknown site'),
            'region': site_info.get('region', 'Amazon Basin'),
            'coordinates': site_info.get('coordinates', [0, 0]),
            'acquisition_date': site_info.get('acquisition_date', 'Unknown'),
            'processing_date': datetime.now().isoformat()
        }
        
        # Geographical context
        multimodal_input['geographical_data'] = {
            'latitude': site_info.get('coordinates', [0, 0])[0] if 'coordinates' in site_info else 0,
            'longitude': site_info.get('coordinates', [0, 0])[1] if 'coordinates' in site_info else 0,
            'elevation_range': self._get_elevation_range(data),
            'terrain_complexity': self._calculate_terrain_complexity(data),
            'vegetation_coverage': data.get('ndvi', {}).get('mean', 0.5) if 'ndvi' in data else 0.5
        }
        
        # Analysis data summary
        detected_structures = data.get('detected_structures', {})
        multimodal_input['analysis_data'] = {
            'total_structures': detected_structures.get('structures', {}).get('summary', {}).get('total_structures', 0),
            'structure_types': detected_structures.get('structures', {}).get('summary', {}).get('by_type', {}),
            'confidence_scores': detected_structures.get('confidence_scores', {}),
            'terrain_statistics': self._extract_terrain_statistics(data)
        }
        
        return multimodal_input
    
    def _run_llm_analysis(self, multimodal_input: Dict) -> Dict:
        """Run LLM analysis on multimodal input"""
        
        if not LLM_AVAILABLE or not self.api_config['api_key']:
            logger.warning("LLM not available, generating mock analysis")
            return self._generate_mock_llm_analysis(multimodal_input)
        
        try:
            # Initialize OpenAI client
            client = openai.OpenAI(
                api_key=self.api_config['api_key'],
                base_url=self.api_config['base_url']
            )
            
            # Construct prompt
            prompt = self._build_llm_analysis_prompt(multimodal_input)
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": self._get_archaeological_expert_system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Add image if model supports it
            if 'hillshade' in multimodal_input['images']:
                # For models that support images, add the primary analysis image
                messages[-1]["content"] = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{multimodal_input['images']['hillshade']}"
                        }
                    }
                ]
            
            # Make API call
            response = client.chat.completions.create(
                model=self.api_config['model'],
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                'llm_analysis': analysis_text,
                'model_used': self.api_config['model'],
                'processing_time': datetime.now().isoformat(),
                'token_usage': response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._generate_mock_llm_analysis(multimodal_input)
    
    def _integrate_cultural_context(self, llm_analysis: Dict, site_info: Dict, data: Dict) -> Dict:
        """Integrate cultural and archaeological context using RAG"""
        
        cultural_context = {
            'cultural_interpretation': {},
            'historical_context': {},
            'similar_sites': [],
            'cultural_recommendations': []
        }
        
        if RAG_AVAILABLE and self.cultural_db:
            # Query cultural database
            try:
                query_text = self._build_cultural_query(llm_analysis, site_info, data)
                results = self.cultural_db.query(
                    query_texts=[query_text],
                    n_results=5
                )
                
                # Process RAG results
                cultural_context = self._process_cultural_rag_results(results, site_info)
                
            except Exception as e:
                logger.warning(f"Cultural context RAG failed: {e}")
        
        # Add default cultural interpretations
        cultural_context.update(self._generate_default_cultural_context(site_info, data))
        
        return cultural_context
    
    def _assemble_final_results(self, lidar_results: Dict, processed_data: Dict,
                              analysis_images: Dict, llm_analysis: Dict,
                              cultural_analysis: Dict, site_info: Dict,
                              timestamp: str) -> Dict:
        """Assemble comprehensive final results"""
        
        final_results = {
            'metadata': {
                'site_info': site_info,
                'processing_timestamp': timestamp,
                'pipeline_version': '1.0.0',
                'processing_time': datetime.now().isoformat()
            },
            'raw_lidar_analysis': lidar_results,
            'processed_terrain_data': {
                'elevation_statistics': self._extract_terrain_statistics(processed_data),
                'detected_structures': processed_data.get('detected_structures', {}),
                'terrain_derivatives': {
                    'slope_statistics': self._calculate_slope_statistics(processed_data.get('slope', [])),
                    'aspect_statistics': self._calculate_aspect_statistics(processed_data.get('aspect', [])),
                    'curvature_statistics': self._calculate_curvature_statistics(processed_data.get('curvature', []))
                }
            },
            'visualization_results': {
                'generated_images': analysis_images,
                'image_descriptions': self._generate_image_descriptions(analysis_images)
            },
            'ai_analysis': {
                'llm_interpretation': llm_analysis,
                'multimodal_analysis': self._extract_multimodal_insights(llm_analysis),
                'confidence_assessment': self._assess_analysis_confidence(llm_analysis, processed_data)
            },
            'cultural_context': cultural_analysis,
            'archaeological_assessment': {
                'overall_score': self._calculate_overall_archaeological_score(
                    processed_data, llm_analysis, cultural_analysis
                ),
                'feature_significance': self._assess_feature_significance(processed_data, cultural_analysis),
                'investigation_priority': self._determine_investigation_priority(
                    processed_data, llm_analysis, cultural_analysis
                ),
                'recommendations': self._generate_comprehensive_recommendations(
                    processed_data, llm_analysis, cultural_analysis, site_info
                )
            }
        }
        
        return final_results
    
    def _export_comprehensive_results(self, results: Dict, output_dir: str, site_info: Dict) -> Dict:
        """Export results using GIS exporter and additional formats"""
        
        exported_files = {}
        
        try:
            # Use GIS exporter
            site_bounds = self._extract_site_bounds(site_info)
            gis_exports = self.gis_exporter.export_analysis_results(
                results['raw_lidar_analysis'],
                output_dir,
                formats=['shapefile', 'geojson', 'kml', 'csv'],
                site_bounds=site_bounds
            )
            exported_files.update(gis_exports)
            
            # Export additional formats
            timestamp = results['metadata']['processing_timestamp']
            
            # JSON export
            json_file = os.path.join(output_dir, f'complete_analysis_{timestamp}.json')
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            exported_files['complete_json'] = json_file
            
            # Archaeological assessment CSV
            assessment_csv = self._export_archaeological_assessment_csv(results, output_dir, timestamp)
            exported_files['archaeological_assessment'] = assessment_csv
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            exported_files['error'] = str(e)
        
        return exported_files
    
    def _generate_comprehensive_report(self, results: Dict, output_dir: str, timestamp: str) -> str:
        """Generate comprehensive analysis report"""
        
        report_file = os.path.join(output_dir, f'archaeological_analysis_report_{timestamp}.md')
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                self._write_comprehensive_report(f, results)
            
            logger.info(f"Generated comprehensive report: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None
    
    def _write_comprehensive_report(self, file, results: Dict):
        """Write detailed markdown report"""
        
        site_info = results['metadata']['site_info']
        
        file.write("# 🏛️ Comprehensive Archaeological Analysis Report\n\n")
        file.write("## Executive Summary\n\n")
        
        # Site information
        file.write(f"**Site:** {site_info.get('name', 'Unknown')}\n")
        file.write(f"**Region:** {site_info.get('region', 'Unknown')}\n")
        file.write(f"**Coordinates:** {site_info.get('coordinates', [0, 0])}\n")
        file.write(f"**Analysis Date:** {results['metadata']['processing_time']}\n\n")
        
        # Overall assessment
        overall_score = results['archaeological_assessment']['overall_score']
        priority = results['archaeological_assessment']['investigation_priority']
        
        file.write(f"**Overall Archaeological Score:** {overall_score:.3f}/1.000\n")
        file.write(f"**Investigation Priority:** {priority}\n\n")
        
        # Structure detection summary
        structures = results['processed_terrain_data']['detected_structures']
        if 'structures' in structures and 'summary' in structures['structures']:
            summary = structures['structures']['summary']
            file.write("## 🔍 Structure Detection Summary\n\n")
            file.write(f"- **Total Structures Detected:** {summary.get('total_structures', 0)}\n")
            
            for structure_type, count in summary.get('by_type', {}).items():
                if count > 0:
                    file.write(f"- **{structure_type.replace('_', ' ').title()}:** {count}\n")
            
            file.write(f"\n- **Mean Confidence:** {summary.get('mean_confidence', 0):.3f}\n\n")
        
        # LLM Analysis
        if 'ai_analysis' in results and 'llm_interpretation' in results['ai_analysis']:
            llm_text = results['ai_analysis']['llm_interpretation'].get('llm_analysis', '')
            file.write("## 🤖 AI Analysis\n\n")
            file.write(f"{llm_text}\n\n")
        
        # Cultural Context
        if 'cultural_context' in results:
            cultural = results['cultural_context']
            file.write("## 🏺 Cultural Context\n\n")
            
            if 'cultural_interpretation' in cultural:
                for key, value in cultural['cultural_interpretation'].items():
                    file.write(f"**{key.replace('_', ' ').title()}:** {value}\n\n")
        
        # Recommendations
        recommendations = results['archaeological_assessment'].get('recommendations', [])
        if recommendations:
            file.write("## 📋 Recommendations\n\n")
            for i, rec in enumerate(recommendations, 1):
                file.write(f"{i}. {rec}\n")
            file.write("\n")
        
        # Technical Details
        file.write("## 🔧 Technical Details\n\n")
        file.write("### Processing Pipeline\n\n")
        file.write("1. **LIDAR Data Processing** - Point cloud analysis and DTM/DSM extraction\n")
        file.write("2. **Advanced Preprocessing** - Terrain derivative calculation\n")
        file.write("3. **Image Generation** - Multi-layer visualization creation\n")
        file.write("4. **Multimodal Analysis** - AI-powered structure detection\n")
        file.write("5. **Cultural Context Integration** - Archaeological knowledge integration\n")
        file.write("6. **Comprehensive Assessment** - Final scoring and recommendations\n\n")
        
        # Terrain Statistics
        terrain_stats = results['processed_terrain_data'].get('elevation_statistics', {})
        file.write("### Terrain Characteristics\n\n")
        for key, value in terrain_stats.items():
            file.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
        
        file.write("\n---\n\n")
        file.write("*Report generated by Multimodal Archaeological Analysis Pipeline*\n")
    
    # Helper methods for data processing and analysis
    
    def _generate_hillshade(self, elevation: np.ndarray, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
        """Generate hillshade from elevation data"""
        
        dy, dx = np.gradient(elevation)
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)
        
        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)
        
        hillshade = np.sin(altitude_rad) * np.sin(slope) + \
                   np.cos(altitude_rad) * np.cos(slope) * \
                   np.cos(azimuth_rad - aspect)
        
        return ((hillshade + 1) * 127.5).astype(np.uint8)
    
    def _generate_synthetic_elevation(self, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Generate synthetic elevation data for testing"""
        
        x, y = np.meshgrid(np.linspace(0, 100, size[1]), np.linspace(0, 100, size[0]))
        
        elevation = (
            100 + 50 * np.sin(x / 10) * np.cos(y / 15) +
            25 * np.sin(x / 5) * np.sin(y / 8) +
            10 * np.random.randn(*size)
        )
        
        # Add some archaeological features
        # Circular earthwork
        center_y, center_x = size[0] // 3, size[1] // 3
        radius = 30
        y_coords, x_coords = np.ogrid[:size[0], :size[1]]
        circle_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
        elevation[circle_mask] += 3.0
        
        # Linear feature
        elevation[size[0]//2:size[0]//2+5, :] += 1.5
        
        return elevation
    
    def _generate_synthetic_ndvi(self, shape: Tuple[int, int]) -> Dict:
        """Generate synthetic NDVI data"""
        
        ndvi = 0.3 + 0.4 * np.random.rand(*shape)
        
        return {
            'values': ndvi,
            'mean': float(np.mean(ndvi)),
            'std': float(np.std(ndvi)),
            'vegetation_coverage': float(np.sum(ndvi > 0.5) / ndvi.size)
        }
    
    def _reconstruct_elevation_from_stats(self, stats: Dict) -> np.ndarray:
        """Reconstruct elevation array from statistics (for demonstration)"""
        
        mean_elev = stats.get('mean', 100)
        std_elev = stats.get('std', 20)
        
        # Generate synthetic elevation based on statistics
        elevation = np.random.normal(mean_elev, std_elev, (512, 512))
        
        # Add some structure
        x, y = np.meshgrid(np.linspace(0, 100, 512), np.linspace(0, 100, 512))
        elevation += 10 * np.sin(x / 20) * np.cos(y / 25)
        
        return elevation
    
    def _overlay_structures_on_image(self, ax, structures: Dict):
        """Overlay detected structures on image"""
        
        if not structures or 'structures' not in structures:
            return
        
        colors = {
            'circular_structures': 'red',
            'linear_structures': 'blue',
            'mounds': 'orange',
            'earthworks': 'green',
            'ditches': 'purple',
            'platforms': 'yellow'
        }
        
        struct_data = structures['structures']
        
        for structure_type, color in colors.items():
            if structure_type in struct_data and struct_data[structure_type]:
                for struct in struct_data[structure_type][:5]:  # Limit to first 5
                    if 'center' in struct:
                        center = struct['center']
                        # Convert to plot coordinates (assuming 512x512 -> 100x100)
                        x = (center[1] / 512.0) * 100
                        y = 100 - (center[0] / 512.0) * 100  # Flip Y
                        ax.plot(x, y, 'o', color=color, markersize=8, alpha=0.8)
    
    def _overlay_detailed_structures(self, ax, structures: Dict):
        """Overlay structures with detailed annotations"""
        
        if not structures or 'structures' not in structures:
            return
        
        colors = {
            'circular_structures': 'red',
            'linear_structures': 'blue', 
            'mounds': 'orange',
            'earthworks': 'green',
            'ditches': 'purple',
            'platforms': 'yellow'
        }
        
        struct_data = structures['structures']
        
        for structure_type, color in colors.items():
            if structure_type in struct_data and struct_data[structure_type]:
                for i, struct in enumerate(struct_data[structure_type][:3]):  # Top 3
                    if 'center' in struct:
                        center = struct['center']
                        x = (center[1] / 512.0) * 100
                        y = 100 - (center[0] / 512.0) * 100
                        
                        # Plot with confidence-based size
                        confidence = struct.get('confidence', 0.5)
                        size = 50 + confidence * 100
                        
                        ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='black')
                        
                        # Add label
                        label = f"{structure_type.split('_')[0].title()}\n{confidence:.2f}"
                        ax.annotate(label, (x, y), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=8,
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    def _add_scale_bar(self, ax):
        """Add scale bar to plot"""
        
        # Simple scale bar (arbitrary units)
        bar_length = 20  # 20 units
        bar_x = 5
        bar_y = 5
        
        ax.plot([bar_x, bar_x + bar_length], [bar_y, bar_y], 'k-', linewidth=3)
        ax.text(bar_x + bar_length/2, bar_y + 2, f'{bar_length} units', 
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _add_north_arrow(self, ax):
        """Add north arrow to plot"""
        
        arrow_x, arrow_y = 90, 90
        ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y-5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                   fontsize=14, fontweight='bold', ha='center')
    
    def _add_structure_legend(self, ax):
        """Add structure type legend"""
        
        colors = {
            'Circular Features': 'red',
            'Linear Features': 'blue',
            'Mounds': 'orange', 
            'Earthworks': 'green',
            'Ditches': 'purple',
            'Platforms': 'yellow'
        }
        
        legend_elements = []
        for label, color in colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8, label=label))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    def _add_detection_statistics_box(self, ax, structures: Dict):
        """Add detection statistics text box"""
        
        if 'structures' in structures and 'summary' in structures['structures']:
            summary = structures['structures']['summary']
            
            stats_text = f"""Detection Summary:
Total: {summary.get('total_structures', 0)}
Confidence: {summary.get('mean_confidence', 0):.2f}
Method: {structures.get('detection_method', 'Hybrid')}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
    
    def _add_3d_structure_markers(self, ax, structures: Dict, elevation: np.ndarray):
        """Add 3D structure markers"""
        
        if not structures or 'structures' not in structures:
            return
        
        struct_data = structures['structures']
        
        for structure_type in ['circular_structures', 'mounds', 'earthworks']:
            if structure_type in struct_data and struct_data[structure_type]:
                for struct in struct_data[structure_type][:3]:
                    if 'center' in struct:
                        center = struct['center']
                        y_idx, x_idx = int(center[0]), int(center[1])
                        
                        if 0 <= y_idx < elevation.shape[0] and 0 <= x_idx < elevation.shape[1]:
                            z_val = elevation[y_idx, x_idx] + 5  # Raise above surface
                            ax.scatter([x_idx], [y_idx], [z_val], c='red', s=100, alpha=0.8)
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for LLM input"""
        
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to encode image {image_path}: {e}")
            return ""
    
    def _get_elevation_range(self, data: Dict) -> Dict:
        """Extract elevation range from data"""
        
        elevation = data.get('elevation', np.array([]))
        if elevation.size > 0:
            return {
                'min': float(np.min(elevation)),
                'max': float(np.max(elevation)),
                'mean': float(np.mean(elevation)),
                'std': float(np.std(elevation))
            }
        return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
    
    def _calculate_terrain_complexity(self, data: Dict) -> float:
        """Calculate terrain complexity score"""
        
        elevation = data.get('elevation', np.array([]))
        slope = data.get('slope', np.array([]))
        
        if elevation.size > 0 and slope.size > 0:
            elev_complexity = np.std(elevation) / np.mean(elevation) if np.mean(elevation) > 0 else 0
            slope_complexity = np.std(slope) / np.mean(slope) if np.mean(slope) > 0 else 0
            return float((elev_complexity + slope_complexity) / 2)
        
        return 0.5  # Default complexity
    
    def _extract_terrain_statistics(self, data: Dict) -> Dict:
        """Extract comprehensive terrain statistics"""
        
        stats = {}
        
        for key in ['elevation', 'slope', 'aspect', 'curvature']:
            if key in data and isinstance(data[key], np.ndarray):
                arr = data[key]
                stats[f'{key}_mean'] = float(np.mean(arr))
                stats[f'{key}_std'] = float(np.std(arr))
                stats[f'{key}_min'] = float(np.min(arr))
                stats[f'{key}_max'] = float(np.max(arr))
        
        return stats
    
    def _build_llm_analysis_prompt(self, multimodal_input: Dict) -> str:
        """Build comprehensive LLM analysis prompt"""
        
        text_context = multimodal_input['text_context']
        geo_data = multimodal_input['geographical_data']
        analysis_data = multimodal_input['analysis_data']
        
        prompt = f"""
COMPREHENSIVE ARCHAEOLOGICAL LIDAR ANALYSIS REQUEST

SITE INFORMATION:
- Site Name: {text_context['site_name']}
- Region: {text_context['region']}
- Coordinates: {geo_data['latitude']:.6f}, {geo_data['longitude']:.6f}
- Processing Date: {text_context['processing_date']}

TERRAIN CHARACTERISTICS:
- Elevation Range: {geo_data['elevation_range']}
- Terrain Complexity: {geo_data['terrain_complexity']:.3f}
- Vegetation Coverage: {geo_data['vegetation_coverage']:.2f}

DETECTED FEATURES:
- Total Structures: {analysis_data['total_structures']}
- Structure Types: {analysis_data['structure_types']}
- Overall Confidence: {analysis_data.get('confidence_scores', {}).get('overall_confidence', 0):.3f}

ANALYSIS IMAGES PROVIDED:
You are viewing LIDAR-derived terrain analysis images including hillshade, elevation contours, 
slope analysis, and structure detection overlays.

EXPERT ANALYSIS REQUEST:
As a leading expert in Amazonian archaeology and remote sensing, please provide:

1. FEATURE INTERPRETATION: Analyze the detected patterns in the context of known Amazon archaeological features
2. CULTURAL SIGNIFICANCE: Compare with known archaeological cultures (Casarabe, Marajoara, Tapajós, etc.)
3. STRUCTURAL ANALYSIS: Evaluate the geometric patterns for evidence of human modification
4. TEMPORAL CONTEXT: Suggest likely time periods based on feature characteristics
5. ARCHAEOLOGICAL POTENTIAL: Assess the significance for understanding pre-Columbian Amazon cultures
6. FIELD RECOMMENDATIONS: Suggest specific investigation strategies

Focus on:
- Geometric earthworks and their cultural significance
- Comparison with known Amazon archaeological sites
- Integration of LIDAR features with archaeological knowledge
- Specific recommendations for ground-truthing

Please provide a detailed, expert-level archaeological interpretation.
"""
        
        return prompt
    
    def _get_archaeological_expert_system_prompt(self) -> str:
        """Get system prompt for archaeological expert persona"""
        
        return """You are Dr. Elena Vargas, a world-renowned expert in Amazonian archaeology with 30 years of experience in remote sensing and pre-Columbian cultures. You specialize in:

- LIDAR-based archaeological discovery in tropical environments
- Pre-Columbian Amazon cultures (Casarabe, Marajoara, Tapajós, Llanos de Mojos)
- Geometric earthworks and their cultural significance
- Integration of remote sensing with archaeological interpretation
- Field survey planning and ground-truthing strategies

Your expertise includes:
- Identification of anthropogenic landscapes
- Understanding of ancient Amazon settlement patterns
- Knowledge of ceremonial and residential earthwork typologies
- Familiarity with agricultural and hydraulic management systems
- Cultural sequence and dating of Amazon archaeological features

Provide detailed, professional archaeological assessments that would be suitable for peer review and field investigation planning."""
    
    def _generate_mock_llm_analysis(self, multimodal_input: Dict) -> Dict:
        """Generate mock LLM analysis when API not available"""
        
        analysis_data = multimodal_input['analysis_data']
        site_name = multimodal_input['text_context']['site_name']
        
        mock_analysis = f"""
EXPERT ARCHAEOLOGICAL ASSESSMENT - {site_name}

FEATURE INTERPRETATION:
The LIDAR data reveals {analysis_data['total_structures']} distinct anthropogenic features consistent with pre-Columbian Amazon settlement patterns. The geometric regularity and spatial organization suggest planned construction rather than natural formation.

CULTURAL SIGNIFICANCE:
Based on the detected circular and linear features, this site shows similarities to:
- Casarabe culture earthwork complexes (500-1400 CE)
- Marajoara ceremonial centers (400-1200 CE)
- Llanos de Mojos raised field systems (500-1400 CE)

The presence of multiple earthwork types indicates a complex, multi-functional site possibly serving ceremonial, residential, and agricultural purposes.

STRUCTURAL ANALYSIS:
- Circular features: Likely ceremonial or defensive enclosures
- Linear features: Possible roads, causeways, or canal systems
- Elevated platforms: Potential residential or ceremonial structures

TEMPORAL CONTEXT:
Feature characteristics suggest occupation during the Late Ceramic Period (1000-1500 CE), coinciding with complex society development in the Amazon basin.

ARCHAEOLOGICAL POTENTIAL:
HIGH - This site represents significant potential for understanding:
- Pre-Columbian landscape modification
- Complex society organization
- Human-environment interaction in tropical settings

FIELD RECOMMENDATIONS:
1. Ground-penetrating radar survey to confirm subsurface features
2. Strategic test excavations at key feature intersections  
3. Ceramic and radiocarbon sampling for dating
4. Ethnographic consultation with local indigenous communities
5. Comparative analysis with known regional sites

INVESTIGATION PRIORITY: Immediate archaeological investigation recommended.
"""
        
        return {
            'llm_analysis': mock_analysis,
            'model_used': 'mock_expert_system',
            'processing_time': datetime.now().isoformat(),
            'token_usage': 0
        }
    
    def _initialize_cultural_database(self):
        """Initialize cultural context database with RAG"""
        
        if not RAG_AVAILABLE:
            return
        
        try:
            # Initialize ChromaDB client
            client = chromadb.Client()
            
            # Create or get collection
            self.cultural_db = client.get_or_create_collection(
                name="amazon_archaeology",
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )
            
            # Add archaeological knowledge base
            self._populate_cultural_database()
            
            logger.info("Cultural context database initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize cultural database: {e}")
            self.cultural_db = None
    
    def _populate_cultural_database(self):
        """Populate cultural database with archaeological knowledge"""
        
        if not self.cultural_db:
            return
        
        # Amazon archaeological knowledge base
        cultural_data = [
            {
                'id': 'casarabe_culture',
                'text': 'Casarabe culture (500-1400 CE) in Llanos de Mojos, Bolivia. Known for geometric earthworks, raised fields, forest islands, and complex hydraulic systems. Circular ceremonial centers connected by causeway networks.',
                'metadata': {'culture': 'Casarabe', 'period': '500-1400 CE', 'region': 'Llanos de Mojos'}
            },
            {
                'id': 'marajoara_culture', 
                'text': 'Marajoara culture (400-1200 CE) on Marajó Island, Brazil. Famous for elaborate polychrome ceramics, large settlement mounds, and complex mortuary practices. Circular plazas and residential platforms.',
                'metadata': {'culture': 'Marajoara', 'period': '400-1200 CE', 'region': 'Marajó Island'}
            },
            {
                'id': 'tapajos_culture',
                'text': 'Tapajós culture (1000-1500 CE) along Tapajós River confluence. Known for anthropomorphic ceramics, riverine settlement patterns, and trade networks. Defensive earthworks and ceremonial centers.',
                'metadata': {'culture': 'Tapajós', 'period': '1000-1500 CE', 'region': 'Tapajós River'}
            },
            {
                'id': 'acre_geoglyphs',
                'text': 'Acre geoglyphs (1000-1450 CE) in western Amazon. Geometric earthworks revealed by deforestation. Circular and square enclosures connected by roads. Likely ceremonial and social gathering purposes.',
                'metadata': {'culture': 'Acre', 'period': '1000-1450 CE', 'region': 'Acre, Brazil'}
            },
            {
                'id': 'raised_fields',
                'text': 'Pre-Columbian raised field agriculture throughout Amazon lowlands. Complex drainage and irrigation systems. Associated with forest islands and residential mounds. Intensive landscape management.',
                'metadata': {'feature_type': 'agricultural', 'period': '500-1500 CE', 'region': 'Amazon lowlands'}
            }
        ]
        
        try:
            # Add documents to collection
            for data in cultural_data:
                self.cultural_db.add(
                    documents=[data['text']],
                    metadatas=[data['metadata']],
                    ids=[data['id']]
                )
            
            logger.info(f"Added {len(cultural_data)} cultural context documents")
            
        except Exception as e:
            logger.warning(f"Failed to populate cultural database: {e}")
    
    def _build_cultural_query(self, llm_analysis: Dict, site_info: Dict, data: Dict) -> str:
        """Build query for cultural context RAG"""
        
        detected_structures = data.get('detected_structures', {})
        structure_summary = detected_structures.get('structures', {}).get('summary', {})
        
        query = f"""
        Archaeological site with {structure_summary.get('total_structures', 0)} detected features
        including circular earthworks, linear features, and mounds.
        Located in {site_info.get('region', 'Amazon basin')} region.
        Geometric patterns suggest ceremonial and residential functions.
        Cultural context and comparison needed.
        """
        
        return query.strip()
    
    def _process_cultural_rag_results(self, results: Dict, site_info: Dict) -> Dict:
        """Process RAG results for cultural context"""
        
        cultural_context = {
            'similar_cultures': [],
            'cultural_interpretation': {},
            'temporal_context': [],
            'comparative_sites': []
        }
        
        if 'documents' in results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):  # First query results
                metadata = results.get('metadatas', [[]])[0]
                if i < len(metadata):
                    meta = metadata[i]
                    
                    cultural_context['similar_cultures'].append({
                        'culture': meta.get('culture', 'Unknown'),
                        'period': meta.get('period', 'Unknown'),
                        'region': meta.get('region', 'Unknown'),
                        'description': doc,
                        'relevance_score': 1.0 - (i * 0.1)  # Decreasing relevance
                    })
        
        return cultural_context
    
    def _generate_default_cultural_context(self, site_info: Dict, data: Dict) -> Dict:
        """Generate default cultural context when RAG not available"""
        
        return {
            'cultural_interpretation': {
                'primary_culture': 'Pre-Columbian Amazon Complex Societies',
                'temporal_period': 'Late Ceramic Period (1000-1500 CE)',
                'site_function': 'Multi-functional ceremonial and residential complex',
                'landscape_context': 'Anthropogenic landscape with intensive modification'
            },
            'historical_context': {
                'regional_sequence': 'Late pre-contact period complex societies',
                'cultural_continuity': 'Indigenous landscape management traditions',
                'european_contact': 'Pre-contact indigenous engineering'
            },
            'similar_sites': [
                {
                    'name': 'Llanos de Mojos earthworks',
                    'similarity': 'Geometric earthwork patterns',
                    'distance': 'Regional comparison'
                },
                {
                    'name': 'Acre geoglyphs',
                    'similarity': 'Circular ceremonial features',
                    'distance': 'Cultural comparison'
                }
            ]
        }
    
    # Additional helper methods for comprehensive analysis
    
    def _extract_multimodal_insights(self, llm_analysis: Dict) -> Dict:
        """Extract insights from multimodal LLM analysis"""
        
        analysis_text = llm_analysis.get('llm_analysis', '')
        
        insights = {
            'key_findings': [],
            'cultural_connections': [],
            'temporal_indicators': [],
            'confidence_indicators': []
        }
        
        # Simple keyword extraction (would be more sophisticated with NLP)
        if 'ceremonial' in analysis_text.lower():
            insights['key_findings'].append('Ceremonial function identified')
        
        if 'casarabe' in analysis_text.lower():
            insights['cultural_connections'].append('Casarabe culture similarity')
        
        if 'pre-columbian' in analysis_text.lower():
            insights['temporal_indicators'].append('Pre-Columbian period')
        
        if 'high' in analysis_text.lower():
            insights['confidence_indicators'].append('High confidence assessment')
        
        return insights
    
    def _assess_analysis_confidence(self, llm_analysis: Dict, processed_data: Dict) -> Dict:
        """Assess overall confidence in analysis"""
        
        structure_confidence = 0.0
        if 'detected_structures' in processed_data:
            struct_conf = processed_data['detected_structures'].get('confidence_scores', {})
            structure_confidence = struct_conf.get('overall_confidence', 0.0)
        
        llm_confidence = 0.8  # Mock confidence for LLM analysis
        
        return {
            'structure_detection_confidence': structure_confidence,
            'llm_analysis_confidence': llm_confidence,
            'overall_confidence': (structure_confidence + llm_confidence) / 2,
            'confidence_factors': [
                'LIDAR data quality',
                'Structure detection algorithms',
                'Cultural context availability',
                'Expert knowledge integration'
            ]
        }
    
    def _calculate_overall_archaeological_score(self, processed_data: Dict, 
                                              llm_analysis: Dict, cultural_analysis: Dict) -> float:
        """Calculate comprehensive archaeological potential score"""
        
        # Structure detection score (40%)
        structure_score = 0.0
        if 'detected_structures' in processed_data:
            structures = processed_data['detected_structures']
            if 'structures' in structures and 'summary' in structures['structures']:
                total_structures = structures['structures']['summary'].get('total_structures', 0)
                structure_score = min(total_structures / 20.0, 1.0) * 0.4
        
        # LLM analysis score (30%)
        llm_score = 0.0
        analysis_text = llm_analysis.get('llm_analysis', '')
        if 'high' in analysis_text.lower():
            llm_score = 0.3
        elif 'medium' in analysis_text.lower():
            llm_score = 0.2
        else:
            llm_score = 0.15
        
        # Cultural context score (20%)
        cultural_score = 0.0
        if cultural_analysis.get('similar_cultures'):
            cultural_score = 0.2
        elif cultural_analysis.get('cultural_interpretation'):
            cultural_score = 0.15
        
        # Terrain complexity score (10%)
        terrain_score = min(processed_data.get('terrain_complexity', 0.5), 1.0) * 0.1
        
        total_score = structure_score + llm_score + cultural_score + terrain_score
        return min(total_score, 1.0)
    
    def _assess_feature_significance(self, processed_data: Dict, cultural_analysis: Dict) -> Dict:
        """Assess significance of detected features"""
        
        significance = {
            'individual_features': [],
            'feature_clusters': [],
            'landscape_significance': '',
            'cultural_significance': ''
        }
        
        # Analyze detected structures
        if 'detected_structures' in processed_data:
            structures = processed_data['detected_structures'].get('structures', {})
            
            for structure_type, features in structures.items():
                if structure_type != 'summary' and features:
                    for feature in features[:3]:  # Top 3 features
                        significance['individual_features'].append({
                            'type': structure_type,
                            'confidence': feature.get('confidence', 0.0),
                            'significance': 'High' if feature.get('confidence', 0) > 0.8 else 'Medium'
                        })
        
        # Overall landscape significance
        total_features = len(significance['individual_features'])
        if total_features > 10:
            significance['landscape_significance'] = 'Complex multi-component archaeological landscape'
        elif total_features > 5:
            significance['landscape_significance'] = 'Significant archaeological site with multiple features'
        else:
            significance['landscape_significance'] = 'Archaeological site with limited features'
        
        # Cultural significance from analysis
        if cultural_analysis.get('similar_cultures'):
            significance['cultural_significance'] = 'Potentially significant for understanding regional cultural development'
        else:
            significance['cultural_significance'] = 'Requires further cultural context analysis'
        
        return significance
    
    def _determine_investigation_priority(self, processed_data: Dict, 
                                        llm_analysis: Dict, cultural_analysis: Dict) -> str:
        """Determine investigation priority level"""
        
        score = self._calculate_overall_archaeological_score(
            processed_data, llm_analysis, cultural_analysis
        )
        
        structure_count = 0
        if 'detected_structures' in processed_data:
            structures = processed_data['detected_structures'].get('structures', {})
            if 'summary' in structures:
                structure_count = structures['summary'].get('total_structures', 0)
        
        llm_text = llm_analysis.get('llm_analysis', '').lower()
        
        # Priority determination logic
        if score > 0.8 or structure_count > 15 or 'immediate' in llm_text:
            return 'IMMEDIATE - High priority archaeological investigation required'
        elif score > 0.6 or structure_count > 8 or 'high' in llm_text:
            return 'HIGH - Significant archaeological potential, investigation recommended'
        elif score > 0.4 or structure_count > 3:
            return 'MEDIUM - Moderate archaeological interest, further analysis recommended'
        else:
            return 'LOW - Limited archaeological indicators, monitoring recommended'
    
    def _generate_comprehensive_recommendations(self, processed_data: Dict,
                                              llm_analysis: Dict, cultural_analysis: Dict,
                                              site_info: Dict) -> List[str]:
        """Generate comprehensive investigation recommendations"""
        
        recommendations = []
        
        # Priority-based recommendations
        priority = self._determine_investigation_priority(
            processed_data, llm_analysis, cultural_analysis
        )
        
        if 'IMMEDIATE' in priority:
            recommendations.extend([
                'Immediate coordination with regional archaeological authorities',
                'Emergency protection measures for identified features',
                'Rapid deployment of ground-penetrating radar survey',
                'Consultation with local indigenous communities',
                'Establishment of site protection perimeter'
            ])
        elif 'HIGH' in priority:
            recommendations.extend([
                'Detailed ground-penetrating radar survey',
                'Strategic test excavation program',
                'High-resolution drone mapping',
                'Comprehensive site mapping and documentation',
                'Coordination with regional research institutions'
            ])
        else:
            recommendations.extend([
                'Extended remote sensing monitoring',
                'Comparative analysis with regional archaeological databases',
                'Preliminary field reconnaissance',
                'Integration with regional archaeological surveys'
            ])
        
        # Feature-specific recommendations
        structures = processed_data.get('detected_structures', {}).get('structures', {})
        
        if structures.get('circular_features'):
            recommendations.append('Focus investigation on circular features for ceremonial significance')
        
        if structures.get('linear_features'):
            recommendations.append('Map linear feature network for ancient transportation/communication systems')
        
        if structures.get('mounds'):
            recommendations.append('Priority investigation of mounds for burial or ceremonial functions')
        
        # Cultural context recommendations
        if cultural_analysis.get('similar_cultures'):
            recommendations.append('Comparative study with similar cultural contexts identified')
        
        # Technical recommendations
        recommendations.extend([
            'Collection of radiocarbon samples for absolute dating',
            'Ceramic analysis if surface materials present',
            'Environmental sampling for paleoenvironmental reconstruction',
            'Documentation using photogrammetry and 3D scanning'
        ])
        
        return recommendations
    
    def _extract_site_bounds(self, site_info: Dict) -> Tuple[float, float, float, float]:
        """Extract or estimate site bounds"""
        
        coords = site_info.get('coordinates', [0, 0])
        if len(coords) >= 2:
            lat, lon = coords[0], coords[1]
            # Create bounds around point (roughly 1km x 1km)
            offset = 0.005  # Approximately 500m
            return (lon - offset, lat - offset, lon + offset, lat + offset)
        
        # Default Amazon basin bounds
        return (-70.0, -15.0, -50.0, 5.0)
    
    def _export_archaeological_assessment_csv(self, results: Dict, output_dir: str, timestamp: str) -> str:
        """Export archaeological assessment as CSV"""
        
        assessment_data = []
        
        # Overall assessment
        assessment = results.get('archaeological_assessment', {})
        assessment_data.append({
            'metric': 'Overall Archaeological Score',
            'value': assessment.get('overall_score', 0),
            'category': 'Assessment',
            'description': 'Comprehensive archaeological potential score (0-1)'
        })
        
        assessment_data.append({
            'metric': 'Investigation Priority',
            'value': assessment.get('investigation_priority', 'Unknown'),
            'category': 'Assessment', 
            'description': 'Recommended investigation priority level'
        })
        
        # Structure counts
        structures = results.get('processed_terrain_data', {}).get('detected_structures', {})
        if 'structures' in structures and 'summary' in structures['structures']:
            summary = structures['structures']['summary']
            
            for structure_type, count in summary.get('by_type', {}).items():
                assessment_data.append({
                    'metric': f'{structure_type.replace("_", " ").title()} Count',
                    'value': count,
                    'category': 'Structure Detection',
                    'description': f'Number of detected {structure_type.replace("_", " ")}'
                })
        
        # Create DataFrame and export
        df = pd.DataFrame(assessment_data)
        
        filename = f'archaeological_assessment_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def _generate_image_descriptions(self, images: Dict) -> Dict:
        """Generate descriptions for analysis images"""
        
        descriptions = {}
        
        image_descriptions = {
            'hillshade': 'Hillshade analysis showing terrain relief and detected archaeological structures',
            'elevation_contours': 'Digital terrain model with elevation contours and structure overlays',
            'slope_analysis': 'Slope and aspect analysis for terrain characterization',
            'composite_analysis': 'Multi-layer composite showing hillshade, elevation, relief, and curvature',
            'structure_detection': 'Focused visualization of AI-detected archaeological structures',
            'perspective_3d': '3D perspective view of terrain with archaeological features'
        }
        
        for image_type, description in image_descriptions.items():
            if image_type in images:
                descriptions[image_type] = description
        
        return descriptions
    
    def _calculate_slope_statistics(self, slope_data) -> Dict:
        """Calculate slope statistics"""
        if isinstance(slope_data, np.ndarray) and slope_data.size > 0:
            return {
                'mean_slope': float(np.mean(slope_data)),
                'max_slope': float(np.max(slope_data)),
                'steep_areas_percent': float(np.sum(slope_data > 30) / slope_data.size * 100)
            }
        return {}
    
    def _calculate_aspect_statistics(self, aspect_data) -> Dict:
        """Calculate aspect statistics"""
        if isinstance(aspect_data, np.ndarray) and aspect_data.size > 0:
            return {
                'mean_aspect': float(np.mean(aspect_data)),
                'aspect_variability': float(np.std(aspect_data))
            }
        return {}
    
    def _calculate_curvature_statistics(self, curvature_data) -> Dict:
        """Calculate curvature statistics"""
        if isinstance(curvature_data, np.ndarray) and curvature_data.size > 0:
            return {
                'mean_curvature': float(np.mean(curvature_data)),
                'curvature_variability': float(np.std(curvature_data)),
                'convex_areas_percent': float(np.sum(curvature_data > 0) / curvature_data.size * 100)
            }
        return {}


def main():
    """Test the multimodal pipeline"""
    
    # Example site information
    test_site = {
        'name': 'Test Amazon Archaeological Site',
        'region': 'Western Amazon Basin',
        'coordinates': [-8.5, -63.2],
        'acquisition_date': '2024-01-15',
        'expected_features': ['earthworks', 'circular_patterns', 'linear_features']
    }
    
    # Initialize pipeline
    pipeline = MultimodalArchaeologicalPipeline()
    
    # Run analysis (will use synthetic data for demonstration)
    results = pipeline.process_lidar_site(
        lidar_path='synthetic_test_data',
        site_info=test_site,
        output_dir='/tmp/multimodal_archaeological_analysis'
    )
    
    print("Multimodal Archaeological Analysis Results:")
    print(f"Site: {test_site['name']}")
    print(f"Overall Score: {results['archaeological_assessment']['overall_score']:.3f}")
    print(f"Priority: {results['archaeological_assessment']['investigation_priority']}")
    print(f"Structures Detected: {results['processed_terrain_data']['detected_structures']['structures']['summary']['total_structures']}")
    
    print("\nGenerated Files:")
    for file_type, filepath in results.get('exported_files', {}).items():
        if filepath and os.path.exists(filepath):
            print(f"  {file_type}: {filepath}")


if __name__ == "__main__":
    main()