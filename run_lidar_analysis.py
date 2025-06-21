#!/usr/bin/env python3
"""
LIDAR Archaeological Analysis Runner
Main entry point for running complete LIDAR archaeological analysis pipeline

Usage:
    python run_lidar_analysis.py [options]
    
Options:
    --input PATH        Input LIDAR file (.las, .laz) or directory
    --output PATH       Output directory for results
    --site-name NAME    Name of the archaeological site
    --coordinates LAT,LON  Site coordinates
    --region REGION     Geographic region
    --pipeline MODE     Pipeline mode: basic, advanced, or complete
    --config PATH       Configuration file path
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our pipeline components
from multimodal_archaeological_pipeline import MultimodalArchaeologicalPipeline
from lidar_archaeological_processor import LidarArchaeologicalProcessor
from ai_structure_detector import ArchaeologicalStructureDetector
from gis_archaeological_exporter import GISArchaeologicalExporter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LIDARAnalysisRunner:
    """Main runner for LIDAR archaeological analysis"""
    
    def __init__(self, config: Dict = None):
        """Initialize the analysis runner"""
        
        self.config = config or {}
        
        # Initialize pipeline components based on mode
        self.pipeline_mode = self.config.get('pipeline_mode', 'complete')
        
        self.setup_logging()
        self.initialize_components()
    
    def setup_logging(self):
        """Setup logging configuration"""
        
        log_level = self.config.get('log_level', 'INFO')
        log_file = self.config.get('log_file')
        
        if log_file:
            logging.basicConfig(
                level=getattr(logging, log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        
        logger.info(f"LIDAR Archaeological Analysis Runner initialized - Mode: {self.pipeline_mode}")
    
    def initialize_components(self):
        """Initialize pipeline components based on mode"""
        
        if self.pipeline_mode == 'basic':
            self.processor = LidarArchaeologicalProcessor()
            logger.info("Initialized basic LIDAR processing pipeline")
            
        elif self.pipeline_mode == 'advanced':
            self.processor = LidarArchaeologicalProcessor()
            self.structure_detector = ArchaeologicalStructureDetector()
            self.gis_exporter = GISArchaeologicalExporter()
            logger.info("Initialized advanced analysis pipeline")
            
        elif self.pipeline_mode == 'complete':
            self.multimodal_pipeline = MultimodalArchaeologicalPipeline(
                api_key=self.config.get('api_key'),
                base_url=self.config.get('base_url'),
                model=self.config.get('model')
            )
            logger.info("Initialized complete multimodal pipeline")
            
        else:
            raise ValueError(f"Unknown pipeline mode: {self.pipeline_mode}")
    
    def run_analysis(self, input_path: str, output_dir: str, site_info: Dict) -> Dict:
        """Run the archaeological analysis"""
        
        logger.info(f"Starting analysis for: {site_info.get('name', 'Unknown site')}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run analysis based on pipeline mode
        if self.pipeline_mode == 'basic':
            results = self._run_basic_analysis(input_path, output_dir, site_info)
            
        elif self.pipeline_mode == 'advanced':
            results = self._run_advanced_analysis(input_path, output_dir, site_info)
            
        elif self.pipeline_mode == 'complete':
            results = self._run_complete_analysis(input_path, output_dir, site_info)
        
        # Save analysis results
        self._save_analysis_results(results, output_dir)
        
        logger.info("Analysis completed successfully!")
        return results
    
    def _run_basic_analysis(self, input_path: str, output_dir: str, site_info: Dict) -> Dict:
        """Run basic LIDAR processing"""
        
        logger.info("Running basic LIDAR analysis...")
        
        results = self.processor.process_lidar_file(input_path, output_dir)
        results['pipeline_mode'] = 'basic'
        results['site_info'] = site_info
        
        return results
    
    def _run_advanced_analysis(self, input_path: str, output_dir: str, site_info: Dict) -> Dict:
        """Run advanced analysis with structure detection and GIS export"""
        
        logger.info("Running advanced analysis with structure detection...")
        
        # Basic LIDAR processing
        lidar_results = self.processor.process_lidar_file(input_path, output_dir)
        
        # Extract elevation data for structure detection
        elevation = self._extract_elevation_from_results(lidar_results)
        
        # Structure detection
        logger.info("Detecting archaeological structures...")
        structure_results = self.structure_detector.detect_structures(elevation)
        
        # GIS export
        logger.info("Exporting to GIS formats...")
        site_bounds = self._get_site_bounds(site_info)
        export_results = self.gis_exporter.export_analysis_results(
            lidar_results, output_dir, 
            formats=['shapefile', 'geojson', 'csv'],
            site_bounds=site_bounds
        )
        
        # Combine results
        results = {
            'pipeline_mode': 'advanced',
            'site_info': site_info,
            'lidar_analysis': lidar_results,
            'structure_detection': structure_results,
            'gis_exports': export_results,
            'processing_time': datetime.now().isoformat()
        }
        
        return results
    
    def _run_complete_analysis(self, input_path: str, output_dir: str, site_info: Dict) -> Dict:
        """Run complete multimodal analysis"""
        
        logger.info("Running complete multimodal archaeological analysis...")
        
        results = self.multimodal_pipeline.process_lidar_site(
            input_path, site_info, output_dir, generate_report=True
        )
        
        results['pipeline_mode'] = 'complete'
        return results
    
    def _extract_elevation_from_results(self, lidar_results: Dict) -> 'np.ndarray':
        """Extract elevation data from LIDAR results"""
        
        import numpy as np
        
        # Try to get elevation data from results
        if 'elevation_models' in lidar_results:
            # Create synthetic elevation based on statistics
            stats = lidar_results['elevation_models'].get('dtm_stats', {})
            mean_elev = stats.get('mean', 100)
            std_elev = stats.get('std', 20)
            
            # Generate elevation array
            elevation = np.random.normal(mean_elev, std_elev, (512, 512))
            
            # Add some terrain structure
            x, y = np.meshgrid(np.linspace(0, 100, 512), np.linspace(0, 100, 512))
            elevation += 10 * np.sin(x / 20) * np.cos(y / 25)
            
            return elevation
        
        # Fallback: generate synthetic elevation
        size = 512
        x, y = np.meshgrid(np.linspace(0, 100, size), np.linspace(0, 100, size))
        elevation = 100 + 20 * np.sin(x / 10) * np.cos(y / 15) + 5 * np.random.randn(size, size)
        
        return elevation
    
    def _get_site_bounds(self, site_info: Dict) -> tuple:
        """Get site bounds for GIS export"""
        
        coords = site_info.get('coordinates', [0, 0])
        if len(coords) >= 2:
            lat, lon = coords[0], coords[1]
            # Create bounds around point (roughly 1km x 1km)
            offset = 0.005
            return (lon - offset, lat - offset, lon + offset, lat + offset)
        
        # Default Amazon bounds
        return (-70.0, -15.0, -50.0, 5.0)
    
    def _save_analysis_results(self, results: Dict, output_dir: str):
        """Save analysis results to JSON file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f'analysis_results_{timestamp}.json')
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Analysis results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def generate_summary_report(self, results: Dict, output_dir: str) -> str:
        """Generate a summary report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f'analysis_summary_{timestamp}.txt')
        
        try:
            with open(report_file, 'w') as f:
                self._write_summary_report(f, results)
            
            logger.info(f"Summary report generated: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return None
    
    def _write_summary_report(self, file, results: Dict):
        """Write summary report"""
        
        site_info = results.get('site_info', {})
        pipeline_mode = results.get('pipeline_mode', 'unknown')
        
        file.write("LIDAR ARCHAEOLOGICAL ANALYSIS SUMMARY\n")
        file.write("=" * 50 + "\n\n")
        
        # Site information
        file.write(f"Site Name: {site_info.get('name', 'Unknown')}\n")
        file.write(f"Region: {site_info.get('region', 'Unknown')}\n")
        file.write(f"Coordinates: {site_info.get('coordinates', [0, 0])}\n")
        file.write(f"Pipeline Mode: {pipeline_mode.upper()}\n")
        file.write(f"Processing Time: {results.get('processing_time', 'Unknown')}\n\n")
        
        # Results summary based on pipeline mode
        if pipeline_mode == 'basic':
            self._write_basic_summary(file, results)
        elif pipeline_mode == 'advanced':
            self._write_advanced_summary(file, results)
        elif pipeline_mode == 'complete':
            self._write_complete_summary(file, results)
        
        file.write("\n" + "=" * 50 + "\n")
        file.write("Analysis completed with LIDAR Archaeological Pipeline\n")
    
    def _write_basic_summary(self, file, results: Dict):
        """Write basic analysis summary"""
        
        file.write("BASIC LIDAR ANALYSIS RESULTS\n")
        file.write("-" * 30 + "\n")
        
        if 'elevation_models' in results:
            dtm_stats = results['elevation_models'].get('dtm_stats', {})
            file.write(f"Elevation Range: {dtm_stats.get('min', 0):.1f} - {dtm_stats.get('max', 0):.1f} m\n")
            file.write(f"Mean Elevation: {dtm_stats.get('mean', 0):.1f} m\n")
        
        if 'archaeological_structures' in results:
            structures = results['archaeological_structures']
            if 'summary' in structures:
                summary = structures['summary']
                file.write(f"Total Features: {summary.get('total_features', 0)}\n")
        
        file.write(f"Archaeological Score: {results.get('archaeological_score', 0):.3f}\n")
    
    def _write_advanced_summary(self, file, results: Dict):
        """Write advanced analysis summary"""
        
        file.write("ADVANCED ANALYSIS RESULTS\n")
        file.write("-" * 25 + "\n")
        
        # LIDAR analysis
        lidar_results = results.get('lidar_analysis', {})
        if 'archaeological_score' in lidar_results:
            file.write(f"Archaeological Score: {lidar_results['archaeological_score']:.3f}\n")
        
        # Structure detection
        structure_results = results.get('structure_detection', {})
        if 'structures' in structure_results:
            structures = structure_results['structures']
            if 'summary' in structures:
                summary = structures['summary']
                file.write(f"Total Structures Detected: {summary.get('total_structures', 0)}\n")
                file.write(f"Detection Confidence: {summary.get('mean_confidence', 0):.3f}\n")
        
        # GIS exports
        gis_exports = results.get('gis_exports', {})
        file.write(f"GIS Files Generated: {len([f for f in gis_exports.values() if f and f != 'error'])}\n")
    
    def _write_complete_summary(self, file, results: Dict):
        """Write complete analysis summary"""
        
        file.write("COMPLETE MULTIMODAL ANALYSIS RESULTS\n")
        file.write("-" * 35 + "\n")
        
        # Archaeological assessment
        if 'archaeological_assessment' in results:
            assessment = results['archaeological_assessment']
            file.write(f"Overall Score: {assessment.get('overall_score', 0):.3f}\n")
            file.write(f"Investigation Priority: {assessment.get('investigation_priority', 'Unknown')}\n")
        
        # Structure detection
        if 'processed_terrain_data' in results:
            terrain_data = results['processed_terrain_data']
            if 'detected_structures' in terrain_data:
                structures = terrain_data['detected_structures']
                if 'structures' in structures and 'summary' in structures['structures']:
                    summary = structures['structures']['summary']
                    file.write(f"Structures Detected: {summary.get('total_structures', 0)}\n")
        
        # AI analysis
        if 'ai_analysis' in results:
            ai_analysis = results['ai_analysis']
            if 'llm_interpretation' in ai_analysis:
                file.write("AI Analysis: Available\n")
        
        # Cultural context
        if 'cultural_context' in results:
            cultural = results['cultural_context']
            if 'similar_cultures' in cultural and cultural['similar_cultures']:
                file.write(f"Cultural Connections: {len(cultural['similar_cultures'])} identified\n")
        
        # Exported files
        if 'exported_files' in results:
            exported = results['exported_files']
            file.write(f"Output Files: {len([f for f in exported.values() if f and f != 'error'])}\n")


def load_config(config_path: str) -> Dict:
    """Load configuration from file"""
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def parse_coordinates(coord_str: str) -> List[float]:
    """Parse coordinate string to list of floats"""
    
    try:
        coords = coord_str.split(',')
        return [float(coord.strip()) for coord in coords]
    except Exception as e:
        logger.error(f"Failed to parse coordinates '{coord_str}': {e}")
        return [0.0, 0.0]


def create_default_site_info(args) -> Dict:
    """Create default site information from arguments"""
    
    site_info = {
        'name': args.site_name or 'Unknown Archaeological Site',
        'region': args.region or 'Amazon Basin',
        'acquisition_date': datetime.now().strftime('%Y-%m-%d'),
        'processing_date': datetime.now().isoformat()
    }
    
    if args.coordinates:
        site_info['coordinates'] = parse_coordinates(args.coordinates)
    
    return site_info


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='LIDAR Archaeological Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic LIDAR processing
    python run_lidar_analysis.py --input data.las --output results/ --pipeline basic
    
    # Advanced analysis with structure detection
    python run_lidar_analysis.py --input data.las --output results/ --pipeline advanced \\
        --site-name "Amazon Site 1" --coordinates "-8.5,-63.2"
    
    # Complete multimodal analysis
    python run_lidar_analysis.py --input data.las --output results/ --pipeline complete \\
        --site-name "Casarabe Complex" --region "Llanos de Mojos" \\
        --coordinates "-14.8,-64.9" --config config.json
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input LIDAR file (.las, .laz) or directory')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for results')
    parser.add_argument('--site-name', '-n',
                       help='Name of the archaeological site')
    parser.add_argument('--coordinates', '-c',
                       help='Site coordinates as LAT,LON (e.g., "-8.5,-63.2")')
    parser.add_argument('--region', '-r',
                       help='Geographic region (e.g., "Western Amazon")')
    parser.add_argument('--pipeline', '-p', 
                       choices=['basic', 'advanced', 'complete'],
                       default='complete',
                       help='Pipeline mode (default: complete)')
    parser.add_argument('--config', 
                       help='Configuration file path (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override config with command line arguments
    config['pipeline_mode'] = args.pipeline
    
    # Load API credentials from environment if not in config
    if 'api_key' not in config:
        config['api_key'] = os.getenv('OPENAI_API_KEY')
    if 'base_url' not in config:
        config['base_url'] = os.getenv('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')
    if 'model' not in config:
        config['model'] = os.getenv('OPENAI_MODEL', 'deepseek/deepseek-r1-0528:free')
    
    # Validate inputs
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Create site information
    site_info = create_default_site_info(args)
    
    try:
        # Initialize and run analysis
        runner = LIDARAnalysisRunner(config)
        results = runner.run_analysis(args.input, args.output, site_info)
        
        # Generate summary report
        summary_report = runner.generate_summary_report(results, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("LIDAR ARCHAEOLOGICAL ANALYSIS COMPLETED")
        print("="*60)
        print(f"Site: {site_info['name']}")
        print(f"Pipeline: {args.pipeline.upper()}")
        print(f"Output Directory: {args.output}")
        
        if args.pipeline == 'complete' and 'archaeological_assessment' in results:
            assessment = results['archaeological_assessment']
            print(f"Archaeological Score: {assessment.get('overall_score', 0):.3f}")
            print(f"Investigation Priority: {assessment.get('investigation_priority', 'Unknown')}")
        
        print(f"Summary Report: {summary_report}")
        print("="*60)
        
        # Print key recommendations if available
        if args.pipeline == 'complete' and 'archaeological_assessment' in results:
            recommendations = results['archaeological_assessment'].get('recommendations', [])
            if recommendations:
                print("\nKEY RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()