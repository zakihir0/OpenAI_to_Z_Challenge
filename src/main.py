#!/usr/bin/env python3
"""
Main Archaeological Analysis System
Orchestrates the complete archaeological site analysis workflow
"""

import json
from datetime import datetime
from typing import Dict, List
import logging

from archaeological_analyzer import ArchaeologicalAnalyzer
from visualization_engine import VisualizationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArchaeologicalAnalysisSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.analyzer = ArchaeologicalAnalyzer()
        self.visualizer = VisualizationEngine()
    
    def analyze_sites(self, target_sites: List[Dict]) -> List[Dict]:
        """Analyze multiple archaeological sites"""
        
        logger.info(f"Starting analysis of {len(target_sites)} sites")
        results = []
        
        for site in target_sites:
            logger.info(f"Analyzing: {site['name']}")
            
            try:
                analysis = self.analyzer.analyze_site(site)
                results.append(analysis)
                
                score = analysis['archaeological_score']
                confidence = analysis['confidence_level']
                features = analysis['computer_vision_analysis']['geometric_features']['total_count']
                
                logger.info(f"  ✓ Score: {score:.2f} | Confidence: {confidence} | Features: {features}")
                
            except Exception as e:
                logger.error(f"  ✗ Error analyzing {site['name']}: {e}")
        
        return results
    
    def generate_reports(self, results: List[Dict]) -> Dict[str, str]:
        """Generate all reports and visualizations"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        results_file = f'results/analysis_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate text report
        text_report = self.visualizer.generate_text_report(results, timestamp)
        
        # Generate visualizations for each site
        visualization_files = []
        for result in results:
            viz_file = self.visualizer.create_complete_site_analysis(result, timestamp)
            visualization_files.append(viz_file)
        
        logger.info(f"Analysis complete. Generated {len(visualization_files)} visualizations")
        
        return {
            'results_file': results_file,
            'text_report': text_report,
            'visualizations': visualization_files
        }

def main():
    """Main execution function"""
    
    # Define target archaeological sites
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
    
    # Initialize system
    system = ArchaeologicalAnalysisSystem()
    
    # Analyze sites
    results = system.analyze_sites(target_sites)
    
    # Generate reports
    output_files = system.generate_reports(results)
    
    # Summary
    print(f"\n{'='*60}")
    print("ARCHAEOLOGICAL ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Sites analyzed: {len(results)}")
    print(f"Results file: {output_files['results_file']}")
    print(f"Text report: {output_files['text_report']}")
    print(f"Visualizations: {len(output_files['visualizations'])} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()