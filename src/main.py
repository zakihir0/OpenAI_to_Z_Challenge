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
from kaggle_integration import integrate_kaggle_analysis

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
    
    # Initialize system first to access Amazon coordinates
    system = ArchaeologicalAnalysisSystem()
    
    # Get Amazon archaeological sites from the analyzer
    target_sites = system.analyzer._search_amazon_coordinates()
    
    # Filter to highest priority sites for initial analysis
    priority_sites = [site for site in target_sites if site['priority'] == 'highest']
    
    logger.info(f"Amazon archaeological survey initialized: {len(target_sites)} total sites, {len(priority_sites)} highest priority")
    
    # Analyze highest priority sites first
    results = system.analyze_sites(priority_sites)
    
    # Generate reports
    output_files = system.generate_reports(results)
    
    # Integrate Kaggle analysis
    kaggle_credentials = {
        "username": "zakihiro",
        "key": "9eee705e4648cebe96a4ed3de9b920d5"
    }
    
    logger.info("Integrating Kaggle analysis capabilities...")
    kaggle_results = integrate_kaggle_analysis(results, kaggle_credentials)
    
    # Summary
    print(f"\n{'='*60}")
    print("ARCHAEOLOGICAL ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Sites analyzed: {len(results)}")
    print(f"Results file: {output_files['results_file']}")
    print(f"Text report: {output_files['text_report']}")
    print(f"Visualizations: {len(output_files['visualizations'])} files")
    print(f"\n{'='*60}")
    print("KAGGLE INTEGRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Datasets found: {kaggle_results['datasets_found']}")
    print(f"Competitions found: {kaggle_results['competitions_found']}")
    print(f"Dataset created: {kaggle_results['dataset_created']}")
    print(f"Submission file: {kaggle_results['submission_file']}")
    print(f"Kaggle ready: {kaggle_results['kaggle_ready']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()