#!/usr/bin/env python3
"""
Comprehensive Example Usage for OpenAI to Z Challenge
Archaeological Site Detection using AI and Remote Sensing

This script demonstrates the integration of:
1. OpenAI API for intelligent analysis
2. RAG system with archaeological knowledge base
3. Satellite data processing for feature detection
4. Complete pipeline from data to site recommendations

Usage:
    python comprehensive_example.py

Requirements:
    - OpenAI API key set in environment variable OPENAI_API_KEY
    - Required packages installed: pip install -r requirements.txt
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Import our custom modules
from openai_archaeological_analysis import ArchaeologicalSiteDetector
from rag_knowledge_base import RAGArchaeologist
from satellite_data_processing import SatelliteImageProcessor

class ComprehensiveArchaeologicalAnalysis:
    """Complete archaeological analysis pipeline integrating all components"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        
        # Initialize all components
        self.site_detector = ArchaeologicalSiteDetector(openai_api_key)
        self.rag_system = RAGArchaeologist(openai_api_key, vector_store_type="memory")
        self.satellite_processor = SatelliteImageProcessor()
        
        # Amazon basin search parameters
        self.amazon_regions = [
            {
                'name': 'Western Amazon - Peru/Ecuador Border',
                'bounds': {'min_lat': -5.0, 'max_lat': -3.0, 'min_lon': -78.0, 'max_lon': -76.0},
                'priority': 'high',
                'characteristics': ['Dense forest', 'River systems', 'Known archaeological presence']
            },
            {
                'name': 'Central Amazon - Brazilian Interior',
                'bounds': {'min_lat': -8.0, 'max_lat': -6.0, 'min_lon': -65.0, 'max_lon': -63.0},
                'priority': 'high',
                'characteristics': ['Savanna-forest transition', 'Ancient river terraces']
            },
            {
                'name': 'Southern Amazon - Rondônia/Acre',
                'bounds': {'min_lat': -12.0, 'max_lat': -10.0, 'min_lon': -68.0, 'max_lon': -66.0},
                'priority': 'medium',
                'characteristics': ['Known geoglyphs', 'Deforestation areas revealing features']
            }
        ]
    
    def analyze_region_comprehensive(self, region: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of a specific region"""
        
        print(f"\n{'='*60}")
        print(f"ANALYZING REGION: {region['name']}")
        print(f"{'='*60}")
        
        region_results = {
            'region_info': region,
            'analysis_timestamp': datetime.now().isoformat(),
            'satellite_analysis': {},
            'potential_sites': [],
            'rag_analyses': [],
            'recommendations': []
        }
        
        # Step 1: Satellite Data Processing
        print("Step 1: Processing satellite imagery...")
        self.satellite_processor.load_image_data(mock_data=True)  # Using mock data for demo
        satellite_results = self.satellite_processor.create_composite_analysis()
        region_results['satellite_analysis'] = satellite_results
        
        print(f"Found {len(satellite_results['high_potential_sites'])} high-potential sites")
        
        # Step 2: Analyze top satellite-detected sites with OpenAI
        print("Step 2: Analyzing sites with OpenAI integration...")
        
        for i, site in enumerate(satellite_results['high_potential_sites'][:3]):  # Analyze top 3
            print(f"  Analyzing site {i+1}/3...")
            
            # Create mock coordinates within region bounds
            lat = np.random.uniform(region['bounds']['min_lat'], region['bounds']['max_lat'])
            lon = np.random.uniform(region['bounds']['min_lon'], region['bounds']['max_lon'])
            coordinates = (lat, lon)
            
            # Mock satellite data for this specific site
            site_satellite_data = {
                'metadata': {
                    'coordinates': coordinates,
                    'acquisition_date': '2024-01-15',
                    'resolution': '30cm',
                    'cloud_cover': 0.05
                },
                'features': site,
                'image_array': np.random.rand(100, 100),
                'red_band': np.random.rand(100, 100),
                'nir_band': np.random.rand(100, 100)
            }
            
            # Analyze with OpenAI integration
            site_analysis = self.site_detector.analyze_potential_site(coordinates, site_satellite_data)
            region_results['potential_sites'].append(site_analysis)
            
            # Step 3: Enhance analysis with RAG system
            print(f"  Enhancing analysis with archaeological knowledge...")
            
            site_description = f"""
            Location: {coordinates}
            Satellite features detected: Archaeological score {site['archaeological_score']:.3f}
            Pattern type: {site.get('type', 'Unknown')}
            Detection method: {site.get('detection_method', 'Unknown')}
            Area: {site.get('area', 'Unknown')} pixels
            """
            
            rag_analysis = self.rag_system.analyze_site_with_context(site_description, coordinates)
            region_results['rag_analyses'].append({
                'site_id': f"site_{i+1}",
                'coordinates': coordinates,
                'analysis': rag_analysis
            })
            
            # Generate investigation plan
            investigation_plan = self.rag_system.generate_investigation_plan(rag_analysis, coordinates)
            region_results['recommendations'].append({
                'site_id': f"site_{i+1}",
                'coordinates': coordinates,
                'investigation_plan': investigation_plan
            })
        
        return region_results
    
    def prioritize_discoveries(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize and rank all discovered sites across regions"""
        
        print(f"\n{'='*60}")
        print("PRIORITIZING DISCOVERIES")
        print(f"{'='*60}")
        
        all_sites = []
        
        # Collect all sites from all regions
        for region_result in all_results:
            region_name = region_result['region_info']['name']
            
            for site in region_result['potential_sites']:
                site_info = {
                    'region': region_name,
                    'coordinates': site['coordinates'],
                    'confidence_score': site['confidence_score'],
                    'openai_analysis': site.get('openai_analysis', {}),
                    'site_hypothesis': site.get('site_hypothesis', ''),
                    'patterns': len(site.get('geometric_patterns', [])),
                    'analysis_timestamp': site['analysis_timestamp']
                }
                
                # Calculate priority score
                priority_score = self._calculate_priority_score(site_info, region_result['region_info'])
                site_info['priority_score'] = priority_score
                
                all_sites.append(site_info)
        
        # Sort by priority score
        all_sites.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return all_sites
    
    def _calculate_priority_score(self, site: Dict[str, Any], region: Dict[str, Any]) -> float:
        """Calculate priority score for site investigation"""
        
        score = 0.0
        
        # Base confidence score
        score += site['confidence_score'] * 0.4
        
        # Region priority
        region_priority_weights = {'high': 0.3, 'medium': 0.2, 'low': 0.1}
        score += region_priority_weights.get(region.get('priority', 'low'), 0.1)
        
        # Pattern complexity
        pattern_count = site.get('patterns', 0)
        if pattern_count > 2:
            score += 0.2
        elif pattern_count > 0:
            score += 0.1
        
        # OpenAI analysis confidence (if available)
        if 'confidence_scores' in site.get('openai_analysis', {}):
            openai_scores = site['openai_analysis']['confidence_scores']
            avg_openai_score = np.mean(list(openai_scores.values()))
            score += avg_openai_score * 0.2
        
        return min(score, 1.0)
    
    def generate_final_report(self, all_results: List[Dict[str, Any]], prioritized_sites: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        print(f"\n{'='*60}")
        print("GENERATING FINAL REPORT")
        print(f"{'='*60}")
        
        report = {
            'executive_summary': {
                'total_regions_analyzed': len(all_results),
                'total_sites_detected': len(prioritized_sites),
                'high_priority_sites': len([s for s in prioritized_sites if s['priority_score'] > 0.7]),
                'analysis_date': datetime.now().isoformat(),
                'methodology': [
                    'Satellite imagery analysis using computer vision',
                    'OpenAI GPT-4 intelligent feature analysis',
                    'RAG-enhanced archaeological knowledge integration',
                    'Multi-criteria site prioritization'
                ]
            },
            'regional_summaries': [],
            'top_discoveries': prioritized_sites[:10],  # Top 10 sites
            'investigation_priorities': [],
            'technical_details': {
                'satellite_processing': 'Multi-spectral analysis with NDVI, geometric pattern detection',
                'ai_integration': 'OpenAI GPT-4o for contextual analysis and hypothesis generation',
                'knowledge_base': 'RAG system with Amazon basin archaeological database',
                'confidence_metrics': 'Multi-factor scoring including geometric regularity, vegetation anomalies'
            }
        }
        
        # Regional summaries
        for result in all_results:
            region_summary = {
                'region': result['region_info']['name'],
                'bounds': result['region_info']['bounds'],
                'sites_found': len(result['potential_sites']),
                'avg_confidence': np.mean([s['confidence_score'] for s in result['potential_sites']]) if result['potential_sites'] else 0,
                'characteristics': result['region_info'].get('characteristics', [])
            }
            report['regional_summaries'].append(region_summary)
        
        # Investigation priorities
        for i, site in enumerate(prioritized_sites[:5]):  # Top 5 for immediate investigation
            priority = {
                'rank': i + 1,
                'coordinates': site['coordinates'],
                'region': site['region'],
                'priority_score': site['priority_score'],
                'confidence_score': site['confidence_score'],
                'key_features': f"{site['patterns']} geometric patterns detected",
                'next_steps': [
                    'High-resolution satellite imagery acquisition',
                    'LiDAR survey if accessible',
                    'Ground survey planning',
                    'Consultation with local archaeological authorities'
                ]
            }
            report['investigation_priorities'].append(priority)
        
        return report
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete archaeological analysis pipeline"""
        
        print("="*80)
        print("OPENAI TO Z CHALLENGE - COMPREHENSIVE ARCHAEOLOGICAL ANALYSIS")
        print("="*80)
        print(f"Analysis started at: {datetime.now().isoformat()}")
        print(f"Analyzing {len(self.amazon_regions)} priority regions in the Amazon basin")
        
        all_results = []
        
        # Analyze each region
        for region in self.amazon_regions:
            try:
                region_result = self.analyze_region_comprehensive(region)
                all_results.append(region_result)
            except Exception as e:
                print(f"Error analyzing region {region['name']}: {e}")
                continue
        
        # Prioritize all discoveries
        prioritized_sites = self.prioritize_discoveries(all_results)
        
        # Generate final report
        final_report = self.generate_final_report(all_results, prioritized_sites)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total sites discovered: {len(prioritized_sites)}")
        print(f"High-priority sites: {final_report['executive_summary']['high_priority_sites']}")
        
        return {
            'final_report': final_report,
            'detailed_results': all_results,
            'prioritized_sites': prioritized_sites
        }

def print_executive_summary(results: Dict[str, Any]):
    """Print executive summary of results"""
    
    report = results['final_report']
    summary = report['executive_summary']
    
    print(f"\n{'='*80}")
    print("EXECUTIVE SUMMARY")
    print(f"{'='*80}")
    print(f"Analysis Date: {summary['analysis_date']}")
    print(f"Regions Analyzed: {summary['total_regions_analyzed']}")
    print(f"Total Sites Detected: {summary['total_sites_detected']}")
    print(f"High-Priority Sites: {summary['high_priority_sites']}")
    
    print(f"\nTOP 5 PRIORITY SITES FOR INVESTIGATION:")
    for priority in report['investigation_priorities']:
        print(f"{priority['rank']}. Coordinates: {priority['coordinates']}")
        print(f"   Region: {priority['region']}")
        print(f"   Priority Score: {priority['priority_score']:.3f}")
        print(f"   Confidence: {priority['confidence_score']:.3f}")
        print(f"   Features: {priority['key_features']}")
        print()

def main():
    """Main execution function"""
    
    # Check for OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Initialize comprehensive analysis system
        analysis_system = ComprehensiveArchaeologicalAnalysis(openai_api_key)
        
        # Run complete analysis
        results = analysis_system.run_complete_analysis()
        
        # Print executive summary
        print_executive_summary(results)
        
        # Save detailed results
        output_files = {
            'comprehensive_report.json': results['final_report'],
            'detailed_analysis.json': results['detailed_results'],
            'prioritized_sites.json': results['prioritized_sites']
        }
        
        for filename, data in output_files.items():
            filepath = f'/home/myuser/OpenAI_to_Z_Challenge/{filename}'
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"Results saved to: {filepath}")
        
        print(f"\n{'='*60}")
        print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        return results
        
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()