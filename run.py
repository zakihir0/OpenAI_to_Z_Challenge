#!/usr/bin/env python3
"""
OpenAI to Z Challenge - Archaeological Site Detection
Main execution script for running different analysis modes
"""

import os
import sys
import argparse
from datetime import datetime

def load_config():
    """Load configuration from config/.env file"""
    config_file = os.path.join('config', '.env')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    else:
        print(f"Warning: Config file {config_file} not found")

def run_basic_analysis():
    """Run basic archaeological analysis"""
    print("Running basic archaeological site detection...")
    sys.path.append('src')
    from main import main
    main()

def run_hybrid_analysis():
    """Run hybrid CV + LLM analysis"""
    print("Running hybrid CV + LLM archaeological analysis...")
    sys.path.append('src')
    from hybrid_cv_llm_solution import run_full_pipeline
    run_full_pipeline()

def run_comprehensive_analysis():
    """Run comprehensive analysis with all features"""
    print("Running comprehensive archaeological analysis...")
    sys.path.append('src')
    from openai_archaeological_analysis import main
    main()

def main():
    parser = argparse.ArgumentParser(description='OpenAI to Z Challenge - Archaeological Site Detection')
    parser.add_argument('mode', choices=['basic', 'hybrid', 'comprehensive'], 
                       help='Analysis mode to run')
    parser.add_argument('--config', default='config/.env', 
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    load_config()
    
    # Check for required API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment or config file")
        print("Please set your OpenRouter API key in config/.env")
        sys.exit(1)
    
    print(f"Starting archaeological analysis at {datetime.now()}")
    print(f"Mode: {args.mode}")
    print(f"Model: {os.getenv('OPENAI_MODEL', 'Not specified')}")
    print("-" * 60)
    
    # Run selected analysis mode
    if args.mode == 'basic':
        run_basic_analysis()
    elif args.mode == 'hybrid':
        run_hybrid_analysis()
    elif args.mode == 'comprehensive':
        run_comprehensive_analysis()
    
    print("-" * 60)
    print(f"Analysis completed at {datetime.now()}")
    print("Results saved in the 'results' directory with timestamp")

if __name__ == "__main__":
    main()