#!/usr/bin/env python3
"""
Archaeological Site Analysis Runner
Clean entry point for the refactored archaeological analysis system
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main execution function"""
    # Load environment variables from config
    config_file = os.path.join('config', '.env')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Import and run main analysis
    from main import main as run_analysis
    run_analysis()

if __name__ == "__main__":
    main()