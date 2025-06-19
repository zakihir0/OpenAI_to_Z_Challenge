"""
OpenAI to Z Challenge Submission
Final optimized version for archaeological site detection
"""

from main import ArchaeologicalDetector
import os
import json

def create_submission():
    """Create final submission for Kaggle competition"""
    
    # Initialize detector with optimized settings
    detector = ArchaeologicalDetector()
    
    # Process test data
    test_data_dir = "test_images"  # Expected test data directory
    
    if os.path.exists(test_data_dir):
        print("Processing test images...")
        sites = detector.process_region(test_data_dir, "test_coordinates.json")
        
        # Export results in competition format
        submission_data = []
        for site in sites:
            submission_data.append({
                'id': f"site_{len(submission_data)+1}",
                'latitude': site.latitude,
                'longitude': site.longitude,
                'confidence': site.confidence,
                'type': site.site_type,
                'description': site.description,
                'evidence': ','.join(site.evidence_type)
            })
        
        # Save submission file
        with open('submission.json', 'w') as f:
            json.dump(submission_data, f, indent=2)
        
        print(f"Submission created with {len(submission_data)} detected sites")
        print("File saved as: submission.json")
        
    else:
        # Create example submission format
        example_submission = [
            {
                'id': 'site_1',
                'latitude': -3.4653,
                'longitude': -62.2159,
                'confidence': 0.89,
                'type': 'Structure',
                'description': 'Rectangular structure with geometric patterns',
                'evidence': 'geometric_pattern,soil_variation'
            }
        ]
        
        with open('submission_format.json', 'w') as f:
            json.dump(example_submission, f, indent=2)
        
        print("Test data not found. Created submission format example.")
        print("File saved as: submission_format.json")

if __name__ == "__main__":
    create_submission()