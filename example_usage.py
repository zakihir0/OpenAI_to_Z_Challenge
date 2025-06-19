"""
Example usage of the Archaeological Site Detection System
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple
from main import ArchaeologicalDetector, ArchaeologicalSite

def create_sample_satellite_image(filename: str, size: Tuple[int, int] = (1024, 1024)):
    """Create a sample satellite image with simulated archaeological features"""
    # Create base vegetation image (green)
    image = Image.new('RGB', size, color=(34, 139, 34))  # Forest green
    draw = ImageDraw.Draw(image)
    
    # Add some rectangular structures (simulated archaeological sites)
    # Structure 1: Large rectangular platform
    draw.rectangle([200, 200, 350, 280], fill=(101, 67, 33))  # Brown soil
    draw.rectangle([210, 210, 340, 270], outline=(139, 69, 19), width=3)
    
    # Structure 2: Circular plaza
    draw.ellipse([500, 400, 650, 550], fill=(160, 82, 45))  # Reddish soil
    draw.ellipse([510, 410, 640, 540], outline=(139, 69, 19), width=2)
    
    # Structure 3: Linear pathway
    draw.rectangle([100, 600, 900, 620], fill=(205, 133, 63))  # Light brown
    
    # Add some random vegetation variation
    np.random.seed(42)
    pixels = np.array(image)
    
    # Add vegetation noise
    for i in range(1000):
        x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
        if np.random.random() > 0.7:
            # Lighter green spots (clearings)
            pixels[y:y+5, x:x+5] = [60, 179, 113]
        elif np.random.random() > 0.9:
            # Darker spots (dense vegetation)
            pixels[y:y+3, x:x+3] = [0, 100, 0]
    
    # Convert back to image
    final_image = Image.fromarray(pixels.astype('uint8'))
    final_image.save(filename)
    return filename

def run_example():
    """Run complete example workflow"""
    print("OpenAI to Z Challenge - Example Usage")
    print("=" * 50)
    
    # Create directories
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Create sample images
    sample_images = []
    coordinates = {}
    
    for i in range(3):
        filename = f'data/images/amazon_satellite_{i+1}.jpg'
        create_sample_satellite_image(filename)
        sample_images.append(filename)
        
        # Generate sample coordinates (Amazon region)
        lat = -3.4653 + (i * 0.1)  # Manaus region
        lon = -62.2159 + (i * 0.1)
        coordinates[f'amazon_satellite_{i+1}.jpg'] = [lat, lon]
    
    # Save coordinates
    with open('data/coordinates.json', 'w') as f:
        json.dump(coordinates, f, indent=2)
    
    print(f"Created {len(sample_images)} sample satellite images")
    print("Images saved to data/images/")
    print("Coordinates saved to data/coordinates.json")
    
    # Initialize detector
    detector = ArchaeologicalDetector()
    
    # Process images
    print("\nProcessing images for archaeological features...")
    sites = detector.process_region('data/images', 'data/coordinates.json')
    
    # Export results
    detector.export_results(sites, 'results/archaeological_sites.json')
    
    # Display results
    print(f"\nDetection complete! Found {len(sites)} potential sites:")
    print("-" * 50)
    
    for i, site in enumerate(sites):
        print(f"Site {i+1}:")
        print(f"  Location: {site.latitude:.4f}, {site.longitude:.4f}")
        print(f"  Confidence: {site.confidence:.2f}")
        print(f"  Type: {site.site_type}")
        print(f"  Description: {site.description}")
        print(f"  Evidence: {', '.join(site.evidence_type)}")
        print()
    
    print("Results exported to:")
    print("- results/archaeological_sites.json")
    print("- results/archaeological_sites.csv")

if __name__ == "__main__":
    run_example()