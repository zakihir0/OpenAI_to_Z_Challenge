# OpenAI to Z Challenge - Archaeological Site Detection

## Overview
This project implements a comprehensive AI-powered system for detecting archaeological sites in the Amazon rainforest using satellite imagery, OpenAI's advanced models, and cutting-edge remote sensing techniques. The system combines computer vision, RAG (Retrieval-Augmented Generation), and expert archaeological knowledge to discover previously unknown sites.

## Features

### Core Capabilities
- **Multi-spectral Satellite Analysis**: Process satellite imagery with NDVI, EVI, and other vegetation indices
- **Geometric Pattern Detection**: Identify circular earthworks, linear features, and geometric anomalies
- **OpenAI GPT-4 Integration**: Intelligent analysis and hypothesis generation
- **RAG Knowledge System**: Archaeological knowledge base with vector similarity search
- **Vegetation Anomaly Detection**: Identify areas of unusual vegetation patterns
- **Machine Learning Classification**: Automated site scoring and prioritization

### Archaeological Techniques
- Remote sensing for earthwork detection
- Vegetation analysis for buried structure identification
- Soil mark analysis using spectral signatures
- Template matching for common archaeological shapes
- Multi-criteria decision analysis for site prioritization

## Project Structure

```
/home/myuser/OpenAI_to_Z_Challenge/
├── openai_archaeological_analysis.py  # Main OpenAI integration and site detection
├── rag_knowledge_base.py              # RAG system with archaeological knowledge
├── satellite_data_processing.py       # Satellite imagery processing and analysis
├── comprehensive_example.py           # Complete pipeline demonstration
├── requirements.txt                   # Python dependencies
├── data/                             # Sample data and coordinates
├── results/                          # Analysis outputs
└── kernels/                          # Downloaded Kaggle notebooks
```

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Required Python packages

### Setup
```bash
# Clone or navigate to project directory
cd /home/myuser/OpenAI_to_Z_Challenge

# Install requirements
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Optional: Install additional geospatial libraries
pip install earthengine-api  # Requires additional Google Earth Engine setup
```

## Usage

### Quick Start - Comprehensive Analysis
```bash
# Run complete archaeological analysis pipeline
python comprehensive_example.py
```

### Individual Components

#### 1. OpenAI Archaeological Analysis
```python
from openai_archaeological_analysis import ArchaeologicalSiteDetector

detector = ArchaeologicalSiteDetector(api_key="your-key")
coordinates = (-8.5, -63.2)  # Example Amazon coordinates
satellite_data = {...}  # Your satellite data

analysis = detector.analyze_potential_site(coordinates, satellite_data)
```

#### 2. RAG Knowledge System
```python
from rag_knowledge_base import RAGArchaeologist

rag_system = RAGArchaeologist(api_key="your-key")

site_description = "Circular earthwork structures near river tributary..."
analysis = rag_system.analyze_site_with_context(site_description, coordinates)
investigation_plan = rag_system.generate_investigation_plan(analysis, coordinates)
```

#### 3. Satellite Data Processing
```python
from satellite_data_processing import SatelliteImageProcessor

processor = SatelliteImageProcessor()
bands = processor.load_image_data()  # Load your satellite data
indices = processor.calculate_spectral_indices()
patterns = processor.detect_geometric_patterns()
results = processor.create_composite_analysis()
```

### Input Data Formats

#### Satellite Data
- **Multi-spectral bands**: Red, NIR, SWIR1, SWIR2, Blue, Green
- **Supported formats**: GeoTIFF, HDF, NetCDF (via rasterio)
- **Coordinate system**: WGS84 (EPSG:4326) preferred
- **Resolution**: 30m or higher recommended

#### Coordinates
```json
{
  "coordinates": [-8.5, -63.2],
  "region": "Western Amazon",
  "metadata": {
    "acquisition_date": "2024-01-15",
    "cloud_cover": 0.05
  }
}
```

## Output Formats

### Analysis Results
```json
{
  "coordinates": [-8.5, -63.2],
  "confidence_score": 0.85,
  "geometric_patterns": [...],
  "openai_analysis": {...},
  "enhanced_analysis": "...",
  "site_hypothesis": "...",
  "vegetation_indices": {...}
}
```

### Final Report
- **Executive Summary**: High-level findings and statistics
- **Site Prioritization**: Ranked list of potential archaeological sites
- **Investigation Plans**: Detailed recommendations for each site
- **Technical Details**: Methodology and confidence metrics

## Methodology

### 1. Satellite Data Processing
- **Spectral Analysis**: NDVI, EVI, NDWI, Iron Oxide Ratio
- **Pattern Detection**: Edge detection, template matching, ML clustering
- **Vegetation Anomalies**: Statistical analysis of vegetation patterns
- **Geometric Features**: Circularity, regularity, aspect ratio analysis

### 2. OpenAI Integration
- **Feature Analysis**: GPT-4 analysis of satellite-detected features
- **Hypothesis Generation**: AI-powered archaeological interpretation
- **Contextual Understanding**: Integration of multiple data sources

### 3. RAG Knowledge System
- **Archaeological Database**: Known Amazon basin sites and research
- **Vector Similarity**: Embedding-based context retrieval
- **Expert Knowledge**: Integration of archaeological methodology
- **Comparative Analysis**: Similarity to known archaeological sites

### 4. Site Prioritization
- **Multi-criteria Scoring**: Confidence, regularity, size, vegetation anomalies
- **Regional Context**: Priority based on known archaeological potential
- **Investigation Feasibility**: Accessibility and research value

## Amazon Basin Focus Areas

### High Priority Regions
1. **Western Amazon (Peru/Ecuador Border)**
   - Dense forest with river systems
   - Known archaeological presence
   - Coordinates: 5°S-3°S, 78°W-76°W

2. **Central Amazon (Brazilian Interior)**
   - Savanna-forest transition zones
   - Ancient river terraces
   - Coordinates: 8°S-6°S, 65°W-63°W

3. **Southern Amazon (Rondônia/Acre)**
   - Known geoglyph locations
   - Deforestation revealing features
   - Coordinates: 12°S-10°S, 68°W-66°W

## Archaeological Context

### Known Site Types
- **Fortified Settlements**: Earthworks, defensive structures (1000-1500 CE)
- **Ceremonial Complexes**: Large earthen mounds, plazas (500-1500 CE)
- **Agricultural Terraces**: Raised fields, drainage systems (1000-1500 CE)
- **Urban Settlements**: Complex societies with trade networks

### Detection Indicators
- Geometric earthwork patterns (circles, squares, octagons)
- Vegetation anomalies (forest islands, species composition)
- Topographic signatures (mounds, platforms, ditches)
- Soil marks and spectral anomalies

## Technical Requirements

### Hardware Recommendations
- **RAM**: 16GB+ for large satellite image processing
- **Storage**: 100GB+ for satellite data and results
- **GPU**: Optional, for accelerated image processing
- **Network**: Stable connection for OpenAI API calls

### Software Dependencies
- **Core**: OpenAI, NumPy, Pandas, Scikit-learn
- **Geospatial**: Rasterio, GeoPandas, Shapely
- **Image Processing**: OpenCV, Scikit-image
- **Visualization**: Matplotlib, Plotly, Folium
- **Optional**: ChromaDB, FAISS for vector storage

## Configuration

### Environment Variables
```bash
export OPENAI_API_KEY='your-openai-api-key'
export GOOGLE_EARTH_ENGINE_KEY='path/to/ee-key.json'  # Optional
```

### API Rate Limits
- OpenAI API calls are optimized to respect rate limits
- Batch processing for multiple site analysis
- Caching implemented for repeated analyses

## Contributing

### Research Applications
This codebase supports research in:
- Archaeological remote sensing
- AI-assisted site discovery
- Amazon basin prehistory
- Conservation archaeology

### Extensions
- Integration with Google Earth Engine
- Real-time satellite data feeds
- Mobile field application development
- Drone survey integration

## Validation and Results

### Mock Data Testing
The system includes realistic mock data generation for:
- Multi-spectral satellite imagery
- Archaeological feature signatures
- Vegetation index calculations
- Pattern detection algorithms

### Performance Metrics
- **Detection Accuracy**: Pattern recognition success rate
- **Confidence Calibration**: Score reliability assessment
- **Processing Speed**: Analysis time per square kilometer
- **API Efficiency**: OpenAI token usage optimization

## Troubleshooting

### Common Issues
1. **OpenAI API Key**: Ensure valid key is set in environment
2. **Memory Issues**: Reduce image tile size for large datasets
3. **Dependency Conflicts**: Use virtual environment for isolation
4. **Coordinate Systems**: Ensure consistent CRS across datasets

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License and Citation

### Academic Use
If using this code for research, please cite:
```
OpenAI to Z Challenge Archaeological Detection System
Amazon Basin Site Discovery using AI and Remote Sensing
2024
```

### Data Sources
- Satellite imagery: Various providers (Landsat, Sentinel, WorldView)
- Archaeological context: Published research and site databases
- Knowledge base: Curated from peer-reviewed literature

## Support and Contact

For technical issues or research collaboration:
- Create detailed issue reports with sample data
- Include system specifications and error messages
- Provide coordinates and analysis parameters used

---

**Note**: This system is designed for research and educational purposes in archaeological site discovery. All potential archaeological sites should be verified through proper archaeological investigation and local authority consultation before any ground-disturbing activities.