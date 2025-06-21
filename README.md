# OpenAI to Z Challenge - Archaeological Site Detection

## Overview
This project implements a comprehensive AI-powered system for detecting archaeological sites in the Amazon rainforest using satellite imagery, OpenAI's advanced models, and cutting-edge remote sensing techniques. The system combines computer vision, RAG (Retrieval-Augmented Generation), and expert archaeological knowledge to discover previously unknown sites.

## Features

### Core Capabilities
- **Multi-spectral Satellite Analysis**: Process satellite imagery with NDVI, EVI, and other vegetation indices
- **LIDAR Point Cloud Processing**: Complete .las/.laz file processing with DTM/DSM extraction
- **Advanced Terrain Analysis**: Hillshade, slope, aspect, curvature, and local relief calculation
- **AI-Powered Structure Detection**: Hybrid CNN/traditional algorithms for archaeological feature identification
- **Multimodal LLM Integration**: Image + text analysis using Deepseek and other vision models
- **Geometric Pattern Detection**: Identify circular earthworks, linear features, and geometric anomalies
- **Cultural Context RAG System**: Archaeological knowledge base with vector similarity search
- **Comprehensive GIS Export**: Shapefile, GeoJSON, KML, GeoTIFF, and CSV format support
- **3D Visualization**: Interactive terrain and structure visualization

### Archaeological Techniques
- **LIDAR Remote Sensing**: Point cloud analysis for ground surface extraction
- **Hillshade Analysis**: Shadow-based terrain visualization for feature detection
- **Structure Detection Algorithms**: Edge detection, Hough transforms, and morphological analysis
- **Deep Learning Classification**: CNN and U-Net models for automated feature recognition
- **Cultural Context Integration**: RAG-based comparison with known archaeological sites
- **Multi-criteria Assessment**: Comprehensive scoring combining terrain, AI, and cultural factors

## Project Structure

```
/home/myuser/OpenAI_to_Z_Challenge/
├── run.py                                    # Main execution script (satellite analysis)
├── run_lidar_analysis.py                     # LIDAR analysis runner
├── src/                                      # Source code
│   ├── main.py                               # Basic archaeological analysis
│   ├── archaeological_analyzer.py            # Core analysis engine
│   ├── lidar_archaeological_processor.py     # LIDAR processing pipeline
│   ├── ai_structure_detector.py              # AI-powered structure detection
│   ├── multimodal_archaeological_pipeline.py # Complete multimodal pipeline
│   ├── gis_archaeological_exporter.py        # GIS format export
│   ├── visualization_engine.py               # Visualization components
│   └── kaggle_integration.py                 # Kaggle dataset integration
├── config/                                   # Configuration files
│   ├── lidar_config.json                     # LIDAR pipeline configuration
│   ├── .env                                  # API keys and settings
│   └── .env.example                          # Configuration template
├── results/                                  # Timestamped analysis outputs
├── data/                                    # Input data and cache
├── requirements.txt                          # Python dependencies
└── README.md                               # This file
```

## Installation

### Prerequisites
- Python 3.8+
- OpenRouter API key (supports Deepseek and other models)
- Required Python packages

### Setup
```bash
# Clone or navigate to project directory
cd /home/myuser/OpenAI_to_Z_Challenge

# Install requirements
pip install -r requirements.txt

# Copy environment template and configure API settings
cp .env.example .env

# Edit .env file with your OpenRouter API key
# Get your API key from https://openrouter.ai/
# OPENAI_API_KEY=your_openrouter_api_key_here
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# OPENAI_MODEL=deepseek/deepseek-r1-0528:free

# Alternative: Set environment variables directly
export OPENAI_API_KEY='your-openrouter-api-key'
export OPENAI_BASE_URL='https://openrouter.ai/api/v1'
export OPENAI_MODEL='deepseek/deepseek-r1-0528:free'

# Optional: Install additional geospatial libraries
pip install earthengine-api  # Requires additional Google Earth Engine setup
```

## Usage

### Quick Start

#### Satellite Analysis (Original)
```bash
# Basic analysis (fast, simple detection)
python run.py basic

# Hybrid CV + LLM analysis (comprehensive, recommended)
python run.py hybrid

# Full comprehensive analysis (all features)
python run.py comprehensive
```

#### LIDAR Analysis (New Multimodal Pipeline)
```bash
# Basic LIDAR processing
python run_lidar_analysis.py --input data.las --output results/ --pipeline basic

# Advanced analysis with structure detection
python run_lidar_analysis.py --input data.las --output results/ --pipeline advanced \
    --site-name "Amazon Site 1" --coordinates "-8.5,-63.2"

# Complete multimodal analysis (recommended)
python run_lidar_analysis.py --input data.las --output results/ --pipeline complete \
    --site-name "Casarabe Complex" --region "Llanos de Mojos" \
    --coordinates "-14.8,-64.9" --config config/lidar_config.json
```

### Analysis Modes

#### Satellite Analysis Modes
1. **Basic Analysis** (`python run.py basic`) - Fast geometric pattern detection
2. **Hybrid CV + LLM** (`python run.py hybrid`) - Computer vision + AI analysis  
3. **Comprehensive** (`python run.py comprehensive`) - Full feature extraction

#### LIDAR Analysis Modes  
1. **Basic LIDAR** (`--pipeline basic`) - DTM/DSM extraction and basic terrain analysis
2. **Advanced** (`--pipeline advanced`) - Structure detection + GIS export
3. **Complete Multimodal** (`--pipeline complete`) - Full AI pipeline with:
   - LIDAR point cloud processing (.las/.laz files)
   - Advanced terrain analysis (hillshade, slope, curvature)
   - AI-powered structure detection (CNN + traditional algorithms)
   - Multimodal LLM analysis (image + text with Deepseek)
   - Cultural context integration (RAG system)
   - Comprehensive GIS export (Shapefile, GeoJSON, KML, GeoTIFF)
   - Archaeological assessment and recommendations

### Individual Module Usage
```python
# Import specific modules
sys.path.append('src')
from main import ArchaeologicalDetector
from hybrid_cv_llm_solution import run_full_pipeline
from openai_archaeological_analysis import ArchaeologicalSiteDetector

# Use individual components as needed
detector = ArchaeologicalDetector()
sites = detector.process_region('data/images', 'data/coordinates.json')
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
# OpenRouter + Deepseek Configuration
export OPENAI_API_KEY='your-openrouter-api-key'
export OPENAI_BASE_URL='https://openrouter.ai/api/v1'
export OPENAI_MODEL='deepseek/deepseek-r1-0528:free'

# Optional: Google Earth Engine
export GOOGLE_EARTH_ENGINE_KEY='path/to/ee-key.json'
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
1. **OpenRouter API Key**: Ensure valid OpenRouter key is set in environment
2. **Model Selection**: Verify model name is correct (e.g., deepseek/deepseek-r1-0528:free)
3. **Base URL**: Confirm OPENAI_BASE_URL points to https://openrouter.ai/api/v1
4. **Memory Issues**: Reduce image tile size for large datasets
5. **Dependency Conflicts**: Use virtual environment for isolation
6. **Coordinate Systems**: Ensure consistent CRS across datasets

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