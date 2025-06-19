# OpenAI to Z Challenge - Archaeological Site Detection

## Overview
This project implements an AI-powered system for detecting archaeological sites in the Amazon rainforest using satellite imagery and OpenAI's advanced models.

## Features
- Multi-modal satellite image analysis
- Computer vision-based geometric pattern detection
- Vegetation index calculation for anomaly detection
- Soil mark analysis for subsurface structure identification
- OpenAI GPT-4V integration for advanced image interpretation
- Automated site classification and confidence scoring

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place satellite images in `data/images/` directory
2. Update `data/coordinates.json` with image coordinates
3. Set OpenAI API key: `export OPENAI_API_KEY=your_key_here`
4. Run detection: `python main.py`

## Output
Results are saved to:
- `results/archaeological_sites.json` - Detailed detection results
- `results/archaeological_sites.csv` - CSV format for analysis

## Methodology
The system combines:
- Edge detection and contour analysis
- Vegetation anomaly detection (NDVI approximation)
- Soil color variation analysis
- AI-powered image interpretation
- Multi-sensor data fusion

## Technical Approach
- **Computer Vision**: OpenCV for image processing and feature extraction
- **Machine Learning**: Pattern recognition for archaeological signatures
- **AI Integration**: OpenAI models for contextual analysis
- **Geospatial**: Coordinate mapping and spatial analysis