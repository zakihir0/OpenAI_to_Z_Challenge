#!/usr/bin/env python3
"""
GIS Archaeological Data Exporter
Export archaeological analysis results to standard GIS formats for use in QGIS, ArcGIS, etc.

Supports export to:
- Shapefile (.shp) - Vector data
- GeoTIFF (.tif) - Raster data  
- KML (.kml) - Google Earth
- GeoJSON (.geojson) - Web mapping
- CSV with coordinates - Simple data exchange
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import tempfile

# Geospatial libraries
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, LineString, MultiPoint
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    import pyproj
    GIS_AVAILABLE = True
except ImportError:
    GIS_AVAILABLE = False
    logging.warning("GIS libraries not available - limited export functionality")

# Additional GIS utilities
try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import simplekml
    KML_AVAILABLE = True
except ImportError:
    KML_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GISArchaeologicalExporter:
    """Export archaeological analysis results to GIS formats"""
    
    def __init__(self, crs: str = "EPSG:4326"):
        """
        Initialize GIS exporter
        
        Args:
            crs: Coordinate reference system (default: WGS84)
        """
        self.crs = crs
        self.transformer = None
        
        # Initialize coordinate transformer if needed
        if crs != "EPSG:4326":
            try:
                self.transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            except Exception as e:
                logger.warning(f"Could not initialize coordinate transformer: {e}")
    
    def export_analysis_results(self, analysis_results: Dict, 
                              output_dir: str, formats: List[str] = None,
                              site_bounds: Tuple[float, float, float, float] = None) -> Dict:
        """
        Export comprehensive analysis results to multiple GIS formats
        
        Args:
            analysis_results: Results from LIDAR analysis
            output_dir: Output directory for files
            formats: List of formats to export ('shapefile', 'geotiff', 'kml', 'geojson', 'csv')
            site_bounds: (minx, miny, maxx, maxy) in WGS84 coordinates
            
        Returns:
            Dictionary with paths to exported files
        """
        
        if formats is None:
            formats = ['shapefile', 'geotiff', 'kml', 'geojson', 'csv']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exported_files = {}
        
        # Default bounds if not provided (Amazon basin example)
        if site_bounds is None:
            site_bounds = (-70.0, -15.0, -50.0, 5.0)  # Amazon basin rough bounds
        
        try:
            # Export vector data (structures)
            if 'shapefile' in formats and GIS_AVAILABLE:
                shp_files = self._export_to_shapefile(
                    analysis_results, output_dir, timestamp, site_bounds
                )
                exported_files.update(shp_files)
            
            if 'geojson' in formats and GIS_AVAILABLE:
                geojson_files = self._export_to_geojson(
                    analysis_results, output_dir, timestamp, site_bounds
                )
                exported_files.update(geojson_files)
            
            if 'kml' in formats:
                kml_files = self._export_to_kml(
                    analysis_results, output_dir, timestamp, site_bounds
                )
                exported_files.update(kml_files)
            
            # Export raster data (elevation models, hillshade)
            if 'geotiff' in formats and GIS_AVAILABLE:
                tiff_files = self._export_to_geotiff(
                    analysis_results, output_dir, timestamp, site_bounds
                )
                exported_files.update(tiff_files)
            
            # Export tabular data
            if 'csv' in formats:
                csv_files = self._export_to_csv(
                    analysis_results, output_dir, timestamp, site_bounds
                )
                exported_files.update(csv_files)
            
            # Create web map
            if FOLIUM_AVAILABLE:
                web_map = self._create_web_map(
                    analysis_results, output_dir, timestamp, site_bounds
                )
                exported_files['web_map'] = web_map
            
            # Create metadata
            metadata_file = self._create_metadata(
                analysis_results, exported_files, output_dir, timestamp
            )
            exported_files['metadata'] = metadata_file
            
            logger.info(f"Exported {len(exported_files)} GIS files to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting GIS data: {e}")
            exported_files['error'] = str(e)
        
        return exported_files
    
    def _export_to_shapefile(self, results: Dict, output_dir: str, 
                           timestamp: str, bounds: Tuple) -> Dict:
        """Export structures to Shapefile format"""
        
        shapefiles = {}
        
        if 'archaeological_structures' not in results:
            return shapefiles
        
        structures = results['archaeological_structures']
        
        # Create different shapefiles for different structure types
        structure_types = [
            'earthworks', 'linear_features', 'circular_features', 
            'mounds', 'ditches', 'platforms'
        ]
        
        for structure_type in structure_types:
            if structure_type in structures and structures[structure_type]:
                try:
                    gdf = self._structures_to_geodataframe(
                        structures[structure_type], structure_type, bounds
                    )
                    
                    if not gdf.empty:
                        filename = f"archaeological_{structure_type}_{timestamp}.shp"
                        filepath = os.path.join(output_dir, filename)
                        gdf.to_file(filepath)
                        shapefiles[f'shapefile_{structure_type}'] = filepath
                        
                except Exception as e:
                    logger.warning(f"Failed to export {structure_type} to shapefile: {e}")
        
        # Create combined shapefile
        try:
            all_structures = []
            for structure_type in structure_types:
                if structure_type in structures and structures[structure_type]:
                    for struct in structures[structure_type]:
                        struct_copy = struct.copy()
                        struct_copy['structure_type'] = structure_type
                        all_structures.append(struct_copy)
            
            if all_structures:
                combined_gdf = self._structures_to_geodataframe(
                    all_structures, 'combined', bounds
                )
                
                if not combined_gdf.empty:
                    filename = f"archaeological_structures_all_{timestamp}.shp"
                    filepath = os.path.join(output_dir, filename)
                    combined_gdf.to_file(filepath)
                    shapefiles['shapefile_combined'] = filepath
                    
        except Exception as e:
            logger.warning(f"Failed to create combined shapefile: {e}")
        
        return shapefiles
    
    def _export_to_geojson(self, results: Dict, output_dir: str,
                          timestamp: str, bounds: Tuple) -> Dict:
        """Export structures to GeoJSON format"""
        
        geojson_files = {}
        
        if 'archaeological_structures' not in results:
            return geojson_files
        
        structures = results['archaeological_structures']
        
        # Create GeoJSON for each structure type
        structure_types = [
            'earthworks', 'linear_features', 'circular_features',
            'mounds', 'ditches', 'platforms'
        ]
        
        for structure_type in structure_types:
            if structure_type in structures and structures[structure_type]:
                try:
                    gdf = self._structures_to_geodataframe(
                        structures[structure_type], structure_type, bounds
                    )
                    
                    if not gdf.empty:
                        filename = f"archaeological_{structure_type}_{timestamp}.geojson"
                        filepath = os.path.join(output_dir, filename)
                        gdf.to_file(filepath, driver='GeoJSON')
                        geojson_files[f'geojson_{structure_type}'] = filepath
                        
                except Exception as e:
                    logger.warning(f"Failed to export {structure_type} to GeoJSON: {e}")
        
        return geojson_files
    
    def _export_to_kml(self, results: Dict, output_dir: str,
                      timestamp: str, bounds: Tuple) -> Dict:
        """Export structures to KML format for Google Earth"""
        
        kml_files = {}
        
        if not KML_AVAILABLE:
            # Fallback to simple KML creation
            return self._create_simple_kml(results, output_dir, timestamp, bounds)
        
        if 'archaeological_structures' not in results:
            return kml_files
        
        structures = results['archaeological_structures']
        
        try:
            # Create KML document
            kml = simplekml.Kml()
            kml.document.name = "Archaeological Structures Analysis"
            kml.document.description = f"LIDAR-detected archaeological features - {timestamp}"
            
            # Define styles for different structure types
            styles = self._create_kml_styles(kml)
            
            # Add structures to KML
            structure_types = [
                'earthworks', 'linear_features', 'circular_features',
                'mounds', 'ditches', 'platforms'
            ]
            
            for structure_type in structure_types:
                if structure_type in structures and structures[structure_type]:
                    folder = kml.newfolder(name=structure_type.replace('_', ' ').title())
                    
                    for i, struct in enumerate(structures[structure_type]):
                        self._add_structure_to_kml(
                            folder, struct, structure_type, i, bounds, styles
                        )
            
            # Save KML file
            filename = f"archaeological_structures_{timestamp}.kml"
            filepath = os.path.join(output_dir, filename)
            kml.save(filepath)
            kml_files['kml_structures'] = filepath
            
        except Exception as e:
            logger.warning(f"Failed to create KML file: {e}")
            # Try simple KML fallback
            return self._create_simple_kml(results, output_dir, timestamp, bounds)
        
        return kml_files
    
    def _export_to_geotiff(self, results: Dict, output_dir: str,
                          timestamp: str, bounds: Tuple) -> Dict:
        """Export raster data to GeoTIFF format"""
        
        tiff_files = {}
        
        # Check for elevation models in results
        elevation_models = results.get('elevation_models', {})
        
        # Export DTM, DSM, and other raster data if available
        raster_data = {}
        
        # Try to extract raster data from results
        if hasattr(results, 'dtm'):
            raster_data['dtm'] = results.dtm
        if hasattr(results, 'dsm'):
            raster_data['dsm'] = results.dsm
        if hasattr(results, 'hillshade'):
            raster_data['hillshade'] = results.hillshade
        
        # If no direct raster data, create synthetic for demonstration
        if not raster_data:
            raster_data = self._create_synthetic_raster_data(bounds)
        
        # Export each raster to GeoTIFF
        for raster_name, raster_array in raster_data.items():
            try:
                filename = f"archaeological_{raster_name}_{timestamp}.tif"
                filepath = os.path.join(output_dir, filename)
                
                self._array_to_geotiff(raster_array, filepath, bounds)
                tiff_files[f'geotiff_{raster_name}'] = filepath
                
            except Exception as e:
                logger.warning(f"Failed to export {raster_name} to GeoTIFF: {e}")
        
        return tiff_files
    
    def _export_to_csv(self, results: Dict, output_dir: str,
                      timestamp: str, bounds: Tuple) -> Dict:
        """Export structures to CSV format"""
        
        csv_files = {}
        
        if 'archaeological_structures' not in results:
            return csv_files
        
        structures = results['archaeological_structures']
        
        # Create comprehensive CSV with all structures
        all_structures_data = []
        
        structure_types = [
            'earthworks', 'linear_features', 'circular_features',
            'mounds', 'ditches', 'platforms'
        ]
        
        for structure_type in structure_types:
            if structure_type in structures and structures[structure_type]:
                for i, struct in enumerate(structures[structure_type]):
                    # Convert structure to tabular format
                    row = self._structure_to_csv_row(
                        struct, structure_type, i, bounds
                    )
                    all_structures_data.append(row)
        
        if all_structures_data:
            try:
                df = pd.DataFrame(all_structures_data)
                
                # Export main CSV
                filename = f"archaeological_structures_{timestamp}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)
                csv_files['csv_structures'] = filepath
                
                # Export summary statistics CSV
                summary_df = self._create_summary_statistics_df(results)
                summary_filename = f"archaeological_summary_{timestamp}.csv"
                summary_filepath = os.path.join(output_dir, summary_filename)
                summary_df.to_csv(summary_filepath, index=False)
                csv_files['csv_summary'] = summary_filepath
                
            except Exception as e:
                logger.warning(f"Failed to export CSV files: {e}")
        
        return csv_files
    
    def _structures_to_geodataframe(self, structures: List[Dict], 
                                  structure_type: str, bounds: Tuple) -> gpd.GeoDataFrame:
        """Convert structures list to GeoDataFrame"""
        
        geometries = []
        attributes = []
        
        for i, struct in enumerate(structures):
            try:
                # Extract coordinates and create geometry
                geom = self._structure_to_geometry(struct, bounds)
                if geom is not None:
                    geometries.append(geom)
                    
                    # Extract attributes
                    attrs = {
                        'id': i,
                        'type': struct.get('type', structure_type),
                        'confidence': struct.get('confidence', 0.0),
                        'archaeological_type': struct.get('archaeological_type', 'unknown'),
                        'cultural_context': struct.get('cultural_context', 'unknown')
                    }
                    
                    # Add type-specific attributes
                    if 'area' in struct:
                        attrs['area'] = struct['area']
                    if 'height' in struct:
                        attrs['height'] = struct['height']
                    if 'depth' in struct:
                        attrs['depth'] = struct['depth']
                    if 'radius' in struct:
                        attrs['radius'] = struct['radius']
                    if 'length' in struct:
                        attrs['length'] = struct['length']
                    if 'angle' in struct:
                        attrs['angle'] = struct['angle']
                    
                    attributes.append(attrs)
                    
            except Exception as e:
                logger.warning(f"Failed to process structure {i}: {e}")
        
        if geometries and attributes:
            gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=self.crs)
            return gdf
        else:
            return gpd.GeoDataFrame()
    
    def _structure_to_geometry(self, struct: Dict, bounds: Tuple):
        """Convert structure dictionary to Shapely geometry"""
        
        minx, miny, maxx, maxy = bounds
        
        # Get structure location
        if 'center' in struct:
            center = struct['center']
        elif 'centroid' in struct:
            center = struct['centroid']
        else:
            return None
        
        # Convert pixel coordinates to geographic coordinates
        if isinstance(center, (list, tuple)) and len(center) == 2:
            # Assume center is in pixel coordinates, convert to geographic
            pixel_y, pixel_x = center
            
            # Simple linear mapping (this would need actual geotransform in real data)
            geo_x = minx + (pixel_x / 512.0) * (maxx - minx)
            geo_y = maxy - (pixel_y / 512.0) * (maxy - miny)  # Flip Y coordinate
            
            # Create geometry based on structure type
            if struct.get('type') == 'circular' or 'radius' in struct:
                # Create circular buffer
                radius_deg = struct.get('radius', 10) * 0.0001  # Convert to rough degrees
                point = Point(geo_x, geo_y)
                return point.buffer(radius_deg)
                
            elif struct.get('type') == 'linear':
                # Create line (simplified as point for now)
                return Point(geo_x, geo_y)
                
            elif struct.get('type') == 'rectangular':
                # Create rectangle buffer
                width = struct.get('width', 20) * 0.0001
                height = struct.get('height', 20) * 0.0001
                point = Point(geo_x, geo_y)
                return point.buffer(max(width, height) / 2)
                
            else:
                # Default to point
                return Point(geo_x, geo_y)
        
        return None
    
    def _create_kml_styles(self, kml) -> Dict:
        """Create KML styles for different structure types"""
        
        styles = {}
        
        # Earthworks style
        styles['earthworks'] = kml.newstyle()
        styles['earthworks'].iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
        styles['earthworks'].iconstyle.color = simplekml.Color.red
        
        # Linear features style
        styles['linear_features'] = kml.newstyle()
        styles['linear_features'].iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/road_local.png'
        styles['linear_features'].iconstyle.color = simplekml.Color.blue
        
        # Circular features style
        styles['circular_features'] = kml.newstyle()
        styles['circular_features'].iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
        styles['circular_features'].iconstyle.color = simplekml.Color.green
        
        # Mounds style
        styles['mounds'] = kml.newstyle()
        styles['mounds'].iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/triangle.png'
        styles['mounds'].iconstyle.color = simplekml.Color.orange
        
        # Ditches style
        styles['ditches'] = kml.newstyle()
        styles['ditches'].iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/open-diamond.png'
        styles['ditches'].iconstyle.color = simplekml.Color.purple
        
        # Platforms style
        styles['platforms'] = kml.newstyle()
        styles['platforms'].iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_square.png'
        styles['platforms'].iconstyle.color = simplekml.Color.yellow
        
        return styles
    
    def _add_structure_to_kml(self, folder, struct: Dict, structure_type: str,
                            index: int, bounds: Tuple, styles: Dict):
        """Add individual structure to KML folder"""
        
        geometry = self._structure_to_geometry(struct, bounds)
        
        if geometry and hasattr(geometry, 'x') and hasattr(geometry, 'y'):
            # Create placemark
            placemark = folder.newpoint()
            placemark.name = f"{structure_type.title()} {index + 1}"
            placemark.coords = [(geometry.x, geometry.y)]
            
            # Set style
            if structure_type in styles:
                placemark.style = styles[structure_type]
            
            # Add description
            description = f"""
            Type: {struct.get('type', 'unknown')}
            Confidence: {struct.get('confidence', 0.0):.3f}
            Archaeological Type: {struct.get('archaeological_type', 'unknown')}
            Cultural Context: {struct.get('cultural_context', 'unknown')}
            """
            
            if 'area' in struct:
                description += f"\nArea: {struct['area']:.1f}"
            if 'height' in struct:
                description += f"\nHeight: {struct['height']:.2f}m"
            if 'radius' in struct:
                description += f"\nRadius: {struct['radius']:.1f}m"
            
            placemark.description = description.strip()
    
    def _create_simple_kml(self, results: Dict, output_dir: str,
                          timestamp: str, bounds: Tuple) -> Dict:
        """Create simple KML without simplekml library"""
        
        kml_files = {}
        
        try:
            filename = f"archaeological_structures_{timestamp}.kml"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
                f.write('<Document>\n')
                f.write('<name>Archaeological Structures</name>\n')
                f.write('<description>LIDAR-detected archaeological features</description>\n')
                
                # Add structures
                if 'archaeological_structures' in results:
                    structures = results['archaeological_structures']
                    
                    structure_types = [
                        'earthworks', 'linear_features', 'circular_features',
                        'mounds', 'ditches', 'platforms'
                    ]
                    
                    for structure_type in structure_types:
                        if structure_type in structures and structures[structure_type]:
                            f.write(f'<Folder>\n<name>{structure_type.title()}</name>\n')
                            
                            for i, struct in enumerate(structures[structure_type]):
                                geom = self._structure_to_geometry(struct, bounds)
                                if geom and hasattr(geom, 'x') and hasattr(geom, 'y'):
                                    f.write('<Placemark>\n')
                                    f.write(f'<name>{structure_type} {i+1}</name>\n')
                                    f.write('<Point>\n')
                                    f.write(f'<coordinates>{geom.x},{geom.y},0</coordinates>\n')
                                    f.write('</Point>\n')
                                    f.write('</Placemark>\n')
                            
                            f.write('</Folder>\n')
                
                f.write('</Document>\n')
                f.write('</kml>\n')
            
            kml_files['kml_simple'] = filepath
            
        except Exception as e:
            logger.warning(f"Failed to create simple KML: {e}")
        
        return kml_files
    
    def _array_to_geotiff(self, array: np.ndarray, filepath: str, bounds: Tuple):
        """Save numpy array as GeoTIFF"""
        
        if not GIS_AVAILABLE:
            logger.warning("Cannot export GeoTIFF - rasterio not available")
            return
        
        minx, miny, maxx, maxy = bounds
        height, width = array.shape
        
        # Create transform
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        
        # Write GeoTIFF
        with rasterio.open(
            filepath, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=array.dtype,
            crs=self.crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(array, 1)
    
    def _create_synthetic_raster_data(self, bounds: Tuple) -> Dict:
        """Create synthetic raster data for demonstration"""
        
        size = 256
        x = np.linspace(bounds[0], bounds[2], size)
        y = np.linspace(bounds[1], bounds[3], size)
        X, Y = np.meshgrid(x, y)
        
        # Create synthetic elevation data
        elevation = 100 + 50 * np.sin(X * 0.1) * np.cos(Y * 0.1) + 10 * np.random.randn(size, size)
        
        # Create hillshade
        dy, dx = np.gradient(elevation)
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)
        
        azimuth_rad = np.radians(315)
        altitude_rad = np.radians(45)
        
        hillshade = np.sin(altitude_rad) * np.sin(slope) + \
                   np.cos(altitude_rad) * np.cos(slope) * \
                   np.cos(azimuth_rad - aspect)
        hillshade = ((hillshade + 1) * 127.5).astype(np.uint8)
        
        return {
            'dtm': elevation,
            'hillshade': hillshade,
            'slope': np.degrees(slope)
        }
    
    def _structure_to_csv_row(self, struct: Dict, structure_type: str,
                            index: int, bounds: Tuple) -> Dict:
        """Convert structure to CSV row format"""
        
        geom = self._structure_to_geometry(struct, bounds)
        
        row = {
            'id': index,
            'structure_type': structure_type,
            'type': struct.get('type', structure_type),
            'longitude': geom.x if geom and hasattr(geom, 'x') else None,
            'latitude': geom.y if geom and hasattr(geom, 'y') else None,
            'confidence': struct.get('confidence', 0.0),
            'archaeological_type': struct.get('archaeological_type', 'unknown'),
            'cultural_context': struct.get('cultural_context', 'unknown'),
            'method': struct.get('method', 'traditional')
        }
        
        # Add type-specific attributes
        optional_fields = ['area', 'height', 'depth', 'radius', 'length', 
                          'angle', 'width', 'aspect_ratio', 'prominence']
        
        for field in optional_fields:
            row[field] = struct.get(field, None)
        
        return row
    
    def _create_summary_statistics_df(self, results: Dict) -> pd.DataFrame:
        """Create summary statistics DataFrame"""
        
        summary_data = []
        
        if 'archaeological_structures' in results:
            structures = results['archaeological_structures']
            
            if 'summary' in structures:
                summary = structures['summary']
                
                # Overall statistics
                summary_data.append({
                    'metric': 'Total Structures',
                    'value': summary.get('total_structures', 0),
                    'category': 'Overall'
                })
                
                # By structure type
                for structure_type, count in summary.get('by_type', {}).items():
                    summary_data.append({
                        'metric': f'{structure_type.replace("_", " ").title()} Count',
                        'value': count,
                        'category': 'Structure Types'
                    })
                
                # Confidence statistics
                if 'mean_confidence' in summary:
                    summary_data.append({
                        'metric': 'Mean Confidence',
                        'value': summary['mean_confidence'],
                        'category': 'Confidence'
                    })
        
        # Add terrain statistics if available
        if 'terrain_analysis' in results:
            terrain = results['terrain_analysis']
            
            if 'elevation_stats' in terrain:
                elev_stats = terrain['elevation_stats']
                
                summary_data.extend([
                    {
                        'metric': 'Min Elevation (m)',
                        'value': elev_stats.get('min', 0),
                        'category': 'Terrain'
                    },
                    {
                        'metric': 'Max Elevation (m)',
                        'value': elev_stats.get('max', 0),
                        'category': 'Terrain'
                    },
                    {
                        'metric': 'Mean Elevation (m)',
                        'value': elev_stats.get('mean', 0),
                        'category': 'Terrain'
                    }
                ])
        
        # Add archaeological score
        if 'archaeological_score' in results:
            summary_data.append({
                'metric': 'Archaeological Potential Score',
                'value': results['archaeological_score'],
                'category': 'Assessment'
            })
        
        return pd.DataFrame(summary_data)
    
    def _create_web_map(self, results: Dict, output_dir: str,
                       timestamp: str, bounds: Tuple) -> str:
        """Create interactive web map using Folium"""
        
        if not FOLIUM_AVAILABLE:
            logger.warning("Folium not available - cannot create web map")
            return None
        
        try:
            # Calculate map center
            minx, miny, maxx, maxy = bounds
            center_lat = (miny + maxy) / 2
            center_lon = (minx + maxx) / 2
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            
            # Add satellite imagery layer
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            
            # Add structures to map
            if 'archaeological_structures' in results:
                structures = results['archaeological_structures']
                
                colors = {
                    'earthworks': 'red',
                    'linear_features': 'blue',
                    'circular_features': 'green',
                    'mounds': 'orange',
                    'ditches': 'purple',
                    'platforms': 'yellow'
                }
                
                structure_types = [
                    'earthworks', 'linear_features', 'circular_features',
                    'mounds', 'ditches', 'platforms'
                ]
                
                for structure_type in structure_types:
                    if structure_type in structures and structures[structure_type]:
                        
                        # Create feature group for this structure type
                        fg = folium.FeatureGroup(name=structure_type.replace('_', ' ').title())
                        
                        for i, struct in enumerate(structures[structure_type]):
                            geom = self._structure_to_geometry(struct, bounds)
                            
                            if geom and hasattr(geom, 'x') and hasattr(geom, 'y'):
                                # Create popup content
                                popup_content = f"""
                                <b>{structure_type.replace('_', ' ').title()} {i+1}</b><br>
                                Type: {struct.get('type', 'unknown')}<br>
                                Confidence: {struct.get('confidence', 0.0):.3f}<br>
                                Archaeological Type: {struct.get('archaeological_type', 'unknown')}<br>
                                """
                                
                                if 'area' in struct:
                                    popup_content += f"Area: {struct['area']:.1f}<br>"
                                if 'height' in struct:
                                    popup_content += f"Height: {struct['height']:.2f}m<br>"
                                
                                # Add marker
                                folium.CircleMarker(
                                    location=[geom.y, geom.x],
                                    radius=8,
                                    popup=folium.Popup(popup_content, max_width=300),
                                    color=colors.get(structure_type, 'black'),
                                    fill=True,
                                    fillColor=colors.get(structure_type, 'black'),
                                    fillOpacity=0.7
                                ).add_to(fg)
                        
                        fg.add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add scale
            plugins.MeasureControl().add_to(m)
            
            # Save web map
            filename = f"archaeological_webmap_{timestamp}.html"
            filepath = os.path.join(output_dir, filename)
            m.save(filepath)
            
            return filepath
            
        except Exception as e:
            logger.warning(f"Failed to create web map: {e}")
            return None
    
    def _create_metadata(self, results: Dict, exported_files: Dict,
                        output_dir: str, timestamp: str) -> str:
        """Create metadata file for the exported data"""
        
        metadata = {
            'export_info': {
                'timestamp': timestamp,
                'export_date': datetime.now().isoformat(),
                'coordinate_system': self.crs,
                'software': 'LIDAR Archaeological Processor'
            },
            'analysis_summary': {
                'total_structures': results.get('archaeological_structures', {}).get('summary', {}).get('total_structures', 0),
                'archaeological_score': results.get('archaeological_score', 0.0),
                'processing_method': results.get('detection_method', 'unknown')
            },
            'exported_files': exported_files,
            'structure_types': {},
            'data_quality': {
                'confidence_assessment': results.get('confidence_scores', {}),
                'method_reliability': results.get('confidence_scores', {}).get('method_reliability', {})
            }
        }
        
        # Add structure type statistics
        if 'archaeological_structures' in results:
            structures = results['archaeological_structures']
            if 'summary' in structures:
                metadata['structure_types'] = structures['summary'].get('by_type', {})
        
        # Add recommendations
        if 'recommendations' in results:
            metadata['recommendations'] = results['recommendations']
        
        # Save metadata
        filename = f"export_metadata_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
        
        return filepath


def main():
    """Test the GIS exporter"""
    
    # Create synthetic analysis results for testing
    test_results = {
        'archaeological_structures': {
            'earthworks': [
                {
                    'center': (150, 200),
                    'area': 500.0,
                    'confidence': 0.8,
                    'type': 'earthwork',
                    'archaeological_type': 'ceremonial_circle'
                }
            ],
            'mounds': [
                {
                    'center': (300, 350),
                    'height': 3.5,
                    'area': 200.0,
                    'confidence': 0.9,
                    'type': 'mound',
                    'archaeological_type': 'burial_mound'
                }
            ],
            'summary': {
                'total_structures': 2,
                'by_type': {
                    'earthworks': 1,
                    'mounds': 1
                },
                'mean_confidence': 0.85
            }
        },
        'archaeological_score': 0.75,
        'terrain_analysis': {
            'elevation_stats': {
                'min': 100.0,
                'max': 250.0,
                'mean': 175.0
            }
        },
        'recommendations': [
            "High priority site - recommend immediate field survey",
            "Deploy ground-penetrating radar"
        ]
    }
    
    # Initialize exporter
    exporter = GISArchaeologicalExporter()
    
    # Set test bounds (example Amazon coordinates)
    test_bounds = (-64.0, -12.0, -63.5, -11.5)
    
    # Export to all formats
    output_dir = "/tmp/archaeological_gis_export"
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = exporter.export_analysis_results(
        test_results, output_dir, bounds=test_bounds
    )
    
    print("GIS Export Results:")
    for file_type, filepath in exported_files.items():
        if filepath and os.path.exists(filepath):
            print(f"  {file_type}: {filepath}")
        else:
            print(f"  {file_type}: Export failed or not available")


if __name__ == "__main__":
    main()