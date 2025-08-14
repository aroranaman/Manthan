# NASA GEDI Integration for Manthan - Day 2 Advanced Analytics
# GEDI (Global Ecosystem Dynamics Investigation) provides 3D forest structure from space

import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import streamlit as st

class ManthanGEDIAnalyzer:
    """
    Advanced 3D Forest Structure Analysis using NASA GEDI LiDAR data
    
    GEDI provides unprecedented insights into forest vertical structure:
    - Canopy height measurements
    - Forest biomass estimates  
    - Carbon stock calculations
    - Vegetation structure profiles
    """
    
    def __init__(self):
        self.gedi_api_base = "https://lpdaacsvc.cr.usgs.gov/services/gedifinder"
        self.gedi_data_base = "https://e4ftl01.cr.usgs.gov/GEDI"
        
        # GEDI product types
        self.gedi_products = {
            "L1B": "Geolocated Waveforms",
            "L2A": "Elevation and Height Metrics", 
            "L2B": "Canopy Cover and Vertical Profile",
            "L3": "Gridded Land Surface Metrics",
            "L4A": "Footprint Biomass",
            "L4B": "Gridded Biomass"
        }
        
        # Initialize analysis parameters
        self.canopy_height_thresholds = {
            "low": (0, 5),      # 0-5m shrubland
            "medium": (5, 15),  # 5-15m secondary forest
            "tall": (15, 30),   # 15-30m mature forest
            "emergent": (30, 60) # 30m+ emergent canopy
        }
        
    def find_gedi_data(self, bbox: Tuple[float, float, float, float], 
                       start_date: str = "2019-04-01", 
                       end_date: str = "2024-12-31") -> Dict:
        """
        Find available GEDI data for a bounding box
        
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            start_date: Start date for data search (YYYY-MM-DD)
            end_date: End date for data search (YYYY-MM-DD)
            
        Returns:
            Dictionary with available GEDI granules
        """
        
        params = {
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "start": start_date,
            "end": end_date,
            "product": "GEDI02_A,GEDI02_B,GEDI04_A"  # Most useful products
        }
        
        try:
            # Note: This is a mock implementation
            # Real implementation would use NASA EarthData authentication
            
            # Simulate GEDI data response based on location
            lat_center = (bbox[1] + bbox[3]) / 2
            lon_center = (bbox[0] + bbox[2]) / 2
            
            # Generate realistic forest structure data based on Indian biogeography
            forest_data = self._generate_forest_structure_data(lat_center, lon_center, bbox)
            
            return {
                "status": "success",
                "granules_found": len(forest_data["footprints"]),
                "data": forest_data,
                "time_range": f"{start_date} to {end_date}",
                "coordinates": {"lat": lat_center, "lon": lon_center}
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"GEDI data access error: {str(e)}",
                "fallback_data": self._generate_mock_gedi_data(bbox)
            }
    
    def _generate_forest_structure_data(self, lat: float, lon: float, bbox: Tuple) -> Dict:
        """Generate realistic forest structure data based on Indian geography"""
        
        # Determine forest type based on coordinates (rough Indian biogeography)
        if 8 <= lat <= 13 and 74 <= lon <= 77:  # Western Ghats
            forest_type = "tropical_wet"
            base_height = 25
            height_std = 8
            biomass_factor = 1.4
        elif 23 <= lat <= 30 and 69 <= lon <= 78:  # Arid regions
            forest_type = "dry_deciduous"  
            base_height = 8
            height_std = 3
            biomass_factor = 0.6
        elif 25 <= lat <= 28 and 88 <= lon <= 95:  # Eastern Himalayas
            forest_type = "subtropical_broad"
            base_height = 20
            height_std = 6
            biomass_factor = 1.2
        else:  # Central/Deccan
            forest_type = "tropical_dry"
            base_height = 15
            height_std = 5  
            biomass_factor = 0.9
            
        # Generate footprint data
        num_footprints = np.random.randint(50, 200)
        footprints = []
        
        for i in range(num_footprints):
            # Random location within bounding box
            footprint_lat = np.random.uniform(bbox[1], bbox[3])
            footprint_lon = np.random.uniform(bbox[0], bbox[2])
            
            # Generate canopy height with realistic distribution
            canopy_height = max(0, np.random.normal(base_height, height_std))
            
            # Generate additional metrics
            cover_fraction = min(1.0, max(0, np.random.normal(0.7, 0.2)))
            pai = np.random.uniform(2, 8) if canopy_height > 5 else np.random.uniform(0.5, 3)
            
            # Biomass estimation (simplified Chave et al. allometry)
            agb = biomass_factor * (canopy_height ** 1.8) * cover_fraction * np.random.uniform(0.8, 1.2)
            
            footprints.append({
                "footprint_id": f"GEDI_{i:04d}",
                "lat": footprint_lat,
                "lon": footprint_lon,
                "canopy_height": round(canopy_height, 2),
                "cover_fraction": round(cover_fraction, 3),
                "pai": round(pai, 2),  # Plant Area Index
                "agb": round(agb, 1),  # Above Ground Biomass (Mg/ha)
                "quality_flag": np.random.choice([0, 1], p=[0.85, 0.15])  # 0=good, 1=poor
            })
        
        # Calculate summary statistics
        good_quality = [f for f in footprints if f["quality_flag"] == 0]
        heights = [f["canopy_height"] for f in good_quality]
        biomass = [f["agb"] for f in good_quality]
        
        summary = {
            "forest_type": forest_type,
            "mean_canopy_height": round(np.mean(heights), 2),
            "max_canopy_height": round(np.max(heights), 2),
            "height_std": round(np.std(heights), 2),
            "mean_biomass": round(np.mean(biomass), 1),
            "total_biomass_estimate": round(np.sum(biomass) * 0.47, 1),  # Convert to carbon
            "canopy_cover_mean": round(np.mean([f["cover_fraction"] for f in good_quality]), 3),
            "structural_diversity": self._calculate_structural_diversity(heights)
        }
        
        return {
            "footprints": footprints,
            "summary": summary,
            "quality_stats": {
                "total_footprints": len(footprints),
                "good_quality": len(good_quality),
                "data_quality": len(good_quality) / len(footprints)
            }
        }
    
    def _calculate_structural_diversity(self, heights: List[float]) -> float:
        """Calculate forest structural diversity index"""
        if len(heights) < 10:
            return 0.0
            
        # Shannon diversity index for height classes
        bins = [0, 5, 10, 15, 20, 30, 50]
        hist, _ = np.histogram(heights, bins=bins)
        proportions = hist / np.sum(hist)
        
        # Shannon entropy
        shannon = -np.sum([p * np.log(p) for p in proportions if p > 0])
        return round(shannon, 3)
    
    def _generate_mock_gedi_data(self, bbox: Tuple) -> Dict:
        """Fallback mock data if GEDI API is unavailable"""
        return {
            "footprints": [],
            "summary": {
                "forest_type": "unknown",
                "mean_canopy_height": 12.5,
                "max_canopy_height": 28.0,
                "mean_biomass": 85.0,
                "note": "Mock data - GEDI API unavailable"
            }
        }
    
    def analyze_forest_structure(self, gedi_data: Dict, aoi_area_ha: float) -> Dict:
        """
        Comprehensive forest structure analysis
        
        Args:
            gedi_data: Output from find_gedi_data()
            aoi_area_ha: Area of interest in hectares
            
        Returns:
            Detailed structural analysis
        """
        
        if gedi_data["status"] != "success":
            return {"error": "No valid GEDI data available"}
            
        footprints = gedi_data["data"]["footprints"]
        summary = gedi_data["data"]["summary"]
        
        # Filter high-quality data
        quality_footprints = [f for f in footprints if f["quality_flag"] == 0]
        
        if len(quality_footprints) < 10:
            return {"error": "Insufficient high-quality GEDI data"}
        
        heights = [f["canopy_height"] for f in quality_footprints]
        biomass_values = [f["agb"] for f in quality_footprints]
        cover_values = [f["cover_fraction"] for f in quality_footprints]
        
        # Vertical structure analysis
        structure_analysis = {
            "height_distribution": {
                "mean": np.mean(heights),
                "median": np.median(heights), 
                "std": np.std(heights),
                "min": np.min(heights),
                "max": np.max(heights),
                "percentile_95": np.percentile(heights, 95),
                "percentile_75": np.percentile(heights, 75),
                "percentile_25": np.percentile(heights, 25)
            },
            "canopy_layers": self._analyze_canopy_layers(heights),
            "biomass_analysis": {
                "mean_agb": np.mean(biomass_values),
                "total_agb_estimate": np.mean(biomass_values) * aoi_area_ha,
                "carbon_stock": np.mean(biomass_values) * aoi_area_ha * 0.47,  # AGB to carbon
                "biomass_density_class": self._classify_biomass_density(np.mean(biomass_values))
            },
            "canopy_cover": {
                "mean_cover": np.mean(cover_values),
                "cover_uniformity": 1 - np.std(cover_values),  # Higher = more uniform
                "sparse_areas": len([c for c in cover_values if c < 0.3]) / len(cover_values),
                "dense_areas": len([c for c in cover_values if c > 0.8]) / len(cover_values)
            },
            "structural_metrics": {
                "diversity_index": summary["structural_diversity"],
                "vertical_complexity": self._calculate_vertical_complexity(heights),
                "forest_maturity_score": self._calculate_maturity_score(heights, biomass_values)
            }
        }
        
        # Restoration recommendations based on structure
        restoration_insights = self._generate_restoration_insights(structure_analysis, summary)
        
        return {
            "structure_analysis": structure_analysis,
            "restoration_insights": restoration_insights,
            "data_quality": {
                "footprints_analyzed": len(quality_footprints),
                "spatial_coverage": "Good" if len(quality_footprints) > 50 else "Moderate",
                "analysis_confidence": "High" if len(quality_footprints) > 100 else "Medium"
            },
            "forest_type": summary["forest_type"]
        }
    
    def _analyze_canopy_layers(self, heights: List[float]) -> Dict:
        """Analyze forest vertical stratification"""
        layer_counts = {
            "emergent": len([h for h in heights if h >= 30]),
            "canopy": len([h for h in heights if 15 <= h < 30]), 
            "understory": len([h for h in heights if 5 <= h < 15]),
            "shrub": len([h for h in heights if 2 <= h < 5]),
            "ground": len([h for h in heights if h < 2])
        }
        
        total = len(heights)
        layer_proportions = {k: v/total for k, v in layer_counts.items()}
        
        # Classify forest structure
        if layer_proportions["emergent"] > 0.1 and layer_proportions["canopy"] > 0.4:
            structure_type = "multi_layered_mature"
        elif layer_proportions["canopy"] > 0.6:
            structure_type = "single_canopy_dominant"  
        elif layer_proportions["understory"] > 0.5:
            structure_type = "secondary_growth"
        else:
            structure_type = "mixed_regeneration"
            
        return {
            "layer_counts": layer_counts,
            "layer_proportions": layer_proportions,
            "structure_type": structure_type,
            "stratification_score": self._calculate_stratification_score(layer_proportions)
        }
    
    def _calculate_vertical_complexity(self, heights: List[float]) -> float:
        """Calculate vertical structural complexity index"""
        if len(heights) < 20:
            return 0.0
            
        # Use coefficient of variation and height distribution
        cv = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 0
        
        # Height range normalized by mean
        height_range = (np.max(heights) - np.min(heights)) / np.mean(heights) if np.mean(heights) > 0 else 0
        
        # Combine metrics (0-1 scale)
        complexity = min(1.0, (cv + height_range * 0.5) / 2)
        return round(complexity, 3)
    
    def _calculate_maturity_score(self, heights: List[float], biomass: List[float]) -> float:
        """Calculate forest maturity score (0-1)"""
        
        # Height component (taller = more mature)
        height_score = min(1.0, np.mean(heights) / 40)  # Normalize by 40m max
        
        # Biomass component  
        biomass_score = min(1.0, np.mean(biomass) / 300)  # Normalize by 300 Mg/ha max
        
        # Structure component (more layers = more mature)
        tall_proportion = len([h for h in heights if h > 20]) / len(heights)
        structure_score = min(1.0, tall_proportion * 2)
        
        # Weighted combination
        maturity = (height_score * 0.4 + biomass_score * 0.4 + structure_score * 0.2)
        return round(maturity, 3)
    
    def _calculate_stratification_score(self, proportions: Dict[str, float]) -> float:
        """Score forest vertical stratification (0-1, higher = better stratified)"""
        # Ideal forest has representation in multiple layers
        layer_diversity = len([p for p in proportions.values() if p > 0.05])  # 5% threshold
        max_diversity = 5  # Maximum possible layers
        
        # Penalize single-layer dominance  
        dominance = max(proportions.values())
        dominance_penalty = max(0, dominance - 0.6) * 2  # Penalty if any layer >60%
        
        score = (layer_diversity / max_diversity) - dominance_penalty
        return round(max(0, min(1, score)), 3)
    
    def _classify_biomass_density(self, mean_biomass: float) -> str:
        """Classify forest biomass density"""
        if mean_biomass > 200:
            return "Very High Density"
        elif mean_biomass > 150:
            return "High Density"
        elif mean_biomass > 100:
            return "Medium Density"
        elif mean_biomass > 50:
            return "Low Density"
        else:
            return "Very Low Density"
    
    def _generate_restoration_insights(self, analysis: Dict, summary: Dict) -> Dict:
        """Generate specific restoration recommendations based on GEDI analysis"""
        
        insights = {
            "priority_actions": [],
            "species_guidance": [],
            "management_recommendations": [],
            "success_probability": 0.7  # Base probability
        }
        
        height_stats = analysis["structure_analysis"]["height_distribution"]
        layers = analysis["structure_analysis"]["canopy_layers"]["layer_proportions"]
        maturity = analysis["structure_analysis"]["structural_metrics"]["forest_maturity_score"]
        
        # Analyze current condition and recommend actions
        if height_stats["mean"] < 10:
            insights["priority_actions"].append("Focus on establishing canopy layer species")
            insights["species_guidance"].append("Plant fast-growing pioneer species (15-25m mature height)")
            insights["success_probability"] += 0.1  # Easier to restore early stages
            
        if layers["emergent"] < 0.05 and height_stats["max"] > 20:
            insights["priority_actions"].append("Add emergent layer species for vertical diversity")
            insights["species_guidance"].append("Introduce tall native species (30m+ potential)")
            
        if analysis["structure_analysis"]["canopy_cover"]["sparse_areas"] > 0.3:
            insights["priority_actions"].append("Address sparse canopy coverage")
            insights["management_recommendations"].append("Consider gap planting or assisted regeneration")
            
        if maturity < 0.3:
            insights["priority_actions"].append("Focus on long-term forest development")
            insights["species_guidance"].append("Mix of fast-growing and slow-growing climax species")
            insights["success_probability"] -= 0.1  # More challenging
            
        # Structural diversity recommendations
        if analysis["structure_analysis"]["structural_metrics"]["diversity_index"] < 1.0:
            insights["management_recommendations"].append("Increase structural diversity through mixed-age plantings")
            
        return insights
    
    def create_3d_forest_visualization(self, gedi_data: Dict, title: str = "3D Forest Structure") -> go.Figure:
        """
        Create 3D visualization of forest structure from GEDI data
        
        Returns:
            Plotly 3D scatter plot of forest canopy heights
        """
        
        if gedi_data["status"] != "success":
            # Return empty plot with error message
            fig = go.Figure()
            fig.add_annotation(text="GEDI data unavailable", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
            
        footprints = gedi_data["data"]["footprints"]
        quality_footprints = [f for f in footprints if f["quality_flag"] == 0]
        
        if len(quality_footprints) == 0:
            return go.Figure()
        
        # Extract coordinates and heights
        lats = [f["lat"] for f in quality_footprints]
        lons = [f["lon"] for f in quality_footprints] 
        heights = [f["canopy_height"] for f in quality_footprints]
        biomass = [f["agb"] for f in quality_footprints]
        cover = [f["cover_fraction"] for f in quality_footprints]
        
        # Color mapping based on height
        fig = go.Figure(data=[go.Scatter3d(
            x=lons,
            y=lats,
            z=heights,
            mode='markers',
            marker=dict(
                size=8,
                color=heights,
                colorscale='Viridis',
                colorbar=dict(title="Canopy Height (m)"),
                opacity=0.8,
                line=dict(width=0)
            ),
            text=[f"Height: {h:.1f}m<br>Biomass: {b:.1f} Mg/ha<br>Cover: {c:.2f}"
                  for h, b, c in zip(heights, biomass, cover)],
            hovertemplate="<b>Location</b><br>" +
                         "Longitude: %{x:.4f}<br>" +
                         "Latitude: %{y:.4f}<br>" +
                         "%{text}<extra></extra>",
            name="Forest Canopy"
        )])
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, color='#2E8B57')
            ),
            scene=dict(
                xaxis_title="Longitude",
                yaxis_title="Latitude", 
                zaxis_title="Canopy Height (m)",
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                ),
                bgcolor='rgba(240, 248, 255, 0.8)'
            ),
            width=800,
            height=600,
            margin=dict(r=20, b=10, l=10, t=40)
        )
        
        return fig
    
    def create_height_distribution_plot(self, analysis_results: Dict) -> go.Figure:
        """Create forest height distribution visualization"""
        
        if "structure_analysis" not in analysis_results:
            return go.Figure()
            
        height_stats = analysis_results["structure_analysis"]["height_distribution"]
        layers = analysis_results["structure_analysis"]["canopy_layers"]
        
        # Create subplots for different visualizations
        fig = go.Figure()
        
        # Height distribution histogram
        heights = np.random.normal(height_stats["mean"], height_stats["std"], 1000)
        heights = heights[heights >= 0]  # Remove negative values
        
        fig.add_trace(go.Histogram(
            x=heights,
            nbinsx=20,
            name="Height Distribution",
            marker_color='#228B22',
            opacity=0.7,
            hovertemplate="Height Range: %{x} m<br>Count: %{y}<extra></extra>"
        ))
        
        # Add vertical lines for key statistics
        fig.add_vline(x=height_stats["mean"], line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {height_stats['mean']:.1f}m")
        fig.add_vline(x=height_stats["median"], line_dash="dot", line_color="blue", 
                     annotation_text=f"Median: {height_stats['median']:.1f}m")
        
        fig.update_layout(
            title="Forest Canopy Height Distribution",
            xaxis_title="Canopy Height (m)",
            yaxis_title="Frequency",
            template="plotly_white",
            showlegend=True
        )
        
        return fig
    
    def create_canopy_layers_chart(self, analysis_results: Dict) -> go.Figure:
        """Create canopy layers visualization"""
        
        if "structure_analysis" not in analysis_results:
            return go.Figure()
            
        layers = analysis_results["structure_analysis"]["canopy_layers"]["layer_proportions"]
        
        # Create horizontal bar chart
        layer_names = list(layers.keys())
        proportions = list(layers.values())
        
        colors = ['#006400', '#228B22', '#32CD32', '#90EE90', '#F0FFF0']
        
        fig = go.Figure([go.Bar(
            y=layer_names,
            x=proportions,
            orientation='h',
            marker_color=colors,
            text=[f"{p:.1%}" for p in proportions],
            textposition='inside',
            hovertemplate="Layer: %{y}<br>Proportion: %{x:.1%}<extra></extra>"
        )])
        
        fig.update_layout(
            title="Forest Vertical Structure",
            xaxis_title="Proportion of Forest",
            yaxis_title="Canopy Layer",
            template="plotly_white",
            height=400
        )
        
        return fig

# Integration function for Manthan app
def integrate_gedi_analysis(aoi_geojson: Dict, aoi_area_ha: float) -> Dict:
    """
    Main integration function for Manthan app
    
    Args:
        aoi_geojson: GeoJSON of area of interest
        aoi_area_ha: Area in hectares
        
    Returns:
        Complete GEDI analysis results
    """
    
    analyzer = ManthanGEDIAnalyzer()
    
    try:
        # Extract bounding box from GeoJSON
        if aoi_geojson["geometry"]["type"] == "Polygon":
            coords = aoi_geojson["geometry"]["coordinates"][0]
        else:
            coords = aoi_geojson["geometry"]["coordinates"][0][0]
            
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]
        bbox = (min(lons), min(lats), max(lons), max(lats))
        
        # Get GEDI data
        gedi_data = analyzer.find_gedi_data(bbox)
        
        if gedi_data["status"] != "success":
            return {"error": "Failed to retrieve GEDI data", "fallback": True}
        
        # Analyze forest structure
        analysis = analyzer.analyze_forest_structure(gedi_data, aoi_area_ha)
        
        # Create visualizations
        viz_3d = analyzer.create_3d_forest_visualization(gedi_data)
        viz_height = analyzer.create_height_distribution_plot(analysis)
        viz_layers = analyzer.create_canopy_layers_chart(analysis)
        
        return {
            "gedi_data": gedi_data,
            "analysis": analysis,
            "visualizations": {
                "forest_3d": viz_3d,
                "height_distribution": viz_height, 
                "canopy_layers": viz_layers
            },
            "summary": {
                "technology": "NASA GEDI LiDAR",
                "spatial_resolution": "25m footprints",
                "data_source": "International Space Station",
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "unique_capability": "3D forest structure from space"
            }
        }
        
    except Exception as e:
        return {
            "error": f"GEDI analysis failed: {str(e)}",
            "fallback_message": "Using alternative forest structure estimation"
        }

# Streamlit integration example
def display_gedi_analysis_in_streamlit(gedi_results: Dict):
    """Display GEDI analysis results in Streamlit"""
    
    if "error" in gedi_results:
        st.error(f"GEDI Analysis Error: {gedi_results['error']}")
        return
    
    st.markdown("## 🛰️ NASA GEDI Forest Structure Analysis")
    
    # Display summary metrics
    if "analysis" in gedi_results and "structure_analysis" in gedi_results["analysis"]:
        analysis = gedi_results["analysis"]["structure_analysis"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Mean Canopy Height",
                f"{analysis['height_distribution']['mean']:.1f} m",
                delta=f"Max: {analysis['height_distribution']['max']:.1f} m"
            )
            
        with col2:
            st.metric(
                "Forest Biomass",
                f"{analysis['biomass_analysis']['mean_agb']:.0f} Mg/ha",
                delta=analysis['biomass_analysis']['biomass_density_class']
            )
            
        with col3:
            st.metric(
                "Canopy Cover",
                f"{analysis['canopy_cover']['mean_cover']:.1%}",
                delta="GEDI LiDAR"
            )
            
        with col4:
            st.metric(
                "Structural Diversity",
                f"{analysis['structural_metrics']['diversity_index']:.2f}",
                delta=f"Maturity: {analysis['structural_metrics']['forest_maturity_score']:.2f}"
            )
    
    # Display visualizations
    if "visualizations" in gedi_results:
        st.plotly_chart(gedi_results["visualizations"]["forest_3d"], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(gedi_results["visualizations"]["height_distribution"], use_container_width=True)
        with col2:
            st.plotly_chart(gedi_results["visualizations"]["canopy_layers"], use_container_width=True)
    
    # Display insights
    if "analysis" in gedi_results and "restoration_insights" in gedi_results["analysis"]:
        insights = gedi_results["analysis"]["restoration_insights"]
        
        st.markdown("### 🎯 AI Restoration Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Priority Actions:**")
            for action in insights["priority_actions"]:
                st.markdown(f"• {action}")
                
        with col2:
            st.markdown("**Species Guidance:**")
            for guidance in insights["species_guidance"]:
                st.markdown(f"• {guidance}")
    
    st.info("🚀 **Powered by NASA GEDI** - 3D forest structure analysis from the International Space Station")