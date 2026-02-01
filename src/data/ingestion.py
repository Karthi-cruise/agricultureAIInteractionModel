"""
Data Ingestion Module

Loads and unifies data from multiple sources: satellite, soil, groundwater, weather.
Supports CSV, Parquet, and synthetic data generation for development.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


class DataIngestion:
    """Ingest and unify multi-source agricultural data."""

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.raw_path = self.data_root / "raw"
        self.processed_path = self.data_root / "processed"

    def load_satellite_data(
        self,
        path: Optional[str] = None,
        use_synthetic: bool = True,
    ) -> pd.DataFrame:
        """Load satellite NDVI/EVI data."""
        if path and (self.raw_path / "satellite" / path).exists():
            return pd.read_csv(self.raw_path / "satellite" / path)
        if use_synthetic:
            return self._generate_synthetic_satellite()
        raise FileNotFoundError("No satellite data found. Set use_synthetic=True for demo.")

    def load_soil_data(
        self,
        path: Optional[str] = None,
        use_synthetic: bool = True,
    ) -> pd.DataFrame:
        """Load soil nutrient (NPK, OC) and moisture data."""
        if path and (self.raw_path / "soil" / path).exists():
            return pd.read_csv(self.raw_path / "soil" / path)
        if use_synthetic:
            return self._generate_synthetic_soil()
        raise FileNotFoundError("No soil data found. Set use_synthetic=True for demo.")

    def load_groundwater_data(
        self,
        path: Optional[str] = None,
        use_synthetic: bool = True,
    ) -> pd.DataFrame:
        """Load groundwater level and irrigation data."""
        if path and (self.raw_path / "groundwater" / path).exists():
            return pd.read_csv(self.raw_path / "groundwater" / path)
        if use_synthetic:
            return self._generate_synthetic_groundwater()
        raise FileNotFoundError("No groundwater data found. Set use_synthetic=True for demo.")

    def load_weather_data(
        self,
        path: Optional[str] = None,
        use_synthetic: bool = True,
    ) -> pd.DataFrame:
        """Load weather data (temp, rainfall, ET)."""
        if path and (self.raw_path / "weather" / path).exists():
            return pd.read_csv(self.raw_path / "weather" / path)
        if use_synthetic:
            return self._generate_synthetic_weather()
        raise FileNotFoundError("No weather data found. Set use_synthetic=True for demo.")

    def load_all(
        self,
        use_synthetic: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load all data sources. Returns dict with keys: satellite, soil, groundwater, weather."""
        return {
            "satellite": self.load_satellite_data(use_synthetic=use_synthetic),
            "soil": self.load_soil_data(use_synthetic=use_synthetic),
            "groundwater": self.load_groundwater_data(use_synthetic=use_synthetic),
            "weather": self.load_weather_data(use_synthetic=use_synthetic),
        }

    def _generate_synthetic_satellite(self, n_regions: int = 10, n_steps: int = 120) -> pd.DataFrame:
        """Generate synthetic NDVI-like time series for development."""
        np.random.seed(42)
        data = []
        for r in range(n_regions):
            base = 0.4 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, n_steps))
            noise = np.random.randn(n_steps) * 0.05
            for t in range(n_steps):
                data.append({
                    "region_id": r,
                    "time_step": t,
                    "ndvi": np.clip(base[t] + noise[t], 0.1, 0.95),
                    "evi": np.clip(base[t] * 1.1 + noise[t] * 0.5, 0.1, 1.0),
                })
        return pd.DataFrame(data)

    def _generate_synthetic_soil(self, n_regions: int = 10, n_steps: int = 120) -> pd.DataFrame:
        """Generate synthetic soil nutrient data."""
        np.random.seed(43)
        data = []
        for r in range(n_regions):
            base_n = 0.5 + 0.2 * np.sin(np.linspace(0, 2 * np.pi, n_steps))
            base_p = 0.3 + 0.15 * np.sin(np.linspace(0.1, 2.1 * np.pi, n_steps))
            base_k = 0.4 + 0.2 * np.sin(np.linspace(0.2, 2.2 * np.pi, n_steps))
            base_oc = 1.5 + 0.5 * np.sin(np.linspace(0, np.pi, n_steps))
            for t in range(n_steps):
                data.append({
                    "region_id": r,
                    "time_step": t,
                    "nitrogen": np.clip(base_n[t] + np.random.randn() * 0.05, 0.1, 1.0),
                    "phosphorus": np.clip(base_p[t] + np.random.randn() * 0.03, 0.05, 0.8),
                    "potassium": np.clip(base_k[t] + np.random.randn() * 0.04, 0.1, 0.9),
                    "organic_carbon": np.clip(base_oc[t] + np.random.randn() * 0.1, 0.5, 3.0),
                    "soil_moisture": np.clip(0.3 + 0.3 * np.sin(t / 12) + np.random.randn() * 0.05, 0.1, 0.6),
                })
        return pd.DataFrame(data)

    def _generate_synthetic_groundwater(
        self,
        n_regions: int = 10,
        n_steps: int = 120,
    ) -> pd.DataFrame:
        """Generate synthetic groundwater level and irrigation data."""
        np.random.seed(44)
        data = []
        for r in range(n_regions):
            trend = -0.002 * np.arange(n_steps)  # depletion trend
            seasonal = 0.5 * np.sin(np.linspace(0, 4 * np.pi, n_steps))
            gw = 50 + trend + seasonal + np.cumsum(np.random.randn(n_steps) * 0.1)
            irrigation = 20 + 10 * np.sin(np.linspace(0, 4 * np.pi, n_steps)) + np.random.randn(n_steps) * 2
            for t in range(n_steps):
                data.append({
                    "region_id": r,
                    "time_step": t,
                    "groundwater_level": np.clip(gw[t], 30, 70),
                    "irrigation_mm": np.clip(irrigation[t], 0, 50),
                })
        return pd.DataFrame(data)

    def _generate_synthetic_weather(
        self,
        n_regions: int = 10,
        n_steps: int = 120,
    ) -> pd.DataFrame:
        """Generate synthetic weather data."""
        np.random.seed(45)
        data = []
        for r in range(n_regions):
            temp = 25 + 10 * np.sin(np.linspace(0, 4 * np.pi, n_steps)) + np.random.randn(n_steps) * 2
            rainfall = np.maximum(0, 50 + 30 * np.sin(np.linspace(0.5, 4.5 * np.pi, n_steps)) + np.random.randn(n_steps) * 10)
            et = 3 + 2 * np.sin(np.linspace(0, 4 * np.pi, n_steps)) + np.random.randn(n_steps) * 0.3
            for t in range(n_steps):
                data.append({
                    "region_id": r,
                    "time_step": t,
                    "temperature": np.clip(temp[t], 10, 40),
                    "rainfall_mm": rainfall[t],
                    "evapotranspiration": np.clip(et[t], 1, 6),
                })
        return pd.DataFrame(data)
