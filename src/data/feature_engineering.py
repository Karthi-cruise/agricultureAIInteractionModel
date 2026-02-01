"""
Feature Engineering Module

Creates domain-informed features for crop-soil-water modeling:
- Lagged variables
- Rolling statistics
- Domain-specific ratios (e.g., irrigation/rainfall)
"""

from typing import List, Optional
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Generate domain-aware features for multi-task prediction."""

    def __init__(
        self,
        lag_periods: List[int] = [1, 3, 6, 12],
        rolling_windows: List[int] = [3, 6, 12],
    ):
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows

    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        group_by: str = "region_id",
    ) -> pd.DataFrame:
        """Add lagged values for time-series modeling."""
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            for lag in self.lag_periods:
                df[f"{col}_lag_{lag}"] = df.groupby(group_by)[col].shift(lag)
        return df

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        group_by: str = "region_id",
    ) -> pd.DataFrame:
        """Add rolling mean and std for trend capture."""
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            for w in self.rolling_windows:
                grouped = df.groupby(group_by)[col]
                df[f"{col}_roll_mean_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
                df[f"{col}_roll_std_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        return df

    def add_domain_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-specific ratio features."""
        df = df.copy()

        # Irrigation vs rainfall (groundwater stress indicator)
        if "irrigation_mm" in df.columns and "rainfall_mm" in df.columns:
            df["irrigation_rainfall_ratio"] = df["irrigation_mm"] / (df["rainfall_mm"] + 1e-6)

        # Water demand vs supply (ET vs rainfall)
        if "evapotranspiration" in df.columns and "rainfall_mm" in df.columns:
            df["water_deficit"] = df["evapotranspiration"] - df["rainfall_mm"] / 30  # rough monthly ET

        # Soil moisture stress (low moisture = stress)
        if "soil_moisture" in df.columns:
            df["soil_moisture_stress"] = 1 - df["soil_moisture"]

        return df

    def add_seasonal_features(self, df: pd.DataFrame, time_col: str = "time_step") -> pd.DataFrame:
        """Add cyclical seasonal encoding (sin/cos)."""
        df = df.copy()
        if time_col not in df.columns:
            return df
        # Assume 12 steps per year
        period = 12
        df["season_sin"] = np.sin(2 * np.pi * df[time_col] / period)
        df["season_cos"] = np.cos(2 * np.pi * df[time_col] / period)
        return df

    def transform(
        self,
        df: pd.DataFrame,
        lag_columns: Optional[List[str]] = None,
        rolling_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["region_id", "time_step"]
        base_cols = [c for c in numeric_cols if c not in exclude]

        lag_cols = lag_columns or base_cols[:8]
        roll_cols = rolling_columns or base_cols[:6]

        df = self.add_lag_features(df, lag_cols[:5])
        df = self.add_rolling_features(df, roll_cols[:4])
        df = self.add_domain_ratios(df)
        df = self.add_seasonal_features(df)

        return df.fillna(0)
