"""
Data Preprocessing Module

Aligns, normalizes, and validates multi-source time-series data
for the multi-task model. Handles missing values and temporal alignment.
"""

from typing import Dict, Optional, Tuple, Union, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """Preprocess and align agricultural time-series data."""

    def __init__(
        self,
        sequence_length: int = 24,
        scaler_type: str = "standard",
    ):
        self.sequence_length = sequence_length
        self.scaler_type = scaler_type
        self.scalers: Dict[str, object] = {}

    def align_data(
        self,
        data: Dict[str, pd.DataFrame],
        key_columns: Tuple[str, ...] = ("region_id", "time_step"),
    ) -> pd.DataFrame:
        """
        Merge all data sources on region_id and time_step.
        Assumes all DataFrames have these keys.
        """
        merged = data["weather"].copy()
        for name, df in data.items():
            if name == "weather":
                continue
            merge_cols = [c for c in df.columns if c not in key_columns]
            merged = merged.merge(
                df,
                on=list(key_columns),
                how="outer",
                suffixes=("", f"_{name}") if any(c in merged.columns for c in merge_cols) else ("", ""),
            )
        merged = merged.sort_values(key_columns).reset_index(drop=True)
        return merged

    def handle_missing(self, df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """Fill missing values. Methods: ffill, bfill, mean, interpolate."""
        df = df.copy()
        if method == "ffill":
            df = df.groupby("region_id", group_keys=False).ffill().bfill()
        elif method == "interpolate":
            df = df.groupby("region_id", group_keys=False).apply(
                lambda g: g.interpolate(method="linear")
            ).bfill().ffill()
        elif method == "mean":
            df = df.fillna(df.mean())
        return df

    def normalize(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Apply StandardScaler or MinMaxScaler to numeric columns."""
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["region_id", "time_step"]
        columns = [c for c in columns if c in df.columns and c not in exclude]

        scaler_cls = StandardScaler if self.scaler_type == "standard" else MinMaxScaler
        for col in columns:
            if fit:
                self.scalers[col] = scaler_cls()
                df[col] = self.scalers[col].fit_transform(df[[col]])
            else:
                if col in self.scalers:
                    df[col] = self.scalers[col].transform(df[[col]])
        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_cols: Dict[str, Union[str, List[str]]],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sliding-window sequences for time-series prediction.
        target_cols: task -> column name(s). Use list for multi-output (e.g., soil NPK+OC).
        Returns: (X, targets_dict) where X is (n_samples, seq_len, n_features).
        """
        regions = df["region_id"].unique()
        X_list = []
        targets_list = {k: [] for k in target_cols}

        for region in regions:
            subset = df[df["region_id"] == region].sort_values("time_step").reset_index(drop=True)
            if len(subset) < self.sequence_length + 1:
                continue
            features = subset[feature_cols].values
            for i in range(len(subset) - self.sequence_length):
                X_list.append(features[i : i + self.sequence_length])
                row = subset.iloc[i + self.sequence_length]
                for task, col in target_cols.items():
                    if isinstance(col, list):
                        vals = row[col].values
                        targets_list[task].append(vals)
                    else:
                        targets_list[task].append(row[col])

        X = np.array(X_list) if X_list else np.array([]).reshape(0, self.sequence_length, len(feature_cols))
        targets = {k: np.array(v) for k, v in targets_list.items()}
        return X, targets

    def preprocess_pipeline(
        self,
        data: Dict[str, pd.DataFrame],
        feature_cols: Optional[list] = None,
        target_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, list, Dict[str, str]]:
        """
        Full preprocessing pipeline: align, handle missing, normalize.
        Returns aligned dataframe, feature columns, and target mapping.
        """
        merged = self.align_data(data)
        merged = self.handle_missing(merged)

        if target_mapping is None:
            target_mapping = {
                "crop": "ndvi",
                "soil": "organic_carbon",
                "water": "groundwater_level",
            }

        if feature_cols is None:
            exclude = ["region_id", "time_step"] + list(target_mapping.values())
            feature_cols = [c for c in merged.select_dtypes(include=[np.number]).columns if c not in exclude]

        merged = self.normalize(merged, columns=feature_cols, fit=True)
        # Re-normalize targets separately if needed
        for task, col in target_mapping.items():
            if col in merged.columns and col not in self.scalers:
                self.scalers[col] = StandardScaler()
                merged[col] = self.scalers[col].fit_transform(merged[[col]])

        return merged, feature_cols, target_mapping
