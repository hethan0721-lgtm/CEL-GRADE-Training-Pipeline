#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading module responsible for reading, merging, and performing
initial cleansing of the raw datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from .config import Config


def _read_table(file_path):
    """
    Read a tabular data file, dispatching on file extension.

    ``.xlsx``/``.xls`` -> ``pandas.read_excel`` (unchanged behaviour).
    ``.csv`` -> ``pandas.read_csv`` (``encoding='utf-8-sig'`` so files with
    or without a UTF-8 BOM -- e.g. saved by Excel on Windows -- both read
    correctly, which matters for the Chinese column names used throughout
    this pipeline).

    This only changes how bytes are read into a DataFrame; every later
    preprocessing/training step operates on that DataFrame identically
    regardless of which format it came from.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(file_path, encoding='utf-8-sig')
    elif suffix in ('.xlsx', '.xls'):
        return pd.read_excel(file_path)
    else:
        raise ValueError(
            f"Unsupported data file extension '{suffix}' for {file_path}; "
            "expected .csv, .xlsx, or .xls"
        )


class DataLoader:
    """Utility class for loading and cleansing raw data."""

    def __init__(self, config=None):
        """
        Initialize the data loader.

        Args:
            config: Optional configuration object. Defaults to ``Config``.
        """
        self.config = config if config is not None else Config()
        self.data = None
        self.train_data = None  # 训练数据
        self.val_data = None    # 验证/测试数据
        self.raw_data_info = {}
        self.preserved_data = None  # Store preserved columns (姓名, 身份证号, 省份)
        self.preserved_train_data = None  # 训练数据的保留列
        self.preserved_val_data = None    # 验证数据的保留列

    def load_data(self, verbose=True):
        """
        Load every configured data file and concatenate them into a single frame.

        Args:
            verbose: Whether to print detailed progress information.

        Returns:
            DataFrame: The merged dataset.
        """
        if verbose:
            print("\n" + "="*70)
            print(" " * 25 + "Starting data ingestion")
            print("="*70)

        all_dataframes = []

        # Resolve absolute paths on demand
        paths = self.config.get_absolute_paths()
        data_files = paths['data_files']

        for i, file_path in enumerate(data_files, 1):
            if verbose:
                print(f"\nReading file {i}/{len(data_files)}: {Path(file_path).name}")

            try:
                # Read the data file (.csv / .xlsx / .xls)
                df = _read_table(file_path)

                # Standardize column names (handle variations)
                # 文本6 has 'A姓名' instead of '姓名'
                if 'A姓名' in df.columns:
                    df = df.rename(columns={'A姓名': '姓名'})

                # Handle different 'dislocation position' column names
                # Some files have '脱位范围', others have '脱位角度' or '脱位范围'
                if '脱位角度' in df.columns and '脱位范围' not in df.columns:
                    df = df.rename(columns={'脱位角度': '脱位范围'})

                if verbose:
                    print(f"  ✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
                    print(f"  Columns: {list(df.columns)}")

                # Record raw dataset metadata
                self.raw_data_info[Path(file_path).name] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'target_distribution': df[self.config.TARGET_COLUMN].value_counts().to_dict() if self.config.TARGET_COLUMN in df.columns else None
                }

                all_dataframes.append(df)

            except Exception as e:
                print(f"  ✗ Failed to read file: {e}")
                continue

        if len(all_dataframes) == 0:
            raise ValueError("No data files could be read successfully.")

        # Merge all frames
        if verbose:
            print(f"\n{'='*70}")
            print("Merging dataframes...")

        self.data = pd.concat(all_dataframes, ignore_index=True)

        if verbose:
            print(f"  ✓ Merge complete: {self.data.shape[0]} rows, {self.data.shape[1]} columns total")
            print("="*70)

        return self.data

    def _load_single_file(self, file_path, file_type='data', verbose=True):
        """
        加载单个数据文件（.csv / .xlsx / .xls）。

        Args:
            file_path: 文件路径
            file_type: 文件类型描述（例如 'train' 或 'val'）
            verbose: 是否打印详细信息

        Returns:
            DataFrame: 加载的数据
        """
        if verbose:
            print(f"\nReading {file_type} file: {Path(file_path).name}")

        try:
            df = _read_table(file_path)

            # Standardize column names
            if 'A姓名' in df.columns:
                df = df.rename(columns={'A姓名': '姓名'})

            if '脱位角度' in df.columns and '脱位范围' not in df.columns:
                df = df.rename(columns={'脱位角度': '脱位范围'})

            if verbose:
                print(f"  ✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"  Columns: {list(df.columns)}")

            # Record metadata
            self.raw_data_info[f"{file_type}_{Path(file_path).name}"] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'target_distribution': df[self.config.TARGET_COLUMN].value_counts().to_dict() if self.config.TARGET_COLUMN in df.columns else None
            }

            return df

        except Exception as e:
            print(f"  ✗ Failed to read file: {e}")
            return None

    def load_train_val_data(self, verbose=True):
        """
        分别加载训练集和验证集。

        Args:
            verbose: 是否打印详细信息

        Returns:
            Tuple[DataFrame, DataFrame]: (训练数据, 验证数据)
        """
        if verbose:
            print("\n" + "="*70)
            print(" " * 20 + "Loading Train & Validation Data")
            print("="*70)

        paths = self.config.get_absolute_paths()

        # 加载训练数据
        self.train_data = self._load_single_file(
            paths['train_data_file'],
            file_type='train',
            verbose=verbose
        )

        if self.train_data is None:
            raise ValueError("Failed to load training data file.")

        # 加载验证数据
        self.val_data = self._load_single_file(
            paths['val_data_file'],
            file_type='validation',
            verbose=verbose
        )

        if self.val_data is None:
            raise ValueError("Failed to load validation data file.")

        if verbose:
            print(f"\n{'='*70}")
            print(f"  ✓ Train data: {self.train_data.shape[0]} samples")
            print(f"  ✓ Validation data: {self.val_data.shape[0]} samples")
            print("="*70)

        return self.train_data, self.val_data

    def get_basic_info(self):
        """Return basic descriptive information about the dataset."""
        if self.data is None:
            raise ValueError("Call load_data() before requesting dataset information.")

        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict()
        }

        return info

    def get_target_distribution(self):
        """Return the distribution of the target variable."""
        if self.data is None:
            raise ValueError("Call load_data() before requesting target distribution.")

        if self.config.TARGET_COLUMN not in self.data.columns:
            raise ValueError(f"Target column not found: {self.config.TARGET_COLUMN}")

        # Raw distribution
        raw_distribution = self.data[self.config.TARGET_COLUMN].value_counts()

        # Percentage distribution
        percentage_distribution = self.data[self.config.TARGET_COLUMN].value_counts(normalize=True) * 100

        return {
            'counts': raw_distribution.to_dict(),
            'percentages': percentage_distribution.to_dict()
        }

    def clean_numeric_columns(self, verbose=True):
        """
        Sanitize numerical columns, coercing unparseable values to NaN instead of
        dropping entire rows.

        Args:
            verbose: Whether to print diagnostic information.
        """
        if verbose:
            print("\n  Cleaning numerical features...")

        numeric_cols = [col for col in self.config.NUMERICAL_FEATURES if col in self.data.columns]
        if not numeric_cols:
            if verbose:
                print("  ⚠ No numerical feature columns were found; skipping")
            return

        # 记录所有对应的真实列位置，便于后续整行缺失判断
        numeric_column_positions = []

        for col in numeric_cols:
            matching_positions = [idx for idx, name in enumerate(self.data.columns) if name == col]
            if not matching_positions:
                continue

            for dup_idx, position in enumerate(matching_positions, start=1):
                numeric_column_positions.append(position)

                original_series = self.data.iloc[:, position]
                display_name = col if len(matching_positions) == 1 else f"{col}#{dup_idx}"

                # 统一处理为字符串，方便清洗符号
                series_str = original_series.astype(str).str.strip()

                # 标准化特殊符号
                series_str = series_str.str.replace('−', '-', regex=False)  # 全角负号
                series_str = series_str.str.replace('＋', '+', regex=False)  # 全角正号

                # 去掉末尾的单个+/-符号
                series_str = series_str.str.replace(r'[+-]$', '', regex=True)

                # 空字符或表示缺失的字符串直接视为缺失
                series_str = series_str.replace({'': np.nan, 'nan': np.nan, 'None': np.nan, '无': np.nan})

                # 尝试提取数字；若提取不到则变为NaN
                numeric_series = pd.to_numeric(series_str, errors='coerce')

                # 统计因清洗新产生的缺失值数量
                newly_invalid = (~original_series.isna()) & numeric_series.isna()
                newly_invalid_count = newly_invalid.sum()

                if verbose:
                    total_non_missing = original_series.notna().sum()
                    ratio = (newly_invalid_count / total_non_missing * 100) if total_non_missing > 0 else 0
                    print(f"    - {display_name}: {total_non_missing - newly_invalid_count} converted, {newly_invalid_count} failed to parse ({ratio:.1f}%)")

                self.data.iloc[:, position] = numeric_series.astype(float)

        # Drop rows where all numerical features are missing
        if numeric_column_positions:
            numeric_subset = self.data.iloc[:, numeric_column_positions]
        else:
            numeric_subset = pd.DataFrame(index=self.data.index)

        all_numeric_nan = numeric_subset.isna().all(axis=1)
        if all_numeric_nan.any():
            drop_count = all_numeric_nan.sum()
            valid_mask = ~all_numeric_nan
            self.data = self.data.loc[valid_mask].reset_index(drop=True)

            # Also filter preserved data to match
            if self.preserved_data is not None:
                self.preserved_data = self.preserved_data.loc[valid_mask].reset_index(drop=True)

            if verbose:
                print(f"\n  ✓ Removed rows with all numerical features missing: {drop_count}")

    def clean_categorical_columns(self, verbose=True):
        """
        Normalise categorical columns (trim whitespace, standardise values).

        HISTORICAL BEHAVIOUR (intentionally preserved, not a bug to fix
        here): casting to `str` below turns real missing values (``NaN``)
        into the literal string ``"nan"``. Because this happens before
        FeatureEngineer's categorical SimpleImputer ever runs, that imputer
        never sees an actual missing value to impute -- ``"nan"`` ends up
        label-encoded as its own explicit category instead of being
        most-frequent-imputed. Changing the order to make the imputer
        effective would change training results, so this pass keeps the
        historical order unchanged. See text_pipeline/README.md.

        Args:
            verbose: Whether to print diagnostic information.
        """
        if verbose:
            print("\n  Cleaning categorical features...")

        for col in self.config.CATEGORICAL_FEATURES:
            if col not in self.data.columns:
                if verbose:
                    print(f"    ⚠ Column not found: {col}")
                continue

            # 转换为字符串并去除空白（缺失值将变为字符串"nan"，见上方说明）
            self.data[col] = self.data[col].astype(str).str.strip()

            # 统计类别
            unique_values = self.data[col].unique()
            if verbose:
                print(f"    ✓ {col}: {len(unique_values)} distinct categories {list(unique_values)[:5]}")

    def clean_data(self, verbose=True):
        """
        Orchestrate the full data-cleaning workflow:
        - Retain only the required columns
        - Clean numerical and categorical features
        - Process and encode the target column
        - Remove rows with missing target values

        Args:
            verbose: Whether to print detailed progress information.

        Returns:
            DataFrame: The cleaned dataset.
        """
        if self.data is None:
            raise ValueError("Call load_data() before attempting to clean data.")

        if verbose:
            print("\n" + "="*70)
            print(" " * 25 + "Data Cleaning")
            print("="*70)

        original_shape = self.data.shape
        original_columns = list(self.data.columns)

        # 1. Store preserved columns before any cleaning
        if verbose:
            print("\n1. Preserving identity columns (姓名, 身份证号, 省份)")

        preserve_cols = [col for col in self.config.PRESERVE_COLUMNS if col in self.data.columns]
        if preserve_cols:
            self.preserved_data = self.data[preserve_cols].copy()
            if verbose:
                print(f"  ✓ Preserved {len(preserve_cols)} columns: {preserve_cols}")
        else:
            if verbose:
                print(f"  ⚠ No preserve columns found")

        # 2. Retain only required columns for training
        if verbose:
            print("\n2. Selecting required columns for training")
            print(f"  Original columns: {original_columns}")

        # Build the whitelist of columns to keep
        needed_columns = self.config.FEATURE_COLUMNS + [self.config.TARGET_COLUMN]

        # Determine which columns are available
        existing_columns = [col for col in needed_columns if col in self.data.columns]
        missing_columns = [col for col in needed_columns if col not in self.data.columns]

        if verbose:
            print(f"  Required columns: {needed_columns}")
            print(f"  Present columns: {existing_columns}")
            if missing_columns:
                print(f"  ⚠ Missing columns: {missing_columns}")

        # Keep only the required columns that exist. `.copy()` is required
        # here (not just style): without it, self.data is a view onto the
        # original frame, and later in-place assignments (clean_numeric_columns,
        # clean_categorical_columns) trigger pandas' SettingWithCopyWarning.
        # This does not change any resulting values, rows, or column order.
        self.data = self.data[existing_columns].copy()

        if verbose:
            removed_count = len(original_columns) - len(existing_columns)
            print(f"  ✓ Retained {len(existing_columns)} columns, removed {removed_count}")

        # 3. Clean numerical columns
        if verbose:
            print("\n3. Cleaning numerical columns (strip trailing +/- symbols)")
        self.clean_numeric_columns(verbose=verbose)

        # 4. Clean categorical columns
        if verbose:
            print("\n4. Cleaning categorical columns")
        self.clean_categorical_columns(verbose=verbose)

        # 5. Handle the target column
        if verbose:
            print(f"\n5. Processing target column: {self.config.TARGET_COLUMN}")

        if self.config.TARGET_COLUMN not in self.data.columns:
            raise ValueError(f"Target column not found: {self.config.TARGET_COLUMN}")

        # Display the current distribution of the raw target labels
        if verbose:
            print("  Raw target distribution:")
            print(self.data[self.config.TARGET_COLUMN].value_counts())

        # Remove rows with missing target labels
        before_drop = len(self.data)
        valid_indices = self.data.index[self.data[self.config.TARGET_COLUMN].notna()]
        self.data = self.data.loc[valid_indices]

        # Also filter preserved data to match
        if self.preserved_data is not None:
            self.preserved_data = self.preserved_data.loc[valid_indices]

        after_drop = len(self.data)

        if verbose:
            if before_drop > after_drop:
                print(f"  ✓ Removed {before_drop - after_drop} rows with missing targets")
            else:
                print("  ✓ No missing target values detected")

        # Encode the target labels using the configured mapping
        if self.config.LABEL_MAPPING:
            self.data[self.config.TARGET_COLUMN] = self.data[self.config.TARGET_COLUMN].map(self.config.LABEL_MAPPING)
            if verbose:
                print(f"  ✓ Target labels encoded with mapping: {self.config.LABEL_MAPPING}")
                print("  Encoded distribution:")
                print(self.data[self.config.TARGET_COLUMN].value_counts())

        # Reset index for both dataframes to ensure alignment
        self.data = self.data.reset_index(drop=True)
        if self.preserved_data is not None:
            self.preserved_data = self.preserved_data.reset_index(drop=True)

        # 6. Summary statistics
        if verbose:
            print("\n6. Cleaning summary")
            print(f"  Original shape: {original_shape}")
            print(f"  Cleaned shape: {self.data.shape}")
            print(f"  Samples retained: {self.data.shape[0]} ({self.data.shape[0]/original_shape[0]*100:.1f}%)")
            print("="*70)

        return self.data

    def _clean_dataframe(self, df, preserve_name='data', verbose=True):
        """
        清洗单个DataFrame（通用方法）。

        Args:
            df: 要清洗的DataFrame
            preserve_name: 数据名称（用于日志）
            verbose: 是否打印详细信息

        Returns:
            Tuple[DataFrame, DataFrame]: (清洗后的数据, 保留列)
        """
        if verbose:
            print(f"\n  Cleaning {preserve_name} data...")

        original_shape = df.shape

        preserve_cols = [col for col in self.config.PRESERVE_COLUMNS if col in df.columns]
        preserved = df[preserve_cols].copy() if preserve_cols else None

        # 2. 保留所需列
        needed_columns = self.config.FEATURE_COLUMNS + [self.config.TARGET_COLUMN]
        existing_columns = [col for col in needed_columns if col in df.columns]
        df = df[existing_columns].copy()

        # 3. 清洗数值列
        numeric_cols = [col for col in self.config.NUMERICAL_FEATURES if col in df.columns]
        for col in numeric_cols:
            series_str = df[col].astype(str).str.strip()
            series_str = series_str.str.replace('−', '-', regex=False)
            series_str = series_str.str.replace('＋', '+', regex=False)
            series_str = series_str.str.replace(r'[+-]$', '', regex=True)
            series_str = series_str.replace({'': np.nan, 'nan': np.nan, 'None': np.nan, '无': np.nan})
            df[col] = pd.to_numeric(series_str, errors='coerce').astype(float)

        # 4. 清洗分类列
        # HISTORICAL BEHAVIOUR (intentionally preserved -- see
        # clean_categorical_columns() docstring above and
        # text_pipeline/README.md): astype(str) turns real NaN into the
        # literal string "nan" before FeatureEngineer's categorical imputer
        # ever runs, so that imputer never performs effective most-frequent
        # imputation on categorical missingness. Not changed by this pass.
        for col in self.config.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # 5. 处理目标列
        if self.config.TARGET_COLUMN in df.columns:
            valid_mask = df[self.config.TARGET_COLUMN].notna()
            df = df.loc[valid_mask].copy()
            if preserved is not None:
                preserved = preserved.loc[valid_mask].copy()

            # 编码目标列
            if self.config.LABEL_MAPPING:
                df[self.config.TARGET_COLUMN] = df[self.config.TARGET_COLUMN].map(self.config.LABEL_MAPPING)

        # 6. 重置索引
        df = df.reset_index(drop=True)
        if preserved is not None:
            preserved = preserved.reset_index(drop=True)

        if verbose:
            print(f"    Original: {original_shape[0]} samples -> Cleaned: {df.shape[0]} samples")

        return df, preserved

    def clean_train_val_data(self, verbose=True):
        """
        分别清洗训练集和验证集。

        Args:
            verbose: 是否打印详细信息

        Returns:
            Tuple[DataFrame, DataFrame]: (清洗后的训练数据, 清洗后的验证数据)
        """
        if self.train_data is None or self.val_data is None:
            raise ValueError("Call load_train_val_data() before cleaning.")

        if verbose:
            print("\n" + "="*70)
            print(" " * 20 + "Cleaning Train & Validation Data")
            print("="*70)

        # 清洗训练数据
        self.train_data, self.preserved_train_data = self._clean_dataframe(
            self.train_data,
            preserve_name='train',
            verbose=verbose
        )

        # 清洗验证数据
        self.val_data, self.preserved_val_data = self._clean_dataframe(
            self.val_data,
            preserve_name='validation',
            verbose=verbose
        )

        if verbose:
            print(f"\n{'='*70}")
            print(f"  ✓ Cleaned train data: {self.train_data.shape[0]} samples")
            print(f"  ✓ Cleaned validation data: {self.val_data.shape[0]} samples")

            # 显示目标分布
            print("\n  Target distribution:")
            print("    Train:")
            train_dist = self.train_data[self.config.TARGET_COLUMN].value_counts()
            for label, count in train_dist.items():
                label_name = self.config.LABEL_NAMES.get(label, label)
                print(f"      {label_name}: {count} ({count/len(self.train_data)*100:.1f}%)")

            print("    Validation:")
            val_dist = self.val_data[self.config.TARGET_COLUMN].value_counts()
            for label, count in val_dist.items():
                label_name = self.config.LABEL_NAMES.get(label, label)
                print(f"      {label_name}: {count} ({count/len(self.val_data)*100:.1f}%)")

            print("="*70)

        return self.train_data, self.val_data

    def print_data_summary(self):
        """Print a high-level summary of the cleaned dataset."""
        if self.data is None:
            raise ValueError("Call load_data() before printing a summary.")

        print("\n" + "="*70)
        print(" " * 25 + "Data Summary")
        print("="*70)

        # Basic dataset statistics
        print("\n[Basic Information]")
        print(f"  Samples: {self.data.shape[0]}")
        print(f"  Features: {self.data.shape[1] - 1}")  # Minus the target column
        print(f"  Total columns: {self.data.shape[1]}")

        # Target variable distribution
        if self.config.TARGET_COLUMN in self.data.columns:
            print("\n[Target Distribution]")
            target_counts = self.data[self.config.TARGET_COLUMN].value_counts()
            for value, count in target_counts.items():
                label = self.config.LABEL_NAMES.get(value, value)
                percentage = count / len(self.data) * 100
                print(f"  {label}: {count} ({percentage:.2f}%)")

            # Compute imbalance ratio for binary targets
            if len(target_counts) == 2:
                imbalance_ratio = target_counts.max() / target_counts.min()
                print(f"  Imbalance ratio: 1:{imbalance_ratio:.2f}")

        # Feature categories
        print("\n[Feature Types]")
        print(f"  Categorical features: {len(self.config.CATEGORICAL_FEATURES)}")
        for feat in self.config.CATEGORICAL_FEATURES:
            if feat in self.data.columns:
                print(f"    - {feat}")

        print(f"  Numerical features: {len(self.config.NUMERICAL_FEATURES)}")
        for feat in self.config.NUMERICAL_FEATURES:
            if feat in self.data.columns:
                print(f"    - {feat}")

        # Missing-value statistics
        missing = self.data.isnull().sum()
        missing_features = missing[missing > 0]

        print("\n[Missing Values]")
        if len(missing_features) > 0:
            print(f"  Features with missing values: {len(missing_features)}")
            for feat, count in missing_features.items():
                percentage = count / len(self.data) * 100
                print(f"    - {feat}: {count} ({percentage:.2f}%)")
        else:
            print("  No missing values detected")

        # Data types present in the dataset
        print("\n[Data Types]")
        dtype_counts = self.data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")

        print("="*70 + "\n")

    def get_data(self):
        """Return the currently loaded (and potentially cleaned) dataset."""
        return self.data

    def get_preserved_data(self):
        """Return the preserved columns (姓名, 身份证号, 省份)."""
        return self.preserved_data

    def get_train_data(self):
        """返回清洗后的训练数据"""
        return self.train_data

    def get_val_data(self):
        """返回清洗后的验证数据"""
        return self.val_data

    def get_preserved_train_data(self):
        """返回训练数据的保留列"""
        return self.preserved_train_data

    def get_preserved_val_data(self):
        """返回验证数据的保留列"""
        return self.preserved_val_data

    def save_cleaned_data(self, output_path='text_classification/cleaned_data.csv'):
        """
        Persist the cleaned dataset to disk.

        Args:
            output_path: Destination path for the CSV file.
        """
        if self.data is None:
            raise ValueError("Call load_data() before attempting to save data.")

        self.data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ Cleaned data saved to: {output_path}")


def load_and_clean_data(verbose=True):
    """
    Convenience helper that loads and cleans all configured data files.

    Args:
        verbose: Whether to print detailed progress information.

    Returns:
        DataFrame: The cleaned dataset.
        DataLoader: The data loader instance (for access to metadata).
    """
    loader = DataLoader()
    loader.load_data(verbose=verbose)
    loader.clean_data(verbose=verbose)

    if verbose:
        loader.print_data_summary()

    return loader.get_data(), loader


def load_and_clean_train_val_data(verbose=True):
    """
    便捷函数：分别加载和清洗训练集与验证集。

    Args:
        verbose: 是否打印详细信息

    Returns:
        Tuple: (train_data, val_data, loader)
    """
    loader = DataLoader()
    loader.load_train_val_data(verbose=verbose)
    loader.clean_train_val_data(verbose=verbose)

    return loader.get_train_data(), loader.get_val_data(), loader
