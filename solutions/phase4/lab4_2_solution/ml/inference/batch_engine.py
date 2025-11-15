"""
Batch inference engine for processing large datasets.

Efficiently processes batches of data for model predictions.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from datetime import datetime
import json

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceDataset(Dataset):
    """PyTorch Dataset for batch inference."""

    def __init__(self, df: pd.DataFrame, feature_columns: List[str]):
        """Initialize dataset.

        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
        """
        self.df = df
        self.feature_columns = feature_columns

        # Extract features
        self.features = df[feature_columns].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.features[idx], dtype=torch.float32)


class BatchInferenceEngine:
    """Engine for running batch inference on large datasets."""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 256,
        num_workers: int = 4,
        device: Optional[str] = None
    ):
        """Initialize batch inference engine.

        Args:
            model_path: Path to model checkpoint
            batch_size: Batch size for inference
            num_workers: Number of data loader workers
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = None
        self.feature_columns = []
        self.model_metadata = {}
        self._load_model()

    def _load_model(self):
        """Load model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Extract model
            if isinstance(checkpoint, dict):
                # Load model architecture
                from torch import nn

                input_dim = checkpoint.get('input_dim', 10)
                self.model = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # Extract metadata
                self.feature_columns = checkpoint.get('feature_names', [])
                self.model_metadata = checkpoint.get('metadata', {})
            else:
                self.model = checkpoint

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully")
            logger.info(f"Feature columns: {self.feature_columns}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        id_column: Optional[str] = None,
        include_features: bool = False
    ) -> pd.DataFrame:
        """Run predictions on a DataFrame.

        Args:
            df: Input DataFrame with features
            id_column: Optional ID column to include in output
            include_features: Whether to include input features in output

        Returns:
            DataFrame with predictions
        """
        logger.info(f"Running batch inference on {len(df)} rows")

        # Create dataset and dataloader
        dataset = InferenceDataset(df, self.feature_columns)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == 'cuda'
        )

        # Run inference
        predictions = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                batch = batch.to(self.device)

                # Get predictions
                outputs = self.model(batch)
                batch_preds = outputs.cpu().numpy().squeeze()

                # Handle single prediction
                if batch_preds.ndim == 0:
                    batch_preds = np.array([batch_preds])

                predictions.extend(batch_preds.tolist())

                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"Processed {(batch_idx + 1) * self.batch_size} rows")

        # Create output DataFrame
        output_data = {
            'prediction_score': predictions,
            'prediction_class': [1 if p >= 0.5 else 0 for p in predictions],
            'timestamp': datetime.utcnow()
        }

        # Add ID column if specified
        if id_column and id_column in df.columns:
            output_data[id_column] = df[id_column].values

        # Add features if requested
        if include_features:
            for col in self.feature_columns:
                if col in df.columns:
                    output_data[col] = df[col].values

        result_df = pd.DataFrame(output_data)

        logger.info(f"Batch inference complete: {len(result_df)} predictions")

        return result_df

    def predict_csv(
        self,
        input_path: str,
        output_path: str,
        id_column: Optional[str] = None,
        include_features: bool = False,
        chunksize: Optional[int] = None
    ):
        """Run predictions on a CSV file.

        Args:
            input_path: Path to input CSV
            output_path: Path to output CSV
            id_column: Optional ID column to include
            include_features: Whether to include features in output
            chunksize: Process in chunks if specified
        """
        logger.info(f"Processing CSV: {input_path}")

        if chunksize:
            # Process in chunks for very large files
            self._predict_csv_chunked(
                input_path,
                output_path,
                id_column,
                include_features,
                chunksize
            )
        else:
            # Load entire file
            df = pd.read_csv(input_path)
            result_df = self.predict_dataframe(df, id_column, include_features)
            result_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")

    def _predict_csv_chunked(
        self,
        input_path: str,
        output_path: str,
        id_column: Optional[str],
        include_features: bool,
        chunksize: int
    ):
        """Process CSV in chunks."""
        logger.info(f"Processing in chunks of {chunksize}")

        first_chunk = True
        total_rows = 0

        for chunk_idx, chunk_df in enumerate(pd.read_csv(input_path, chunksize=chunksize)):
            # Run predictions on chunk
            result_df = self.predict_dataframe(chunk_df, id_column, include_features)

            # Write to output
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            result_df.to_csv(output_path, mode=mode, header=header, index=False)

            total_rows += len(result_df)
            first_chunk = False

            logger.info(f"Processed chunk {chunk_idx + 1}: {total_rows} total rows")

        logger.info(f"Chunked processing complete: {total_rows} predictions saved")

    def predict_batch(
        self,
        features_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run predictions on a list of feature dictionaries.

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of prediction dictionaries
        """
        # Convert to DataFrame
        df = pd.DataFrame(features_list)

        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        # Run predictions
        result_df = self.predict_dataframe(df, include_features=True)

        # Convert to list of dictionaries
        results = result_df.to_dict('records')

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'feature_count': len(self.feature_columns),
            'features': self.feature_columns,
            'metadata': self.model_metadata
        }


def run_batch_inference(
    input_path: str,
    output_path: str,
    model_path: str,
    batch_size: int = 256,
    id_column: Optional[str] = None,
    include_features: bool = False,
    chunksize: Optional[int] = None
):
    """Run batch inference on a file.

    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        model_path: Path to model checkpoint
        batch_size: Batch size for inference
        id_column: Optional ID column
        include_features: Include features in output
        chunksize: Process in chunks if specified
    """
    # Create engine
    engine = BatchInferenceEngine(
        model_path=model_path,
        batch_size=batch_size
    )

    # Run inference
    engine.predict_csv(
        input_path=input_path,
        output_path=output_path,
        id_column=id_column,
        include_features=include_features,
        chunksize=chunksize
    )

    # Print stats
    stats = engine.get_stats()
    logger.info(f"Inference stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch inference engine")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--id-column", help="ID column name")
    parser.add_argument("--include-features", action="store_true", help="Include features")
    parser.add_argument("--chunksize", type=int, help="Process in chunks")

    args = parser.parse_args()

    run_batch_inference(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        batch_size=args.batch_size,
        id_column=args.id_column,
        include_features=args.include_features,
        chunksize=args.chunksize
    )
