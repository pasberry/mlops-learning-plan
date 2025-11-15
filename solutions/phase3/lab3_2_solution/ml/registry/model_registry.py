"""Model registry for versioning and managing models."""
import os
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ModelRegistry:
    """Simple file-based model registry.

    Structure:
        models/staging/{model_name}/
            v1/
                model.pt
                config.yaml
                metrics.json
                metadata.json
            v2/
            ...

        models/production/{model_name}/
            current -> v2/  (symlink)
            v1/
            v2/
            ...
    """

    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.staging_dir = self.base_dir / "staging"
        self.production_dir = self.base_dir / "production"

        # Create directories
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.production_dir.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model_name: str,
        model_path: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        stage: str = "staging",
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a trained model.

        Args:
            model_name: Name of the model (e.g., 'ctr_model')
            model_path: Path to model.pt file
            config: Model configuration dict
            metrics: Evaluation metrics dict
            stage: 'staging' or 'production'
            version: Version string (auto-generated if None)
            metadata: Additional metadata

        Returns:
            Path to registered model directory
        """
        # Determine version
        if version is None:
            version = self._get_next_version(model_name, stage)

        # Create model directory
        if stage == "staging":
            model_dir = self.staging_dir / model_name / version
        elif stage == "production":
            model_dir = self.production_dir / model_name / version
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 'staging' or 'production'")

        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        shutil.copy(model_path, model_dir / "model.pt")

        # Save config
        with open(model_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f)

        # Save metrics
        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save metadata
        metadata = metadata or {}
        metadata.update({
            'model_name': model_name,
            'version': version,
            'stage': stage,
            'registered_at': datetime.now().isoformat(),
            'model_path': str(model_dir / "model.pt")
        })

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Model registered: {model_name} v{version} ({stage})")
        print(f"   Path: {model_dir}")
        print(f"   Metrics: {metrics}")

        return str(model_dir)

    def _get_next_version(self, model_name: str, stage: str) -> str:
        """Get next version number for a model."""
        if stage == "staging":
            model_base_dir = self.staging_dir / model_name
        else:
            model_base_dir = self.production_dir / model_name

        if not model_base_dir.exists():
            return "v1"

        # Find existing versions
        existing_versions = [
            d.name for d in model_base_dir.iterdir()
            if d.is_dir() and d.name.startswith('v')
        ]

        if not existing_versions:
            return "v1"

        # Extract version numbers
        version_numbers = []
        for v in existing_versions:
            try:
                version_numbers.append(int(v[1:]))  # Remove 'v' prefix
            except ValueError:
                continue

        next_version = max(version_numbers) + 1 if version_numbers else 1
        return f"v{next_version}"

    def get_model_path(self, model_name: str, version: str, stage: str = "staging") -> Path:
        """Get path to a specific model version."""
        if stage == "staging":
            return self.staging_dir / model_name / version / "model.pt"
        else:
            return self.production_dir / model_name / version / "model.pt"

    def get_latest_version(self, model_name: str, stage: str = "staging") -> Optional[str]:
        """Get the latest version of a model."""
        if stage == "staging":
            model_base_dir = self.staging_dir / model_name
        else:
            model_base_dir = self.production_dir / model_name

        if not model_base_dir.exists():
            return None

        versions = [
            d.name for d in model_base_dir.iterdir()
            if d.is_dir() and d.name.startswith('v')
        ]

        if not versions:
            return None

        # Sort by version number
        versions.sort(key=lambda v: int(v[1:]))
        return versions[-1]

    def list_models(self, stage: str = "staging") -> Dict[str, list]:
        """List all models in a stage."""
        if stage == "staging":
            base_dir = self.staging_dir
        else:
            base_dir = self.production_dir

        models = {}
        for model_dir in base_dir.iterdir():
            if model_dir.is_dir():
                versions = [
                    v.name for v in model_dir.iterdir()
                    if v.is_dir() and v.name.startswith('v')
                ]
                if versions:
                    models[model_dir.name] = sorted(versions, key=lambda v: int(v[1:]))

        return models

    def promote_to_production(self, model_name: str, version: str) -> str:
        """Promote a staging model to production.

        Args:
            model_name: Name of the model
            version: Version to promote

        Returns:
            Path to production model
        """
        staging_model_dir = self.staging_dir / model_name / version

        if not staging_model_dir.exists():
            raise FileNotFoundError(f"Model not found: {staging_model_dir}")

        # Copy to production
        production_model_dir = self.production_dir / model_name / version
        shutil.copytree(staging_model_dir, production_model_dir, dirs_exist_ok=True)

        # Update metadata
        metadata_path = production_model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metadata['stage'] = 'production'
        metadata['promoted_at'] = datetime.now().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create 'current' symlink
        current_link = self.production_dir / model_name / "current"
        if current_link.exists():
            current_link.unlink()

        current_link.symlink_to(version, target_is_directory=True)

        print(f"✅ Model promoted to production: {model_name} {version}")
        print(f"   Current production model: {current_link} -> {version}")

        return str(production_model_dir)


# Test
if __name__ == '__main__':
    registry = ModelRegistry()

    # Test registration
    print("Testing model registry...")

    # Simulate registering a model
    test_config = {
        'model': {'hidden_dims': [256, 128, 64]},
        'training': {'batch_size': 128}
    }

    test_metrics = {
        'test_accuracy': 0.85,
        'test_auc': 0.90
    }

    # List models
    print("\nStaging models:", registry.list_models('staging'))
    print("Production models:", registry.list_models('production'))
