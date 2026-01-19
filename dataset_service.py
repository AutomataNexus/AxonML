"""
NexusConnectBridge - Dataset Service

Manages dataset uploads, parsing, and built-in datasets for AI model training.

Copyright 2026 AutomataNexus, LLC. All rights reserved.
"""

import os
import json
import uuid
import asyncio
import logging
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from io import BytesIO
import zipfile

from ..config.settings import settings

logger = logging.getLogger(__name__)

# Tier-based upload limits (in bytes)
TIER_UPLOAD_LIMITS = {
    "free": 10 * 1024 * 1024,       # 10 MB
    "basic": 50 * 1024 * 1024,      # 50 MB
    "pro": 500 * 1024 * 1024,       # 500 MB
    "enterprise": 2 * 1024 * 1024 * 1024,  # 2 GB
}

# Tier persistence
TIER_PERSISTENCE = {
    "free": False,      # Session only
    "basic": False,     # Session only
    "pro": True,        # Persisted to DB
    "enterprise": True, # Persisted to DB
}


@dataclass
class DatasetInfo:
    """Dataset metadata and schema information."""
    id: str
    name: str
    description: str
    source: str  # "builtin", "uploaded", "kaggle", "uci"
    format: str  # "csv", "images", "json", "parquet"
    size_bytes: int
    num_samples: int
    num_features: int
    num_classes: Optional[int] = None
    feature_names: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    data_types: Dict[str, str] = field(default_factory=dict)
    shape: Optional[Tuple[int, ...]] = None
    preview_data: Optional[List[Dict]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[int] = None
    persistent: bool = False
    file_path: Optional[str] = None


@dataclass
class ParsedDataset:
    """Result of parsing a dataset."""
    info: DatasetInfo
    loading_code: str  # PyTorch data loading code
    preprocessing_code: str  # Preprocessing/transform code
    input_shape: Tuple[int, ...]
    output_size: int
    is_image: bool = False
    is_tabular: bool = True


# Built-in datasets available for training
BUILTIN_DATASETS = {
    "mnist": DatasetInfo(
        id="builtin-mnist",
        name="MNIST Handwritten Digits",
        description="70,000 grayscale images of handwritten digits (0-9). 28x28 pixels. Classic image classification dataset.",
        source="builtin",
        format="images",
        size_bytes=50 * 1024 * 1024,
        num_samples=70000,
        num_features=784,
        num_classes=10,
        feature_names=["pixel_" + str(i) for i in range(784)],
        target_column="label",
        shape=(1, 28, 28),
    ),
    "fashion_mnist": DatasetInfo(
        id="builtin-fashion-mnist",
        name="Fashion MNIST",
        description="70,000 grayscale images of clothing items (10 categories). 28x28 pixels. Drop-in MNIST replacement.",
        source="builtin",
        format="images",
        size_bytes=50 * 1024 * 1024,
        num_samples=70000,
        num_features=784,
        num_classes=10,
        feature_names=["pixel_" + str(i) for i in range(784)],
        target_column="label",
        shape=(1, 28, 28),
    ),
    "cifar10": DatasetInfo(
        id="builtin-cifar10",
        name="CIFAR-10",
        description="60,000 color images in 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck). 32x32 RGB.",
        source="builtin",
        format="images",
        size_bytes=170 * 1024 * 1024,
        num_samples=60000,
        num_features=3072,
        num_classes=10,
        feature_names=[],
        target_column="label",
        shape=(3, 32, 32),
    ),
    "cifar100": DatasetInfo(
        id="builtin-cifar100",
        name="CIFAR-100",
        description="60,000 color images in 100 fine-grained classes. 32x32 RGB. More challenging than CIFAR-10.",
        source="builtin",
        format="images",
        size_bytes=170 * 1024 * 1024,
        num_samples=60000,
        num_features=3072,
        num_classes=100,
        feature_names=[],
        target_column="label",
        shape=(3, 32, 32),
    ),
    "iris": DatasetInfo(
        id="builtin-iris",
        name="Iris Flower Dataset",
        description="150 samples of iris flowers with 4 features. 3 classes: Setosa, Versicolor, Virginica.",
        source="builtin",
        format="csv",
        size_bytes=5 * 1024,
        num_samples=150,
        num_features=4,
        num_classes=3,
        feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        target_column="species",
        data_types={"sepal_length": "float", "sepal_width": "float", "petal_length": "float", "petal_width": "float", "species": "int"},
        shape=(4,),
    ),
    "wine": DatasetInfo(
        id="builtin-wine",
        name="Wine Quality Dataset",
        description="6,497 samples of wine with chemical properties. Predict quality rating (0-10).",
        source="builtin",
        format="csv",
        size_bytes=400 * 1024,
        num_samples=6497,
        num_features=11,
        num_classes=10,
        feature_names=["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"],
        target_column="quality",
        shape=(11,),
    ),
    "breast_cancer": DatasetInfo(
        id="builtin-breast-cancer",
        name="Breast Cancer Wisconsin",
        description="569 samples for breast cancer classification. 30 features. Binary classification (malignant/benign).",
        source="builtin",
        format="csv",
        size_bytes=100 * 1024,
        num_samples=569,
        num_features=30,
        num_classes=2,
        feature_names=["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness"],
        target_column="target",
        shape=(30,),
    ),
}


class DatasetService:
    """Service for managing datasets for AI training."""

    def __init__(self):
        """Initialize dataset service."""
        self.upload_dir = Path(settings.upload_path) / "datasets"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage for session datasets
        self._session_datasets: Dict[str, ParsedDataset] = {}

        # Cleanup old session datasets periodically
        self._cleanup_interval = timedelta(hours=1)

        logger.info(f"DatasetService initialized. Upload dir: {self.upload_dir}")

    def get_builtin_datasets(self) -> List[DatasetInfo]:
        """Get list of built-in datasets."""
        return list(BUILTIN_DATASETS.values())

    def get_builtin_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """Get a specific built-in dataset."""
        # Handle various ID formats
        clean_id = dataset_id.lower().replace("-", "_").replace("builtin_", "").replace("builtin-", "")
        return BUILTIN_DATASETS.get(clean_id)

    def get_upload_limit(self, tier: str) -> int:
        """Get upload limit in bytes for a tier."""
        return TIER_UPLOAD_LIMITS.get(tier.lower(), TIER_UPLOAD_LIMITS["free"])

    def is_persistent_tier(self, tier: str) -> bool:
        """Check if tier has persistent dataset storage."""
        return TIER_PERSISTENCE.get(tier.lower(), False)

    async def parse_uploaded_dataset(
        self,
        file_content: bytes,
        filename: str,
        user_id: int,
        tier: str = "free"
    ) -> ParsedDataset:
        """
        Parse an uploaded dataset file.

        Supports: CSV, JSON, Parquet, ZIP (images)
        """
        file_size = len(file_content)
        max_size = self.get_upload_limit(tier)

        if file_size > max_size:
            raise ValueError(f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds tier limit ({max_size / 1024 / 1024:.0f}MB)")

        file_ext = Path(filename).suffix.lower()
        dataset_id = f"user_{user_id}_{uuid.uuid4().hex[:8]}"

        try:
            if file_ext == ".csv":
                return await self._parse_csv(file_content, filename, dataset_id, user_id, tier)
            elif file_ext == ".json":
                return await self._parse_json(file_content, filename, dataset_id, user_id, tier)
            elif file_ext == ".parquet":
                return await self._parse_parquet(file_content, filename, dataset_id, user_id, tier)
            elif file_ext == ".zip":
                return await self._parse_image_zip(file_content, filename, dataset_id, user_id, tier)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Supported: .csv, .json, .parquet, .zip")
        except Exception as e:
            logger.error(f"Failed to parse dataset {filename}: {e}")
            raise

    async def _parse_csv(
        self,
        content: bytes,
        filename: str,
        dataset_id: str,
        user_id: int,
        tier: str
    ) -> ParsedDataset:
        """Parse a CSV file and generate loading code."""
        df = pd.read_csv(BytesIO(content))

        # Analyze the dataframe
        num_samples, num_features = df.shape
        feature_names = df.columns.tolist()
        data_types = {col: str(df[col].dtype) for col in df.columns}

        # Try to identify target column (last column or 'target', 'label', 'y', 'class')
        target_candidates = ["target", "label", "y", "class", "output"]
        target_column = None
        for candidate in target_candidates:
            if candidate in [c.lower() for c in feature_names]:
                target_column = feature_names[[c.lower() for c in feature_names].index(candidate)]
                break
        if not target_column:
            target_column = feature_names[-1]  # Default to last column

        # Count classes if target is categorical
        num_classes = None
        if target_column:
            unique_values = df[target_column].nunique()
            if unique_values <= 100:  # Likely classification
                num_classes = unique_values

        # Feature columns (excluding target)
        feature_cols = [c for c in feature_names if c != target_column]

        # Save file if persistent tier
        file_path = None
        persistent = self.is_persistent_tier(tier)
        if persistent:
            file_path = str(self.upload_dir / f"{dataset_id}.csv")
            with open(file_path, 'wb') as f:
                f.write(content)

        # Create dataset info
        info = DatasetInfo(
            id=dataset_id,
            name=Path(filename).stem,
            description=f"Uploaded CSV: {num_samples} samples, {len(feature_cols)} features",
            source="uploaded",
            format="csv",
            size_bytes=len(content),
            num_samples=num_samples,
            num_features=len(feature_cols),
            num_classes=num_classes,
            feature_names=feature_cols,
            target_column=target_column,
            data_types=data_types,
            shape=(len(feature_cols),),
            preview_data=df.head(5).to_dict('records'),
            user_id=user_id,
            persistent=persistent,
            file_path=file_path,
        )

        # Generate loading code
        loading_code = self._generate_csv_loading_code(info, file_path)
        preprocessing_code = self._generate_tabular_preprocessing_code(info)

        parsed = ParsedDataset(
            info=info,
            loading_code=loading_code,
            preprocessing_code=preprocessing_code,
            input_shape=(len(feature_cols),),
            output_size=num_classes or 1,
            is_image=False,
            is_tabular=True,
        )

        # Store in session
        self._session_datasets[dataset_id] = parsed

        return parsed

    async def _parse_json(
        self,
        content: bytes,
        filename: str,
        dataset_id: str,
        user_id: int,
        tier: str
    ) -> ParsedDataset:
        """Parse a JSON file."""
        data = json.loads(content.decode('utf-8'))

        # Convert to DataFrame if possible
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            raise ValueError("JSON must be an array of objects or have a 'data' key")

        # Reuse CSV parsing logic
        csv_content = df.to_csv(index=False).encode('utf-8')
        return await self._parse_csv(csv_content, filename.replace('.json', '.csv'), dataset_id, user_id, tier)

    async def _parse_parquet(
        self,
        content: bytes,
        filename: str,
        dataset_id: str,
        user_id: int,
        tier: str
    ) -> ParsedDataset:
        """Parse a Parquet file."""
        df = pd.read_parquet(BytesIO(content))
        csv_content = df.to_csv(index=False).encode('utf-8')
        return await self._parse_csv(csv_content, filename.replace('.parquet', '.csv'), dataset_id, user_id, tier)

    async def _parse_image_zip(
        self,
        content: bytes,
        filename: str,
        dataset_id: str,
        user_id: int,
        tier: str
    ) -> ParsedDataset:
        """
        Parse a ZIP file containing images.

        Expected structure:
        - images/class1/*.png
        - images/class2/*.png
        OR
        - train/class1/*.png
        - train/class2/*.png
        """
        with zipfile.ZipFile(BytesIO(content), 'r') as zf:
            # Analyze structure
            file_list = zf.namelist()
            image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

            if not image_files:
                raise ValueError("ZIP file contains no image files (.png, .jpg, .jpeg)")

            # Detect class structure from directories
            classes = set()
            for f in image_files:
                parts = Path(f).parts
                if len(parts) >= 2:
                    # Second-to-last part is likely the class
                    classes.add(parts[-2])

            num_classes = len(classes) if classes else 1
            num_samples = len(image_files)

            # Try to detect image size from first image
            from PIL import Image
            first_image = image_files[0]
            with zf.open(first_image) as img_file:
                img = Image.open(img_file)
                width, height = img.size
                channels = len(img.getbands())

        # Save if persistent
        file_path = None
        persistent = self.is_persistent_tier(tier)
        if persistent:
            file_path = str(self.upload_dir / f"{dataset_id}.zip")
            with open(file_path, 'wb') as f:
                f.write(content)

        info = DatasetInfo(
            id=dataset_id,
            name=Path(filename).stem,
            description=f"Image dataset: {num_samples} images, {num_classes} classes, {width}x{height}",
            source="uploaded",
            format="images",
            size_bytes=len(content),
            num_samples=num_samples,
            num_features=width * height * channels,
            num_classes=num_classes,
            feature_names=list(classes),
            target_column="class",
            shape=(channels, height, width),
            user_id=user_id,
            persistent=persistent,
            file_path=file_path,
        )

        loading_code = self._generate_image_loading_code(info, file_path)
        preprocessing_code = self._generate_image_preprocessing_code(info)

        parsed = ParsedDataset(
            info=info,
            loading_code=loading_code,
            preprocessing_code=preprocessing_code,
            input_shape=(channels, height, width),
            output_size=num_classes,
            is_image=True,
            is_tabular=False,
        )

        self._session_datasets[dataset_id] = parsed
        return parsed

    def _generate_csv_loading_code(self, info: DatasetInfo, file_path: Optional[str]) -> str:
        """Generate PyTorch code for loading a CSV dataset."""
        feature_cols = info.feature_names
        target_col = info.target_column

        if file_path:
            # Persistent dataset - load from file
            return f'''# Load CSV dataset
import pandas as pd
from torch.utils.data import TensorDataset

df = pd.read_csv("{file_path}")
X = df[{feature_cols}].values.astype(np.float32)
y = df["{target_col}"].values

# Encode labels if needed
from sklearn.preprocessing import LabelEncoder
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)

y = y.astype(np.int64)

# Normalize features
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# Create tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
'''
        else:
            # Session dataset - data embedded
            return f'''# Load embedded dataset
# Dataset: {info.name}
# Features: {len(feature_cols)}, Samples: {info.num_samples}

# Data will be provided via API
X = dataset_X  # Shape: ({info.num_samples}, {len(feature_cols)})
y = dataset_y  # Shape: ({info.num_samples},)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
'''

    def _generate_tabular_preprocessing_code(self, info: DatasetInfo) -> str:
        """Generate preprocessing code for tabular data."""
        return f'''# Preprocessing for {info.name}
# Normalize numerical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
'''

    def _generate_image_loading_code(self, info: DatasetInfo, file_path: Optional[str]) -> str:
        """Generate PyTorch code for loading an image dataset."""
        return f'''# Load image dataset from ZIP
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import zipfile
import tempfile

# Extract ZIP to temp directory
with zipfile.ZipFile("{file_path or 'dataset.zip'}", 'r') as zf:
    zf.extractall('/tmp/dataset_{info.id}')

transform = transforms.Compose([
    transforms.Resize(({info.shape[1]}, {info.shape[2]})),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*{info.shape[0]}, std=[0.5]*{info.shape[0]})
])

dataset = ImageFolder('/tmp/dataset_{info.id}', transform=transform)
'''

    def _generate_image_preprocessing_code(self, info: DatasetInfo) -> str:
        """Generate preprocessing code for image data."""
        return f'''# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(({info.shape[1]}, {info.shape[2]})),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*{info.shape[0]}, std=[0.5]*{info.shape[0]})
])
'''

    def generate_dataset_code(self, dataset_id: str) -> Tuple[str, str]:
        """
        Generate complete dataset loading and preprocessing code.

        Returns: (loading_code, preprocessing_code)
        """
        # Check builtin first
        builtin = self.get_builtin_dataset(dataset_id)
        if builtin:
            return self._generate_builtin_code(builtin)

        # Check session datasets
        if dataset_id in self._session_datasets:
            parsed = self._session_datasets[dataset_id]
            return parsed.loading_code, parsed.preprocessing_code

        raise ValueError(f"Dataset not found: {dataset_id}")

    def _generate_builtin_code(self, info: DatasetInfo) -> Tuple[str, str]:
        """Generate code for a built-in dataset."""
        dataset_name = info.id.replace("builtin-", "").replace("-", "_")

        if dataset_name == "mnist":
            loading = '''# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f'Training samples: {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')
'''
        elif dataset_name == "fashion_mnist":
            loading = '''# Fashion MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
val_dataset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f'Training samples: {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')
'''
        elif dataset_name == "cifar10":
            loading = '''# CIFAR-10 Dataset
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
val_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f'Training samples: {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')
'''
        elif dataset_name == "cifar100":
            loading = '''# CIFAR-100 Dataset
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = datasets.CIFAR100('data', train=True, download=True, transform=transform_train)
val_dataset = datasets.CIFAR100('data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f'Training samples: {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')
'''
        elif dataset_name in ["iris", "wine", "breast_cancer"]:
            loading = f'''# {info.name} Dataset
from sklearn.datasets import load_{dataset_name}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_{dataset_name}()
X, y = data.data, data.target

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f'Training samples: {{len(train_dataset)}}')
print(f'Validation samples: {{len(val_dataset)}}')
'''
        else:
            # Default to MNIST
            loading = '''# Default: MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
'''

        preprocessing = f"# Dataset: {info.name}\n# Samples: {info.num_samples}, Classes: {info.num_classes}"
        return loading, preprocessing

    def get_dataset_info(self, dataset_id: str) -> Optional[DatasetInfo]:
        """Get info for any dataset (builtin or uploaded)."""
        # Check builtin
        builtin = self.get_builtin_dataset(dataset_id)
        if builtin:
            return builtin

        # Check session
        if dataset_id in self._session_datasets:
            return self._session_datasets[dataset_id].info

        return None

    def get_session_dataset(self, dataset_id: str) -> Optional[ParsedDataset]:
        """Get a parsed session dataset."""
        return self._session_datasets.get(dataset_id)

    def cleanup_session_datasets(self, user_id: int):
        """Clean up non-persistent datasets for a user."""
        to_remove = [
            k for k, v in self._session_datasets.items()
            if v.info.user_id == user_id and not v.info.persistent
        ]
        for k in to_remove:
            del self._session_datasets[k]


# Module-level service instance
_dataset_service: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    """Get or create the dataset service instance."""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service
