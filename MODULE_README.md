# Face Mask Detection - Separated Modules

This project contains the face mask detection code separated into two main modules:

1. **Data Preprocessing** (`src/data_preprocessing.py`)
2. **Model Training** (`src/model_training.py`)

## Project Structure

```
face-mask-detection-mlops/
├── src/
│   ├── data_preprocessing.py    # Data loading, preprocessing, and augmentation
│   ├── model_training.py        # Model building, training, and evaluation
│   └── __init__.py
├── data/
│   ├── raw/                     # Raw dataset (images.zip)
│   └── processed/               # Processed data
├── models/                      # Trained models
├── config/
│   └── config.yaml             # Configuration file
├── run_pipeline.py             # Main pipeline runner
└── README.md
```

## Features

### Data Preprocessing Module (`data_preprocessing.py`)

- **Dataset Organization**: Automatically organizes images into train/test folders by class
- **Data Loading**: Loads images from organized folder structure
- **Image Preprocessing**: Resizes images to required dimensions (224x224)
- **Data Augmentation**: Applies rotation, shifts, flips, and zoom for better training
- **Visualization**: Shows sample images and augmented data
- **Data Generators**: Creates Keras data generators for training

**Key Methods:**
- `setup_paths()`: Sets up project directory structure
- `organize_dataset()`: Organizes raw data into class folders
- `get_dataset()`: Loads and preprocesses images
- `prepare_data_generators()`: Creates training and validation generators
- `process_full_pipeline()`: Runs complete preprocessing pipeline

### Model Training Module (`model_training.py`)

- **Model Architecture**: MobileNetV2-based transfer learning model
- **Custom Layers**: Adds custom classification layers as per original implementation
- **Training Pipeline**: Complete training with callbacks and monitoring
- **Visualization**: Plots training history (accuracy and loss)
- **Model Saving**: Saves trained model for later use
- **Evaluation**: Evaluates model performance

**Key Methods:**
- `build_model()`: Builds MobileNetV2-based architecture
- `compile_model()`: Compiles model with optimizer and loss function
- `train_model()`: Trains the model with given data generators
- `visualize_training_history()`: Shows training plots
- `save_model()`: Saves the trained model
- `run_training_pipeline()`: Runs complete training pipeline

## Usage

### Method 1: Using Individual Modules

```python
from src.data_preprocessing import DataPreprocessor
from src.model_training import FaceMaskDetector

# Data preprocessing
preprocessor = DataPreprocessor()
train_gen, eval_gen, classes, train_data, eval_data = preprocessor.process_full_pipeline()

# Model training
detector = FaceMaskDetector()
history = detector.train_model(train_gen, eval_gen, classes)
detector.visualize_training_history()
detector.save_model('my_mask_detector.h5')
```

### Method 2: Using the Pipeline Runner

```bash
# Run complete pipeline
python run_pipeline.py --mode complete

# Run only data preprocessing
python run_pipeline.py --mode preprocess

# Run only model training
python run_pipeline.py --mode train
```

### Method 3: Direct Module Execution

```bash
# Run data preprocessing only
python src/data_preprocessing.py

# Run model training only (includes preprocessing)
python src/model_training.py
```

## Configuration

The modules can use a YAML configuration file (`config/config.yaml`) or fall back to default settings:

```yaml
data:
  image_size: [224, 224]
  batch_size: 32
  classes: ['0-NO_MASK', '1-CORRECT_MASK', '2-INCORRECT_MASK']

model:
  input_shape: [224, 224, 3]
  num_classes: 3
  dropout_rate: 0.5

training:
  epochs: 30
  learning_rate: 0.0001
  patience: 10
```

## Data Setup

1. Place your `images.zip` file in the `data/raw/` directory
2. Ensure you have the `train_images_labels.csv` file in `data/processed/`
3. The preprocessing module will automatically organize the data

## Key Features from Original Code

- **Same Architecture**: Uses MobileNetV2 with identical custom layers
- **Same Training Setup**: Maintains original training parameters and augmentation
- **Same Preprocessing**: Preserves original data organization and processing
- **Same Visualizations**: Includes all original plotting functionality

## Dependencies

- TensorFlow/Keras
- OpenCV (cv2)
- pandas
- numpy
- matplotlib
- scikit-learn
- tqdm
- PyYAML

## Model Output

The trained model will be saved as `mask_detector.h5` and can detect:
- 0-NO_MASK: Person not wearing a mask
- 1-CORRECT_MASK: Person wearing mask correctly
- 2-INCORRECT_MASK: Person wearing mask incorrectly

## Paths Configuration

The modules automatically configure paths based on your project structure:
- Raw data: `data/raw/images.zip`
- Processed data: `data/processed/images/`
- Training folders: `data/processed/images/train/`
- Test folder: `data/processed/images/test/`
- Models: Root directory (configurable)

This separation allows for better code organization, easier maintenance, and more flexible usage of individual components.
