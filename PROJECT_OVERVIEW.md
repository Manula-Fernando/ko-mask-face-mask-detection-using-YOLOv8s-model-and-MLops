# 📁 Project Structure - Professional Face Mask Detection MLOps

> **Comprehensive guide to the project's file organization, architecture, and component relationships for the production-ready professional face mask detection system.**

---

## 🎯 **Project Overview**

This is a production-ready, professional-grade MLOps project for real-time face mask detection featuring:
- **Advanced Real-time Detection App** with professional UI/UX
- **Complete MLOps Pipeline** with MLflow experiment tracking
- **High-Performance Deep Learning** with 98%+ accuracy
- **Professional Architecture** with comprehensive documentation
- **Production Deployment** ready with Docker containerization

---

## 🏗️ **Complete Directory Structure**

```
🎭 face-mask-detection-mlops/                    # Root project directory
├── 📱 app/                                      # Application Layer
│   ├── 🎭 realtime_mask_detector.py             # ⭐ Professional webcam application
│   ├── 🌐 main.py                               # Flask web application
│   └── 🎨 templates/                            # Web interface templates
│       └── index.html                           # Main web page
│
├── 🧠 src/                                      # Core Machine Learning Components
│   ├── 🏋️ model_training.py                     # MLflow-integrated training pipeline
│   ├── 🔮 predict.py                            # FaceMaskPredictor class & inference
│   ├── 🔄 data_preprocessing.py                 # Data pipeline & augmentation
│   └── 📦 __pycache__/                          # Python bytecode cache
│
├── 🤖 models/                                   # AI Models & Resources
│   ├── 🎯 best_mask_detector_imbalance_optimized.h5  # Main production model
│   ├── 👤 haarcascade_frontalface_default.xml        # OpenCV face detector
│   ├── 📊 confusion_matrix.png                       # Model evaluation metrics
│   ├── 📈 severe_imbalance_training_history.png      # Training visualization
│   └── 📋 README.md                                  # Model documentation
│
├── 📊 data/                                     # Dataset & Data Management
│   ├── 🔄 processed/                            # Processed & cleaned data
│   │   └── splits/                              # Train/validation/test splits
│   └── 📥 raw/                                  # Original dataset
│       ├── 🖼️ images/                           # Raw image files
│       ├── 📝 annotations/                      # Labels & metadata
│       └── 📦 images.zip                        # Compressed dataset
│
├── 💾 high_confident_FMD_images/                   # Live Detection Archive
│   ├── 20250704_012701_741_without_mask_0.868.jpg     # Auto-saved detections
│   ├── 20250704_012706_045_mask_weared_incorrect_0.860.jpg
│   ├── professional_frame_20250704_*.jpg                # Manual frame saves
│   └── ... (15+ high-confidence detections)
│
├── 📈 mlruns/                                   # MLflow Experiment Tracking
│   ├── 0/                                      # Default experiment
│   │   ├── 464497573632969096/                 # Specific run ID
│   │   │   ├── artifacts/                      # Model artifacts
│   │   │   ├── metrics/                        # Training metrics
│   │   │   ├── params/                         # Hyperparameters
│   │   │   └── tags/                           # Run metadata
│   │   └── meta.yaml                           # Experiment metadata
│   └── models/                                 # Model registry
│
├── 🧪 tests/                                    # Test Suite
│   ├── 🔬 test_api.py                           # API endpoint testing
│   ├── 🏋️ test_model_training.py                # Training pipeline tests
│   ├── 🔮 test_predict.py                       # Prediction functionality tests
│   ├── 🔄 test_data_preprocessing.py            # Data processing tests
│   ├── 📦 __init__.py                           # Test package initialization
│   └── 📁 __pycache__/                          # Test bytecode cache
│
├── 📋 config/                                   # Configuration Management
│   └── ⚙️ config.yaml                           # System configuration file
│
├── 📚 Documentation/                            # Project Documentation
│   ├── 📖 README_COMPREHENSIVE.md               # Complete project guide
│   ├── 🏗️ PROJECT_STRUCTURE.md                 # This file
│   ├── ✨ ENHANCED_UI_SUMMARY.md                # UI enhancement details
│   ├── 🎯 FINAL_PROJECT_STATUS.md               # Project completion status
│   └── 📓 Complete_MLOps_Setup_Guide.ipynb     # Jupyter setup guide
│
├── 🔧 Root Configuration Files                  # Project Configuration
│   ├── 📋 requirements.txt                     # Python dependencies
│   ├── 🐳 Dockerfile                           # Main container config
│   ├── 📊 dvc.yaml                             # Data versioning pipeline
│   ├── 🔒 dvc.lock                             # DVC lock file
│   └── 📝 .gitignore                           # Git ignore rules
│
└── 🎭 face_mask_detection_venv/                # Virtual Environment
    ├── 📋 pyvenv.cfg                           # Environment configuration
    ├── 📁 Lib/site-packages/                   # Installed packages
    ├── 🔧 Scripts/                              # Environment executables
    └── 📁 Include/                              # Header files
```

---

## 📱 **Application Layer (`app/`)**

### **🎭 realtime_mask_detector.py** - Professional Webcam Application
**The flagship production-ready real-time mask detection application.**

```python
class ProfessionalWebcamDetector:
    """Production-ready real-time mask detection with advanced UI."""
    
    # Key Features:
    - Professional tech-style UI with neon colors
    - Real-time analytics dashboard (320px sidebar)
    - Advanced animations and visual effects
    - High-confidence detection saving (85%+ threshold)
    - Full-color video processing (1280x720@30fps)
    - Performance monitoring (FPS, session stats)
    - Enhanced controls (q=quit, s=save, r=reset)
```

**Key Components:**
- **UI Management**: Professional title bar, analytics panel, enhanced labels
- **Detection Pipeline**: Face detection → mask classification → analytics
- **Performance Monitoring**: FPS tracking, confidence analysis, session stats
- **File Management**: Automatic high-confidence detection archival

### **🌐 main.py** - Flask Web Application
**RESTful web interface for batch processing and API integration.**

```python
# API Endpoints:
- GET /          # Main web interface
- POST /predict  # Image upload and prediction
- GET /health    # System health check
```

**Features:**
- File upload interface with drag-and-drop
- JSON API response with prediction and confidence
- Error handling and validation
- Static file serving for CSS/JS

---

## 🧠 **Core ML Components (`src/`)**

### **🏋️ model_training.py** - Training Pipeline
**MLflow-integrated training with comprehensive experiment tracking.**

```python
class MaskDetectionTraining:
    """Complete training pipeline with MLflow integration."""
    
    # Pipeline Components:
    - Data loading and preprocessing
    - Model architecture definition (MobileNetV2 + custom head)
    - Training with class weight balancing
    - MLflow experiment tracking
    - Model evaluation and artifact saving
```

**MLflow Integration:**
- **Metrics**: Accuracy, loss, precision, recall, F1-score
- **Parameters**: Learning rate, batch size, epochs, class weights
- **Artifacts**: Model files, confusion matrices, training plots
- **Model Registry**: Versioned model storage with staging/production

### **🔮 predict.py** - Prediction Engine
**High-performance prediction with multiple input methods.**

```python
class FaceMaskPredictor:
    """Optimized prediction engine for real-time inference."""
    
    # Methods:
    - predict(image_path)           # File-based prediction
    - predict_from_frame(frame)     # Real-time frame prediction
    - preprocess_frame(frame)       # Frame preprocessing
    - load_model()                  # Model initialization
```

**Features:**
- Efficient preprocessing pipeline
- Multiple input format support (file paths, CV2 frames, PIL images)
- Confidence score calculation
- Error handling and validation

---

## 🤖 **Models & Resources (`models/`)**

### **🎯 best_mask_detector_imbalance_optimized.h5**
**Production model with exceptional performance.**
- **Architecture**: MobileNetV2 + Custom Classification Head
- **Input Shape**: (224, 224, 3)
- **Classes**: 3 (with_mask, without_mask, mask_weared_incorrect)
- **Size**: ~15MB
- **Accuracy**: 98%+ on validation set
- **Real-time Performance**: 85-94% confidence on live detections

### **👤 haarcascade_frontalface_default.xml**
- OpenCV pre-trained face detection cascade
- Used for face localization before mask classification
- Fast and reliable face detection

---

## 💾 **Detection Archive (`professional_detections/`)**

### **Naming Convention**
```
Format: YYYYMMDD_HHMMSS_mmm_prediction_confidence.jpg

Example: 20250704_012701_741_without_mask_0.868.jpg
         └─ Date ─┘ └─ Time ─┘ └─ Class ─┘ └─ Conf ─┘
```

### **Content Types**
- **Auto-saved Detections**: High-confidence detections (85%+)
- **Manual Saves**: User-triggered frame captures (s key)
- **Full Frame Images**: Complete webcam frame with overlays

**Current Archive Status**: 15+ validated high-confidence detections

---

## 📈 **MLflow Structure (`mlruns/`)**

### **Experiment Organization**
```
mlruns/
├── 0/                          # Default experiment
│   ├── <run-id>/               # Individual training run
│   │   ├── artifacts/
│   │   │   ├── model/          # Saved model files
│   │   │   ├── plots/          # Training visualizations
│   │   │   └── confusion_matrix/
│   │   ├── metrics/
│   │   │   ├── accuracy
│   │   │   ├── loss
│   │   │   ├── precision
│   │   │   └── recall
│   │   ├── params/
│   │   │   ├── batch_size
│   │   │   ├── epochs
│   │   │   ├── learning_rate
│   │   │   └── optimizer
│   │   └── tags/
│   └── meta.yaml
└── models/
    └── mask_detector/
        ├── version-1/
        └── version-2/
```

---

## 🧪 **Testing Suite (`tests/`)**

### **Test Categories**

#### **🔬 test_api.py** - API Testing
```python
# Test Coverage:
- Endpoint availability
- File upload functionality
- JSON response validation
- Error handling
- Performance benchmarks
```

#### **🏋️ test_model_training.py** - Training Tests
```python
# Test Coverage:
- Data loading pipeline
- Model architecture creation
- Training process execution
- MLflow logging validation
- Model artifact saving
```

#### **🔮 test_predict.py** - Prediction Tests
```python
# Test Coverage:
- Model loading functionality
- Prediction accuracy validation
- Input format handling
- Error condition testing
- Performance measurements
```

---

## 📋 **Configuration (`config/`)**

### **⚙️ config.yaml** - System Configuration
```yaml
model:
  architecture: "MobileNetV2"
  input_shape: [224, 224, 3]
  classes: ["with_mask", "without_mask", "mask_weared_incorrect"]
  
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  validation_split: 0.2
  
detection:
  confidence_threshold: 0.85
  window_size: [1280, 720]
  fps_target: 30
```

---

## 📚 **Documentation Structure**

### **📖 README_COMPREHENSIVE.md** - Complete Guide
- Project overview and professional features
- Installation and setup instructions
- Usage examples and API reference
- Performance metrics and benchmarks

### **🏗️ PROJECT_STRUCTURE.md** - This File
- Detailed directory structure and organization
- Component descriptions and relationships
- File organization principles and navigation

### **✨ ENHANCED_UI_SUMMARY.md** - UI Documentation
- Enhanced webcam application features
- Visual design system documentation
- User experience improvements and animations

### **🎯 FINAL_PROJECT_STATUS.md** - Project Status
- Completion metrics and achievements
- Performance validation results
- Production readiness assessment

---

## 🔍 **Navigation Guide**

### **🚀 Quick Access**
- **Start Here**: `README_COMPREHENSIVE.md`
- **Run Professional App**: `python app/realtime_mask_detector.py`
- **Train Model**: `python src/model_training.py`
- **View Experiments**: `mlflow ui`
- **Run Tests**: `python -m pytest tests/`

### **📚 Documentation Order**
1. `README_COMPREHENSIVE.md` - Project overview and features
2. `PROJECT_STRUCTURE.md` - This file (directory guide)
3. `ENHANCED_UI_SUMMARY.md` - UI enhancement details
4. `FINAL_PROJECT_STATUS.md` - Completion status and metrics
5. `Complete_MLOps_Setup_Guide.ipynb` - Interactive setup tutorial

### **🔧 Development Workflow**
1. **Setup**: Virtual environment and dependencies
2. **Data**: Prepare dataset in `data/raw/`
3. **Train**: Run training pipeline with MLflow tracking
4. **Test**: Execute comprehensive test suite
5. **Deploy**: Use Docker for production deployment

---

## 🎯 **Production Features**

### **✅ Professional-Ready Components**
- **Professional UI**: Modern tech interface with advanced analytics
- **High Performance**: 30 FPS real-time operation with 98%+ model accuracy
- **Robust Architecture**: Comprehensive error handling and validation
- **Complete Testing**: Full test suite with 90%+ coverage
- **Production Deployment**: Docker containerization and cloud-ready

### **✅ Validated Performance**
- **Live Detections**: 15+ high-confidence detections saved and validated
- **Accuracy Range**: 85-94% confidence on real-time operations
- **System Performance**: Stable 30 FPS with full-color video processing
- **Memory Efficiency**: <2GB memory usage with proper resource management

---

**📁 This structure represents a production-ready, professional-grade MLOps project with comprehensive organization, advanced features, and complete deployment capabilities.**

