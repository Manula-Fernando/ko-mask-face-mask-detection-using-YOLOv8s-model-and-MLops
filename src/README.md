# 🏥 Medical Face Mask Detection - Source Applications

This directory contains the **production-ready, real-data-only** applications for medical face mask detection and monitoring.

## 🚀 Applications Overview

### 1. 🎥 Real-time Webcam Detection
**File**: `realtime_webcam_app.py`

High-tech medical UI for real-time face mask compliance monitoring.

**Features**:
- ✅ **1310x1010 window size** for optimal viewing
- ✅ **High-tech medical UI theming** with cyan accents and professional panels
- ✅ **Real-time YOLO model predictions** (no fake data)
- ✅ **Medical compliance status indicators**:
  - ✅ COMPLIANT (with_mask)
  - ❌ NON-COMPLIANT (without_mask)  
  - ⚠️ IMPROPER USAGE (mask_weared_incorrect)
- ✅ **Live analytics and statistics**
- ✅ **Automatic detection saving** for high-confidence results
- ✅ **Scanner line effects** and medical-grade UI elements

**Usage**:
```bash
python src/realtime_webcam_app.py
```

**Controls**:
- `Q` or `ESC` - Quit application
- `S` - Save current frame manually
- `R` - Reset analytics

### 2. 🌐 Medical API Server
**File**: `inference/api.py`

FastAPI server providing real medical detection endpoints.

**Features**:
- ✅ **Real YOLO model integration** (no fake predictions)
- ✅ **Production-ready endpoints**:
  - `GET /` - Health check
  - `POST /predict` - Image upload prediction
  - `POST /scan` - Base64 image scanning
  - `GET /recent_activity` - Real detection statistics
- ✅ **Automatic detection logging**
- ✅ **CORS enabled** for frontend integration

**Usage**:
```bash
python src/inference/api.py
```

**Access**: http://localhost:8001/docs for API documentation

### 3. 📊 Medical Monitoring Dashboard
**File**: `monitoring/dashboard.py`

Streamlit dashboard for monitoring API statistics and system performance.

**Features**:
- ✅ **Real-time API statistics** (no fake/sample data)
- ✅ **Medical compliance monitoring**
- ✅ **Detection history visualization**
- ✅ **System performance metrics**
- ✅ **Professional medical theming**

**Usage**:
```bash
streamlit run src/monitoring/dashboard.py
```

**Access**: http://localhost:8502

## 🔧 Core Components

### Predictor (`inference/predictor.py`)
- Real YOLO model integration
- Optimized face detection and mask classification
- Returns structured predictions with confidence scores

### API (`inference/api.py`)  
- FastAPI server with real model endpoints
- Automatic detection logging
- Production-ready error handling

## 🎯 Key Achievements

### ✅ **Real Data Only**
- All applications use **real model predictions**
- **No fake/sample data** anywhere in the src/ directory
- Real detection results, real analytics, real API responses

### ✅ **Full Feature Parity**
- **Matches or exceeds** the app/ directory functionality
- All features from app/ are implemented in src/
- **High-tech medical UI theming** identical to app/realtime_mask_detector.py

### ✅ **Production Ready**
- Proper error handling and logging
- Optimized performance
- Medical-grade UI/UX
- Real-time processing capabilities

## 🏥 Medical UI Theme

The applications feature a **consistent medical-grade interface**:

- **Colors**: Cyan accents (#00FFFF), dark backgrounds, medical status colors
- **Typography**: Professional fonts with clear medical labeling
- **Layout**: Clean panels, structured information display
- **Status Indicators**: Clear compliance/non-compliance messaging
- **Analytics**: Real-time statistics and performance metrics

## 🔄 Data Flow

```
Real Camera/Images → YOLO Model → Predictions → UI Display
                                      ↓
                               Detection Storage → API Endpoints → Dashboard
```

## 📝 Requirements

- Python 3.8+
- OpenCV (`cv2`)
- FastAPI + Uvicorn
- Streamlit
- PyTorch + Ultralytics
- Other dependencies in `requirements.txt`

## 🚀 Quick Start

1. **Start API Server**:
   ```bash
   python src/inference/api.py
   ```

2. **Launch Monitoring Dashboard**:
   ```bash
   streamlit run src/monitoring/dashboard.py
   ```

3. **Run Real-time Webcam**:
   ```bash
   python src/realtime_webcam_app.py
   ```

All applications are now **production-ready** and use **real data only**! 🎉
