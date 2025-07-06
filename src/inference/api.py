#!/usr/bin/env python3
"""
Medical Face Mask Detection API (Unified)

This FastAPI app serves all inference endpoints using the single, unified
FaceMaskPredictor class from predictor.py. All prediction, batch, and model info
requests are handled by this class. This is the only API you need for production.
"""
import base64
import os
import io
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
import logging
import mlflow

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # Go to project root

try:
    from src.inference.predictor import FaceMaskPredictor
    print("✅ Medical predictor module loaded")
except ImportError as e:
    print(f"❌ Could not import FaceMaskPredictor: {e}")
    FaceMaskPredictor = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🔬 Medical Face Mask Detection API",
    description="AI-Powered Healthcare Safety & Compliance Monitoring System",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize the medical predictor
medical_predictor = None
if FaceMaskPredictor:
    try:
        medical_predictor = FaceMaskPredictor()
        print("✅ Medical predictor initialized")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize predictor: {e}")

# Medical Statistics
real_stats = {
    "start_time": datetime.now(),
    "total_scans": 0,
    "successful_detections": 0,
    "with_mask_count": 0,
    "without_mask_count": 0,
    "incorrect_mask_count": 0,
    "average_confidence": 0.0,
    "recent_scans": []
}

# Directories for saving uploaded images and labels in YOLO format
FASTAPI_IMAGES_DIR = Path("data/collected/fastapi/images")
FASTAPI_LABELS_DIR = Path("data/collected/fastapi/labels")
FASTAPI_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FASTAPI_LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Pydantic models

class ScanResponse(BaseModel):
    scan_id: str
    timestamp: str
    detections: List[DetectionResult]
    compliance_status: str
    total_detections: int
    processing_time_ms: float
    annotated_image: str  # base64-encoded
    avg_confidence: float
    risk_level: str
    compliance_score: float
    medical_status: str
    medical_recommendations: str

class SystemStatus(BaseModel):
    system_status: str
    model_loaded: bool
    total_analyses: int
    compliance_rate: float
    average_processing_time_ms: float

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("🔬 Medical Face Mask Detection API starting up...")
    print("🏥 Medical AI System Ready")

@app.get("/", response_class=HTMLResponse)
async def medical_dashboard(request: Request):
    """Medical dashboard with real-time stats"""
    
    # Calculate real-time statistics
    uptime_hours = (datetime.now() - real_stats["start_time"]).total_seconds() / 3600
    model_status = "ONLINE" if medical_predictor and medical_predictor.model else "OFFLINE"
    
    total_people = real_stats["with_mask_count"] + real_stats["without_mask_count"] + real_stats["incorrect_mask_count"]
    compliance_rate = (real_stats["with_mask_count"] / total_people * 100) if total_people > 0 else 0
    
    # Medical Interface HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🔬 Medical Face Mask Detection System</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;400;600&display=swap');
            
            :root {{
                --medical-cyan: #00d4aa;
                --medical-green: #4caf50;
                --medical-blue: #2196f3;
                --medical-red: #f44336;
                --medical-orange: #ff9800;
                --bg-dark: #0a0e1a;
                --bg-card: #1a1f3a;
                --text-primary: #ffffff;
                --text-secondary: #b0bec5;
            }}
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Rajdhani', sans-serif;
                background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-card) 100%);
                color: var(--text-primary);
                min-height: 100vh;
                overflow-x: hidden;
            }}
            
            /* Medical Grid Background */
            body::before {{
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: 
                    linear-gradient(rgba(0, 212, 170, 0.1) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(0, 212, 170, 0.1) 1px, transparent 1px);
                background-size: 30px 30px;
                z-index: -1;
                animation: medicalGridPulse 4s ease-in-out infinite;
            }}
            
            @keyframes medicalGridPulse {{
                0%, 100% {{ opacity: 0.3; }}
                50% {{ opacity: 0.6; }}
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 30px;
                background: linear-gradient(135deg, rgba(0, 212, 170, 0.1), rgba(26, 31, 58, 0.9));
                border: 2px solid var(--medical-cyan);
                border-radius: 20px;
                box-shadow: 0 0 40px rgba(0, 212, 170, 0.3);
            }}
            
            .header h1 {{
                font-family: 'Orbitron', monospace;
                font-size: 3em;
                font-weight: 700;
                margin-bottom: 15px;
                text-shadow: 0 0 20px var(--medical-cyan);
                color: var(--medical-cyan);
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
                margin: 30px 0;
            }}
            
            .stat-card {{
                background: rgba(0, 212, 170, 0.1);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                border: 1px solid rgba(0, 212, 170, 0.3);
                transition: all 0.3s ease;
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0, 212, 170, 0.2);
            }}
            
            .stat-value {{
                font-size: 2.5em;
                font-weight: 700;
                color: var(--medical-cyan);
                font-family: 'Orbitron', monospace;
            }}
            
            .stat-label {{
                font-size: 1.1em;
                color: var(--text-secondary);
                margin-top: 8px;
            }}
            
            .upload-section {{
                background: linear-gradient(135deg, rgba(26, 31, 58, 0.8), rgba(16, 20, 40, 0.9));
                padding: 40px;
                border-radius: 20px;
                border: 2px solid rgba(0, 212, 170, 0.3);
                margin: 30px 0;
            }}
            
            .upload-area {{
                border: 3px dashed rgba(0, 212, 170, 0.5);
                padding: 50px;
                text-align: center;
                border-radius: 15px;
                cursor: pointer;
                transition: all 0.3s ease;
            }}
            
            .upload-area:hover {{
                border-color: var(--medical-cyan);
                background-color: rgba(0, 212, 170, 0.1);
                box-shadow: 0 0 30px rgba(0, 212, 170, 0.3);
            }}
            
            .medical-btn {{
                background: linear-gradient(45deg, var(--medical-cyan), var(--medical-green));
                color: white;
                border: none;
                padding: 15px 35px;
                border-radius: 30px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                margin: 10px;
                transition: all 0.3s ease;
                text-transform: uppercase;
            }}
            
            .medical-btn:hover {{
                transform: translateY(-3px);
                box-shadow: 0 12px 35px rgba(0, 212, 170, 0.5);
            }}
            
            .result-section {{
                background: rgba(26, 31, 58, 0.9);
                padding: 30px;
                border-radius: 15px;
                margin: 20px 0;
                display: none;
            }}
            
            .result-section.show {{
                display: block;
                animation: slideIn 0.5s ease-out;
            }}
            
            @keyframes slideIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 MEDICAL FACE MASK DETECTION</h1>
                <p>AI-Powered Healthcare Safety & Compliance System</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{real_stats['total_scans']}</div>
                    <div class="stat-label">Total Scans</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{compliance_rate:.1f}%</div>
                    <div class="stat-label">Compliance Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{uptime_hours:.1f}h</div>
                    <div class="stat-label">System Uptime</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{model_status}</div>
                    <div class="stat-label">AI Model Status</div>
                </div>
            </div>
            
            <div class="upload-section">
                <h2 style="text-align: center; color: var(--medical-cyan); margin-bottom: 30px;">🏥 MEDICAL IMAGE ANALYSIS</h2>
                
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div style="font-size: 4em; margin-bottom: 20px;">🔬</div>
                    <h3>Upload Medical Image for Analysis</h3>
                    <p>Click to select or drag & drop your image</p>
                </div>
                
                <input type="file" id="fileInput" style="display: none;" accept="image/*" onchange="handleFileSelect(this.files[0])">
                
                <div style="text-align: center; margin-top: 20px;">
                    <button class="medical-btn" onclick="analyzeImage()" id="analyzeBtn" disabled>
                        🔍 ANALYZE COMPLIANCE
                    </button>
                </div>
            </div>
            
            <div class="result-section" id="resultSection">
                <h3 style="color: var(--medical-cyan);">📊 Analysis Results</h3>
                <div id="resultContent"></div>
            </div>
        </div>
        
        <script>
            let selectedFile = null;
            
            function handleFileSelect(file) {{
                if (file && file.type.startsWith('image/')) {{
                    selectedFile = file;
                    document.getElementById('analyzeBtn').disabled = false;
                    document.querySelector('.upload-area').innerHTML = `
                        <div style="font-size: 4em; margin-bottom: 20px;">✅</div>
                        <h3>Image Selected: ${{file.name}}</h3>
                        <p>Ready for medical analysis</p>
                    `;
                }}
            }}
            
            async function analyzeImage() {{
                if (!selectedFile) return;
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                try {{
                    const response = await fetch('/scan', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const result = await response.json();
                    displayResults(result);
                }} catch (error) {{
                    console.error('Error:', error);
                    alert('Error analyzing image');
                }}
            }}
            
            function displayResults(result) {{
                const resultSection = document.getElementById('resultSection');
                const resultContent = document.getElementById('resultContent');
                
                resultContent.innerHTML = `
                    <p><strong>Scan ID:</strong> ${{result.scan_id}}</p>
                    <p><strong>Status:</strong> ${{result.compliance_status}}</p>
                    <p><strong>Detections:</strong> ${{result.total_detections}}</p>
                    <p><strong>Processing Time:</strong> ${{result.processing_time_ms}}ms</p>
                `;
                
                resultSection.classList.add('show');
            }}
        </script>
    </body>
    </html>
    """
    
    return html_content

@app.post("/scan", response_model=ScanResponse)
async def medical_scan(file: UploadFile = File(...)):
    """Perform medical face mask compliance scan and save clean image/label for training. Logs each scan as an MLflow run."""
    start_time = time.time()
    scan_id = str(uuid.uuid4())[:8]
    try:
        # Read and process image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        # Save clean image (no overlays) for training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_filename = f"fastapi_{timestamp}"
        img_filename = f"{base_filename}.jpg"
        label_filename = f"{base_filename}.txt"
        img_path = FASTAPI_IMAGES_DIR / img_filename
        label_path = FASTAPI_LABELS_DIR / label_filename
        cv2.imwrite(str(img_path), image)
        # Update statistics
        real_stats["total_scans"] += 1
        detections = []
        compliance_status = "UNKNOWN"
        avg_conf = 0.0
        risk_level = "LOW"
        compliance_score = 1.0
        medical_status = "NORMAL"
        medical_recommendations = "No action needed."
        annotated_image_b64 = ""
        processing_time = 0.0

        with mlflow.start_run(run_name=f"scan_{scan_id}"):
            mlflow.log_param("scan_id", scan_id)
            mlflow.log_param("timestamp", timestamp)
            mlflow.log_param("image_file", str(img_path))
            if medical_predictor and medical_predictor.model:
                try:
                    result = medical_predictor.predict(image)
                    if result and 'prediction' in result and result['prediction'] != 'no_detection':
                        detection = DetectionResult(
                            class_name=result.get('prediction', 'unknown'),
                            confidence=float(result.get('confidence', 0.0)),
                            bbox=result.get('bbox', [0, 0, 0, 0])
                        )
                        detections.append(detection)
                        # Save YOLO label (class_id x_center y_center width height, normalized)
                        bbox = result.get('bbox', [0, 0, 0, 0])
                        x1, y1, x2, y2 = bbox
                        class_map = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}
                        class_id = class_map.get(result.get('prediction', 'with_mask'), 0)
                        x_center = ((x1 + x2) / 2) / image.shape[1]
                        y_center = ((y1 + y2) / 2) / image.shape[0]
                        width = (x2 - x1) / image.shape[1]
                        height = (y2 - y1) / image.shape[0]
                        with open(label_path, 'w') as f:
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        mlflow.log_param("label_file", str(label_path))
                        mlflow.log_metric("confidence", float(result.get('confidence', 0.0)))
                        avg_conf = float(result.get('confidence', 0.0))
                        # Update statistics based on detection
                        class_name = result.get('prediction', '').lower()
                        if 'with_mask' in class_name:
                            real_stats["with_mask_count"] += 1
                        elif 'without_mask' in class_name:
                            real_stats["without_mask_count"] += 1
                        else:
                            real_stats["incorrect_mask_count"] += 1
                    else:
                        # No detection: save empty label file
                        with open(label_path, 'w') as f:
                            f.write("")
                        mlflow.log_param("label_file", str(label_path))
                        mlflow.log_metric("confidence", 0.0)
                    # Determine compliance status
                    has_compliant = any('with_mask' in d.class_name.lower() for d in detections)
                    has_non_compliant = any('without_mask' in d.class_name.lower() for d in detections)
                    if has_compliant and not has_non_compliant:
                        compliance_status = "COMPLIANT"
                        risk_level = "LOW"
                        compliance_score = 1.0
                        medical_status = "NORMAL"
                        medical_recommendations = "No action needed."
                    elif has_non_compliant:
                        compliance_status = "NON_COMPLIANT"
                        risk_level = "HIGH"
                        compliance_score = 0.0
                        medical_status = "ALERT"
                        medical_recommendations = "Immediate intervention required."
                    else:
                        compliance_status = "PARTIAL_COMPLIANCE"
                        risk_level = "MEDIUM"
                        compliance_score = 0.5
                        medical_status = "WARNING"
                        medical_recommendations = "Check mask fit and coverage."
                    real_stats["successful_detections"] += 1
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    compliance_status = "ANALYSIS_ERROR"
                    risk_level = "UNKNOWN"
                    compliance_score = 0.0
                    medical_status = "ERROR"
                    medical_recommendations = "Analysis failed."
                    mlflow.log_param("error", str(e))
            else:
                # Model unavailable: save empty label file
                with open(label_path, 'w') as f:
                    f.write("")
                compliance_status = "MODEL_UNAVAILABLE"
                risk_level = "UNKNOWN"
                compliance_score = 0.0
                medical_status = "ERROR"
                medical_recommendations = "Model unavailable."
                mlflow.log_param("label_file", str(label_path))
            processing_time = (time.time() - start_time) * 1000
            mlflow.log_metric("processing_time_ms", processing_time)
            mlflow.log_metric("total_detections", len(detections))
            mlflow.log_metric("compliance_status", {"COMPLIANT": 2, "NON_COMPLIANT": 1, "PARTIAL_COMPLIANCE": 0, "MODEL_UNAVAILABLE": -1, "ANALYSIS_ERROR": -2}.get(compliance_status, -3))
            mlflow.log_metric("average_confidence", avg_conf)
            mlflow.log_artifact(str(img_path))
            mlflow.log_artifact(str(label_path))

        # Annotate image with bounding boxes (if any)
        annotated_img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = (0, 255, 0) if 'with_mask' in det.class_name else (0, 0, 255)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, f"{det.class_name} {det.confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Store recent scan for dashboard
        if detections:
            first_detection = detections[0]
            detection_class = first_detection.class_name.lower()
            confidence = first_detection.confidence
        else:
            detection_class = 'no_detection'
            confidence = 0
        recent_scan = {
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Detection": detection_class,
            "Confidence": confidence,
            "Status": compliance_status
        }
        real_stats["recent_scans"].append(recent_scan)
        if len(real_stats["recent_scans"]) > 10:
            real_stats["recent_scans"] = real_stats["recent_scans"][-10:]

        # Update average confidence
        if real_stats["successful_detections"] > 0:
            real_stats["average_confidence"] = (
                (real_stats["average_confidence"] * (real_stats["successful_detections"] - 1) + avg_conf)
                / real_stats["successful_detections"]
            )

        return ScanResponse(
            scan_id=scan_id,
            timestamp=datetime.now().isoformat(),
            detections=detections,
            compliance_status=compliance_status,
            total_detections=len(detections),
            processing_time_ms=processing_time,
            annotated_image=annotated_image_b64,
            avg_confidence=avg_conf,
            risk_level=risk_level,
            compliance_score=compliance_score,
            medical_status=medical_status,
            medical_recommendations=medical_recommendations
        )
    except Exception as e:
        logger.error(f"Medical scan error: {e}")
        raise HTTPException(status_code=500, detail=f"Medical scan failed: {str(e)}")

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """System health status"""
    uptime_minutes = (datetime.now() - real_stats["start_time"]).total_seconds() / 60
    total_subjects = real_stats["with_mask_count"] + real_stats["without_mask_count"] + real_stats["incorrect_mask_count"]
    compliance_rate = real_stats["with_mask_count"] / total_subjects if total_subjects > 0 else 1.0

    # Calculate average processing time
    # For simplicity, you can store and update a running average in real_stats, or calculate from MLflow runs if needed.
    average_processing_time_ms = 0.0
    if real_stats["successful_detections"] > 0:
        # Optionally, keep a running sum in real_stats, or query MLflow for the average.
        # Here, just set to 0.0 as a placeholder.
        pass

    return SystemStatus(
        system_status="ONLINE" if medical_predictor and medical_predictor.model else "LIMITED",
        model_loaded=medical_predictor and medical_predictor.model is not None,
        total_analyses=real_stats["total_scans"],
        compliance_rate=compliance_rate,
        average_processing_time_ms=average_processing_time_ms
    )

@app.get("/stats")
async def get_real_stats():
    """Get real system statistics"""
    uptime_hours = (datetime.now() - real_stats["start_time"]).total_seconds() / 3600
    
    return {
        "system_info": {
            "status": "Medical AI Online" if medical_predictor and medical_predictor.model else "Limited Mode",
            "model_loaded": medical_predictor and medical_predictor.model is not None,
            "uptime_hours": uptime_hours,
            "start_time": real_stats["start_time"].isoformat()
        },
        "detection_stats": {
            "total_scans": real_stats["total_scans"],
            "successful_detections": real_stats["successful_detections"], 
            "with_mask": real_stats["with_mask_count"],
            "without_mask": real_stats["without_mask_count"],
            "incorrect_mask": real_stats["incorrect_mask_count"],
            "average_confidence": real_stats["average_confidence"]
        },
        "recent_activity": real_stats["recent_scans"][-5:],
        "model_info": medical_predictor.get_model_info() if medical_predictor else None
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
