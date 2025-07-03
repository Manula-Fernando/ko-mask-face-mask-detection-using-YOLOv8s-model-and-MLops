# Face Mask Detection - Production Data Processor with MLflow Tracking
import os
import zipfile
import logging
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import mlflow
import mlflow.sklearn
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Production-ready data processing pipeline for face mask detection with MLflow tracking."""
    
    CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    IMAGE_SIZE = (224, 224)
    
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.images_dir = raw_data_dir / "images"
        self.annotations_dir = raw_data_dir / "annotations"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLflow
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking for data preprocessing."""
        mlflow.set_experiment("Face_Mask_Detection_Data_Preprocessing")
        
    def log_data_quality_metrics(self, df: pd.DataFrame):
        """Log comprehensive data quality metrics to MLflow."""
        with mlflow.start_run(run_name="data_quality_analysis", nested=True):
            # Basic metrics
            mlflow.log_metric("total_samples", len(df))
            mlflow.log_metric("total_classes", len(self.CLASSES))
            mlflow.log_metric("unique_images", df['filename'].nunique())
            
            # Class distribution
            class_counts = df['class'].value_counts()
            for class_name, count in class_counts.items():
                mlflow.log_metric(f"class_count_{class_name}", count)
                mlflow.log_metric(f"class_percentage_{class_name}", count / len(df) * 100)
            
            # Data quality metrics
            mlflow.log_metric("class_balance_ratio", min(class_counts) / max(class_counts))
            mlflow.log_metric("missing_values", df.isnull().sum().sum())
            mlflow.log_metric("duplicate_images", df['filename'].duplicated().sum())
            
            # Create and log visualizations
            self.create_data_visualizations(df)
            
            # Log dataset metadata
            metadata = {
                'dataset_name': 'Face Mask Detection',
                'data_source': 'Kaggle - Face Mask Detection Dataset',
                'processing_date': datetime.now().isoformat(),
                'classes': self.CLASSES,
                'image_size': self.IMAGE_SIZE,
                'total_samples': len(df),
                'class_distribution': class_counts.to_dict()
            }
            
            mlflow.log_dict(metadata, "dataset_metadata.json")
            
    def create_data_visualizations(self, df: pd.DataFrame):
        """Create comprehensive data visualizations."""
        # Set style for better looking plots
        plt.style.use('seaborn-v0_8')
        
        # 1. Class Distribution Bar Chart
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        class_counts = df['class'].value_counts()
        bars = plt.bar(class_counts.index, class_counts.values, 
                      color=['#2E8B57', '#DC143C', '#FF8C00'])
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Class Distribution Pie Chart
        plt.subplot(2, 2, 2)
        colors = ['#2E8B57', '#DC143C', '#FF8C00']
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        # 3. Data Quality Metrics
        plt.subplot(2, 2, 3)
        quality_metrics = {
            'Total Samples': len(df),
            'Unique Images': df['filename'].nunique(),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicates': df['filename'].duplicated().sum()
        }
        
        bars = plt.bar(quality_metrics.keys(), quality_metrics.values(), 
                      color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
        plt.title('Data Quality Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, quality_metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Class Balance Analysis
        plt.subplot(2, 2, 4)
        balance_ratio = min(class_counts) / max(class_counts)
        imbalance_severity = ['Severe', 'Moderate', 'Mild', 'Balanced']
        severity_colors = ['#F44336', '#FF9800', '#FFC107', '#4CAF50']
        
        if balance_ratio < 0.3:
            severity = 0
        elif balance_ratio < 0.5:
            severity = 1
        elif balance_ratio < 0.8:
            severity = 2
        else:
            severity = 3
            
        plt.bar(['Class Balance'], [balance_ratio], color=severity_colors[severity])
        plt.title(f'Class Balance: {imbalance_severity[severity]}', fontsize=14, fontweight='bold')
        plt.ylabel('Balance Ratio', fontsize=12)
        plt.ylim(0, 1)
        plt.text(0, balance_ratio + 0.05, f'{balance_ratio:.2f}', 
                ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('data_analysis_overview.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('data_analysis_overview.png')
        plt.close()
        
        # Create additional detailed visualizations
        self.create_detailed_visualizations(df)
        
    def create_detailed_visualizations(self, df: pd.DataFrame):
        """Create detailed data analysis visualizations."""
        # Sample images from each class
        plt.figure(figsize=(15, 10))
        
        for i, class_name in enumerate(self.CLASSES):
            class_samples = df[df['class'] == class_name].head(3)
            
            for j, (_, row) in enumerate(class_samples.iterrows()):
                plt.subplot(3, 3, i * 3 + j + 1)
                
                try:
                    image_path = Path(row['image_path'])
                    if image_path.exists():
                        img = cv2.imread(str(image_path))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            plt.imshow(img)
                            plt.title(f'{class_name}\n{row["filename"]}', fontsize=10)
                            plt.axis('off')
                except Exception as e:
                    plt.text(0.5, 0.5, f'Error loading\n{class_name}', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.axis('off')
        
        plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sample_images_by_class.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('sample_images_by_class.png')
        plt.close()
        
    def log_split_analysis(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Log comprehensive split analysis to MLflow."""
        with mlflow.start_run(run_name="data_split_analysis", nested=True):
            # Split sizes
            mlflow.log_metric("train_size", len(train_df))
            mlflow.log_metric("val_size", len(val_df))
            mlflow.log_metric("test_size", len(test_df))
            mlflow.log_metric("total_size", len(train_df) + len(val_df) + len(test_df))
            
            # Split ratios
            total = len(train_df) + len(val_df) + len(test_df)
            mlflow.log_metric("train_ratio", len(train_df) / total)
            mlflow.log_metric("val_ratio", len(val_df) / total)
            mlflow.log_metric("test_ratio", len(test_df) / total)
            
            # Class distributions per split
            splits = {'train': train_df, 'val': val_df, 'test': test_df}
            
            for split_name, split_df in splits.items():
                class_counts = split_df['class'].value_counts()
                for class_name, count in class_counts.items():
                    mlflow.log_metric(f"{split_name}_{class_name}_count", count)
                    mlflow.log_metric(f"{split_name}_{class_name}_percentage", count / len(split_df) * 100)
            
            # Create split visualization
            self.create_split_visualizations(train_df, val_df, test_df)
            
            # Log split metadata
            split_metadata = {
                'split_strategy': 'stratified',
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'train_distribution': train_df['class'].value_counts().to_dict(),
                'val_distribution': val_df['class'].value_counts().to_dict(),
                'test_distribution': test_df['class'].value_counts().to_dict(),
                'split_date': datetime.now().isoformat()
            }
            
            mlflow.log_dict(split_metadata, "split_metadata.json")
            
    def create_split_visualizations(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Create visualizations for data splits."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Split sizes
        splits = {'Train': len(train_df), 'Val': len(val_df), 'Test': len(test_df)}
        colors = ['#4CAF50', '#2196F3', '#FF9800']
        
        axes[0, 0].bar(splits.keys(), splits.values(), color=colors)
        axes[0, 0].set_title('Dataset Split Sizes', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Samples')
        
        # Add value labels
        for i, (split, count) in enumerate(splits.items()):
            axes[0, 0].text(i, count + 10, f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Split ratios pie chart
        axes[0, 1].pie(splits.values(), labels=splits.keys(), autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 1].set_title('Dataset Split Ratios', fontsize=14, fontweight='bold')
        
        # 3. Class distribution across splits
        split_data = []
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            for class_name in self.CLASSES:
                count = len(split_df[split_df['class'] == class_name])
                split_data.append({'Split': split_name, 'Class': class_name, 'Count': count})
        
        split_analysis_df = pd.DataFrame(split_data)
        pivot_df = split_analysis_df.pivot(index='Split', columns='Class', values='Count')
        
        pivot_df.plot(kind='bar', ax=axes[1, 0], color=['#2E8B57', '#DC143C', '#FF8C00'])
        axes[1, 0].set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].legend(title='Class')
        axes[1, 0].tick_params(axis='x', rotation=0)
        
        # 4. Class balance analysis per split
        balance_ratios = []
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            class_counts = split_df['class'].value_counts()
            balance_ratio = min(class_counts) / max(class_counts) if len(class_counts) > 1 else 1.0
            balance_ratios.append(balance_ratio)
        
        bars = axes[1, 1].bar(['Train', 'Val', 'Test'], balance_ratios, color=colors)
        axes[1, 1].set_title('Class Balance by Split', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Balance Ratio')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels
        for bar, ratio in zip(bars, balance_ratios):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('data_split_analysis.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('data_split_analysis.png')
        plt.close()
        
    def extract_dataset(self) -> bool:
        """Extract dataset from zip archive with validation and MLflow tracking."""
        with mlflow.start_run(run_name="dataset_extraction", nested=True):
            archive_path = self.raw_data_dir / "images.zip"
            
            if not archive_path.exists():
                self.logger.error(f"Dataset archive not found: {archive_path}")
                self.logger.info("Download from: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/data")
                mlflow.log_metric("extraction_success", 0)
                return False
                
            # Check if already extracted
            if self.images_dir.exists() and self.annotations_dir.exists():
                image_count = len(list(self.images_dir.glob("*.jpg")))
                annotation_count = len(list(self.annotations_dir.glob("*.xml")))
                if image_count > 800 and annotation_count > 800:
                    self.logger.info(f"Dataset already extracted: {image_count} images, {annotation_count} annotations")
                    mlflow.log_metric("images_found", image_count)
                    mlflow.log_metric("annotations_found", annotation_count)
                    mlflow.log_metric("extraction_success", 1)
                    return True
            
            self.logger.info("Extracting dataset...")
            try:
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_data_dir)
                
                # Count extracted files
                image_count = len(list(self.images_dir.glob("*.jpg")))
                annotation_count = len(list(self.annotations_dir.glob("*.xml")))
                
                mlflow.log_metric("images_extracted", image_count)
                mlflow.log_metric("annotations_extracted", annotation_count)
                mlflow.log_metric("extraction_success", 1)
                
                self.logger.info("✅ Dataset extracted successfully")
                return True
            except Exception as e:
                self.logger.error(f"Extraction failed: {e}")
                mlflow.log_metric("extraction_success", 0)
                mlflow.log_text(str(e), "extraction_error.txt")
                return False
    
    def validate_image(self, image_path: Path) -> bool:
        """Validate if image can be loaded and is valid."""
        try:
            image = cv2.imread(str(image_path))
            return image is not None and image.size > 0
        except Exception:
            return False
    
    def load_annotations(self) -> pd.DataFrame:
        """Load and parse XML annotations into DataFrame with validation and MLflow tracking."""
        with mlflow.start_run(run_name="annotation_loading", nested=True):
            annotations = []
            failed_files = []
            
            xml_files = list(self.annotations_dir.glob("*.xml"))
            mlflow.log_metric("total_xml_files", len(xml_files))
            
            for xml_file in xml_files:
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    filename_elem = root.find('filename')
                    if filename_elem is None or filename_elem.text is None:
                        failed_files.append(f"No filename in {xml_file}")
                        continue
                        
                    filename = filename_elem.text
                    image_path = self.images_dir / filename
                    
                    # Validate image exists and is readable
                    if not image_path.exists() or not self.validate_image(image_path):
                        failed_files.append(f"Invalid/missing image: {filename}")
                        continue
                        
                    # Extract object annotations
                    for obj in root.findall('object'):
                        name_elem = obj.find('name')
                        if name_elem is None or name_elem.text is None:
                            continue
                            
                        class_name = name_elem.text
                        if class_name in self.CLASSES:
                            annotations.append({
                                'filename': filename,
                                'image_path': str(image_path),
                                'class': class_name,
                                'class_id': self.CLASSES.index(class_name)
                            })
                            break  # Take first valid annotation per image
                            
                except Exception as e:
                    failed_files.append(f"Parse error {xml_file}: {e}")
                    continue
            
            df = pd.DataFrame(annotations)
            
            # Log metrics to MLflow
            mlflow.log_metric("valid_annotations", len(df))
            mlflow.log_metric("failed_files", len(failed_files))
            mlflow.log_metric("success_rate", len(df) / len(xml_files) if xml_files else 0)
            
            # Log failed files for debugging
            if failed_files:
                mlflow.log_text("\n".join(failed_files), "failed_files.txt")
            
            self.logger.info(f"Loaded {len(df)} valid annotations")
            
            # Log class distribution
            if len(df) > 0:
                dist = df['class'].value_counts()
                self.logger.info(f"Class distribution: {dict(dist)}")
                
                # Log individual class counts
                for class_name, count in dist.items():
                    mlflow.log_metric(f"raw_class_count_{class_name}", count)
            
            # Perform comprehensive data quality analysis
            if len(df) > 0:
                self.log_data_quality_metrics(df)
            
            return df
    
    def create_splits(self, df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.2, 
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/val/test splits with MLflow tracking."""
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['class']
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=random_state,
            stratify=train_val['class']
        )
        
        self.logger.info(f"Dataset splits: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        # Log split analysis
        self.log_split_analysis(train, val, test)
        
        return train, val, test
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save split metadata and file lists with MLflow tracking."""
        with mlflow.start_run(run_name="data_split_saving", nested=True):
            splits_dir = self.processed_data_dir / "splits"
            splits_dir.mkdir(exist_ok=True)
            
            # Save DataFrames
            train_df.to_csv(splits_dir / "train.csv", index=False)
            val_df.to_csv(splits_dir / "val.csv", index=False)
            test_df.to_csv(splits_dir / "test.csv", index=False)
            
            # Log split files as artifacts
            mlflow.log_artifact(str(splits_dir / "train.csv"))
            mlflow.log_artifact(str(splits_dir / "val.csv"))
            mlflow.log_artifact(str(splits_dir / "test.csv"))
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'total_samples': len(train_df) + len(val_df) + len(test_df),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'classes': self.CLASSES,
                'train_distribution': train_df['class'].value_counts().to_dict(),
                'val_distribution': val_df['class'].value_counts().to_dict(),
                'test_distribution': test_df['class'].value_counts().to_dict()
            }
            
            with open(splits_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Log metadata as artifact
            mlflow.log_artifact(str(splits_dir / "metadata.json"))
            mlflow.log_dict(metadata, "processing_metadata.json")
            
            # Log final metrics
            mlflow.log_metric("final_train_samples", len(train_df))
            mlflow.log_metric("final_val_samples", len(val_df))
            mlflow.log_metric("final_test_samples", len(test_df))
            mlflow.log_metric("processing_success", 1)
            
            self.logger.info(f"✅ Splits saved to {splits_dir}")

def main():
    """Main data preprocessing pipeline with comprehensive MLflow tracking."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"
    
    # Create directories
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = DataProcessor(raw_data_dir, processed_data_dir)
    
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Run complete pipeline with MLflow tracking
    with mlflow.start_run(run_name="complete_data_preprocessing_pipeline"):
        try:
            # Log pipeline parameters
            mlflow.log_param("image_size", processor.IMAGE_SIZE)
            mlflow.log_param("classes", processor.CLASSES)
            mlflow.log_param("raw_data_path", str(raw_data_dir))
            mlflow.log_param("processed_data_path", str(processed_data_dir))
            
            # Step 1: Extract dataset
            print("🔄 Step 1: Extracting dataset...")
            if not processor.extract_dataset():
                print("❌ Dataset extraction failed")
                mlflow.log_metric("pipeline_success", 0)
                return
            
            # Step 2: Load annotations
            print("🔄 Step 2: Loading annotations...")
            df = processor.load_annotations()
            if len(df) == 0:
                print("❌ No valid annotations found")
                mlflow.log_metric("pipeline_success", 0)
                return
            
            # Step 3: Create splits
            print("🔄 Step 3: Creating data splits...")
            train_df, val_df, test_df = processor.create_splits(df)
            
            # Step 4: Save splits
            print("🔄 Step 4: Saving splits...")
            processor.save_splits(train_df, val_df, test_df)
            
            # Log final pipeline metrics
            mlflow.log_metric("total_processed_samples", len(df))
            mlflow.log_metric("pipeline_success", 1)
            mlflow.log_param("pipeline_completion_time", datetime.now().isoformat())
            
            print("✅ Data preprocessing pipeline completed successfully!")
            print(f"📊 Total samples: {len(df)}")
            print(f"📈 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            print("🌐 Check MLflow UI at http://127.0.0.1:5000 for detailed metrics and visualizations")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            mlflow.log_metric("pipeline_success", 0)
            mlflow.log_text(str(e), "pipeline_error.txt")
            raise

if __name__ == "__main__":
    main()
