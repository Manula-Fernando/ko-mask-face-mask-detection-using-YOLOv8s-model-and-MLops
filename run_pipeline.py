"""
Example script demonstrating how to use the separated data preprocessing and model training modules.
This script shows how to run the complete face mask detection pipeline.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.model_training import FaceMaskDetector

def run_complete_pipeline():
    """Run the complete face mask detection pipeline."""
    print("="*60)
    print("FACE MASK DETECTION - COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Data Preprocessing
    print("\n[STEP 1] Starting Data Preprocessing...")
    preprocessor = DataPreprocessor()
    
    try:
        # Run the complete preprocessing pipeline
        train_gen, eval_gen, classes, train_data, eval_data = preprocessor.process_full_pipeline()
        
        if train_gen is None:
            print("❌ Data preprocessing failed!")
            return
        
        print("✅ Data preprocessing completed successfully!")
        print(f"   - Classes: {classes}")
        print(f"   - Training samples: {train_gen.n}")
        print(f"   - Validation samples: {eval_gen.n}")
        
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        return
    
    # Step 2: Model Training
    print("\n[STEP 2] Starting Model Training...")
    detector = FaceMaskDetector()
    
    try:
        # Train the model using the preprocessed data
        history = detector.train_model(train_gen, eval_gen, classes)
        
        # Visualize training results
        detector.visualize_training_history()
        
        # Save the model
        detector.save_model('face_mask_detector_final.h5')
        
        print("✅ Model training completed successfully!")
        
        # Display final results
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        print(f"   - Final Training Accuracy: {final_train_acc:.4f}")
        print(f"   - Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"   - Final Training Loss: {final_train_loss:.4f}")
        print(f"   - Final Validation Loss: {final_val_loss:.4f}")
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Your face mask detection model is ready!")
    print("Model saved as: face_mask_detector_final.h5")

def run_preprocessing_only():
    """Run only the data preprocessing part."""
    print("="*60)
    print("FACE MASK DETECTION - DATA PREPROCESSING ONLY")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    
    try:
        # Run preprocessing
        train_gen, eval_gen, classes, train_data, eval_data = preprocessor.process_full_pipeline()
        
        if train_gen is not None:
            print("✅ Data preprocessing completed!")
            print(f"   - Classes: {classes}")
            print(f"   - Training samples: {train_gen.n}")
            print(f"   - Validation samples: {eval_gen.n}")
        else:
            print("❌ Data preprocessing failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def run_training_only():
    """Run only the model training part (assumes data is already preprocessed)."""
    print("="*60)
    print("FACE MASK DETECTION - MODEL TRAINING ONLY")
    print("="*60)
    
    detector = FaceMaskDetector()
    
    try:
        # Run the complete training pipeline (includes data preprocessing)
        results = detector.run_training_pipeline()
        
        if results:
            print("✅ Training completed successfully!")
            for key, value in results.items():
                if key != 'classes':
                    print(f"   - {key}: {value:.4f}")
                else:
                    print(f"   - {key}: {value}")
        else:
            print("❌ Training failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Mask Detection Pipeline')
    parser.add_argument('--mode', choices=['complete', 'preprocess', 'train'], 
                       default='complete', help='Pipeline mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'complete':
        run_complete_pipeline()
    elif args.mode == 'preprocess':
        run_preprocessing_only()
    elif args.mode == 'train':
        run_training_only()
