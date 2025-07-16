"""
Training execution script - Main entry point for model training
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.service import TrainingService
from src.common.logger import get_logger

logger = get_logger("training.train")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Face Mask Detection Model")
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--prepare-data', 
        action='store_true',
        help='Prepare data before training'
    )
    
    parser.add_argument(
        '--force-convert', 
        action='store_true',
        help='Force data conversion even if already exists'
    )
    
    parser.add_argument(
        '--experiment-name', 
        type=str,
        help='MLflow experiment name'
    )
    
    parser.add_argument(
        '--run-name', 
        type=str,
        help='MLflow run name'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=35,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=16,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--patience', 
        type=int, 
        default=15,
        help='Early stopping patience'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=0.01,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='yolov8s.pt',
        help='Base model to use'
    )
    
    parser.add_argument(
        '--export', 
        action='store_true',
        help='Export model after training'
    )
    
    parser.add_argument(
        '--export-format', 
        type=str, 
        default='onnx',
        choices=['onnx', 'torchscript', 'tensorflow'],
        help='Export format'
    )
    
    parser.add_argument(
        '--cleanup', 
        action='store_true',
        help='Clean up old runs after training'
    )
    
    return parser.parse_args()

def main():
    """Main training function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        logger.info("Starting Face Mask Detection training pipeline")
        logger.info(f"Configuration: {args.config}")
        
        # Initialize training service
        training_service = TrainingService(args.config)
        
        # Check training status
        status = training_service.get_training_status()
        logger.info(f"Training status: {status}")
        
        # Prepare data if requested or if not prepared
        if args.prepare_data or not status.get('data_prepared', False):
            logger.info("Preparing training data...")
            if not training_service.prepare_data(force_convert=args.force_convert):
                logger.error("Data preparation failed")
                return False
        
        # Setup training parameters
        training_params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'patience': args.patience,
            'learning_rate': args.learning_rate
        }
        
        logger.info(f"Training parameters: {training_params}")
        
        # Train model
        results = training_service.train_model(
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            **training_params
        )
        
        if not results['success']:
            logger.error(f"Training failed: {results.get('error', 'Unknown error')}")
            return False
        
        logger.info("Training completed successfully!")
        logger.info(f"Run ID: {results['run_id']}")
        logger.info(f"Model path: {results['model_path']}")
        logger.info(f"Training metrics: {results['train_metrics']}")
        logger.info(f"Validation metrics: {results['val_metrics']}")
        
        # Export model if requested
        if args.export and results['model_path']:
            logger.info(f"Exporting model to {args.export_format} format...")
            export_results = training_service.export_model(
                model_path=results['model_path'],
                export_format=args.export_format
            )
            
            if export_results['success']:
                logger.info(f"Model exported to: {export_results['export_path']}")
            else:
                logger.warning(f"Model export failed: {export_results.get('error', 'Unknown error')}")
        
        # Cleanup old runs if requested
        if args.cleanup:
            logger.info("Cleaning up old training runs...")
            training_service.cleanup_old_runs(keep_last_n=5)
        
        logger.info("Training pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
