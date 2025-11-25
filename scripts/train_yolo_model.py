"""
Train YOLO Model on Half Sword Dataset
Uses the converted Roboflow dataset
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from half_sword_ai.perception.yolo_detector import YOLODetector
from half_sword_ai.config import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    """Train YOLO model on Half Sword dataset"""
    dataset_path = project_root / "data" / "yolo_dataset" / "dataset.yaml"
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Run: python scripts/convert_roboflow_to_yolo.py first")
        return False
    
    logger.info("=" * 80)
    logger.info("Training YOLO Model on Half Sword Dataset")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info("=" * 80)
    
    try:
        # Initialize detector
        detector = YOLODetector()
        
        # Train model
        logger.info("Starting training...")
        results_dir = detector.train_custom_model(
            dataset_path=str(dataset_path),
            epochs=100  # Adjust as needed
        )
        
        if results_dir:
            logger.info("=" * 80)
            logger.info("Training Complete!")
            logger.info(f"Model saved to: {results_dir}")
            logger.info("=" * 80)
            logger.info("\nTo use the trained model:")
            logger.info(f"  Update config.YOLO_MODEL_PATH to: {results_dir}/weights/best.pt")
            logger.info("  Set config.YOLO_USE_CUSTOM_MODEL = True")
            return True
        else:
            logger.error("Training failed")
            return False
            
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)

