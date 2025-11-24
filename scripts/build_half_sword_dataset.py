"""
Script to build comprehensive Half Sword dataset
Usage: python scripts/build_half_sword_dataset.py [--name DATASET_NAME] [--output OUTPUT_DIR] [--fps FPS]
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from half_sword_ai.tools.half_sword_dataset_builder import HalfSwordDatasetBuilder
from half_sword_ai.utils.logger import setup_logger
import logging

def main():
    """Main entry point"""
    import argparse
    
    # Setup logging
    logger = setup_logger(__name__)
    
    parser = argparse.ArgumentParser(description="Build Enhanced Half Sword Dataset")
    parser.add_argument("--name", type=str, help="Dataset name (default: auto-generated)")
    parser.add_argument("--output", type=str, help="Output directory (default: data/half_sword_datasets/)")
    parser.add_argument("--fps", type=int, default=60, help="Recording FPS (default: 60)")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("HALF SWORD COMPREHENSIVE DATASET BUILDER")
    logger.info("=" * 80)
    logger.info("This will record:")
    logger.info("  - Physics data (CoM, support polygon, joint states)")
    logger.info("  - HEMA pose classification (Fiore guards)")
    logger.info("  - Edge alignment calculations")
    logger.info("  - Gap targeting (armor weak points)")
    logger.info("  - Weapon state (grip: standard/half-sword/mordhau)")
    logger.info("  - Historical reward shaping")
    logger.info("=" * 80)
    
    try:
        builder = HalfSwordDatasetBuilder(output_dir=args.output, dataset_name=args.name)
        builder.run_recording_loop(target_fps=args.fps)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Dataset collection interrupted by user")
    except Exception as e:
        logger.error(f"Error during dataset collection: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

