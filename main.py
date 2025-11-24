"""
Main Entry Point: Half Sword AI Agent
ScrimBrain Integration - Modular architecture entry point
"""
from half_sword_ai.core.agent import HalfSwordAgent
from half_sword_ai.config import config
from half_sword_ai.utils.pretty_logger import setup_pretty_logging
from half_sword_ai.utils.safe_logger import setup_safe_logging
import logging
import sys

def main():
    """Main entry point"""
    # Setup safe logging first (handles Windows Unicode issues)
    setup_safe_logging(strip_emojis=None)  # Auto-detect Windows console
    # Then setup pretty logging (will use safe handlers)
    setup_pretty_logging(use_colors=True, use_emojis=not (sys.platform == 'win32' and sys.stdout.encoding in ('cp1252', 'ascii', 'latin-1')))
    logger = logging.getLogger(__name__)
    
    model_type = "DQN (ScrimBrain-style)" if config.USE_DISCRETE_ACTIONS else "PPO (Continuous)"
    
    # Beautiful startup banner
    print("\n" + "="*80)
    print(" " * 20 + "⚔️  HALF SWORD AI AGENT  ⚔️")
    print(" " * 15 + "Autonomous Learning Agent")
    print(" " * 18 + "ScrimBrain Integration")
    print("="*80 + "\n")
    
    logger.info(f"📋 Model Type: {model_type}")
    logger.info(f"📐 Frame Size: {config.CAPTURE_WIDTH}x{config.CAPTURE_HEIGHT} (ScrimBrain standard)")
    logger.info(f"🎬 Frame Stack: {config.FRAME_STACK_SIZE} frames")
    logger.info(f"⏭️  Frame Skip: {config.FRAME_SKIP} (physics stability)")
    print("")
    
    agent = HalfSwordAgent()
    agent.initialize()
    agent.start()

if __name__ == "__main__":
    main()
