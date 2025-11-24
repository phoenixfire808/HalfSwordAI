"""
Interception Driver Installation Helper
Helps install and verify the interception driver for Windows
"""
import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_admin():
    """Check if running as administrator"""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

def check_interception_driver():
    """Check if interception driver is installed"""
    try:
        # Try to import interception Python library
        try:
            from interception import Interception
            interception = Interception()
            if not interception.valid:
                logger.warning("⚠️ Interception driver may not be installed")
                logger.info("   The Python library is installed but driver may be missing")
                return False
            devices = interception.devices
            mouse_devices = [d for d in devices if not d.is_keyboard]
            keyboard_devices = [d for d in devices if d.is_keyboard]
            logger.info(f"✅ Interception driver is installed and working!")
            logger.info(f"   Found {len(mouse_devices)} mouse device(s)")
            logger.info(f"   Found {len(keyboard_devices)} keyboard device(s)")
            return True
        except ImportError:
            logger.warning("❌ Interception Python library not installed")
            logger.info("   Install with: pip install interception-python")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Interception driver may not be installed: {e}")
            logger.info("   The Python library is installed but driver may be missing")
            return False
    except Exception as e:
        logger.error(f"Error checking interception: {e}")
        return False

def install_interception_python():
    """Install interception Python library"""
    logger.info("Installing interception-python library...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "interception-python"])
        logger.info("✅ interception-python installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install interception-python: {e}")
        return False

def download_interception_driver():
    """Download interception driver installer"""
    logger.info("Downloading interception driver...")
    logger.info("")
    logger.info("Please download interception driver manually:")
    logger.info("  1. Visit: https://github.com/oblitum/Interception/releases")
    logger.info("  2. Download the latest 'interception.zip'")
    logger.info("  3. Extract it to a folder")
    logger.info("  4. Run 'install-interception.exe /install' as Administrator")
    logger.info("")
    logger.info("Or use the command line installer:")
    logger.info("  cd command_line_installer")
    logger.info("  install-interception.exe /install")
    return False

def install_interception_driver(driver_path: str = None):
    """Install interception driver (requires admin)"""
    if not check_admin():
        logger.error("❌ Administrator privileges required to install driver")
        logger.info("   Please run this script as Administrator")
        return False
    
    if driver_path is None:
        logger.info("Please provide path to install-interception.exe")
        logger.info("Example: python install_interception.py --driver-path C:\\path\\to\\install-interception.exe")
        return download_interception_driver()
    
    driver_path = Path(driver_path)
    if not driver_path.exists():
        logger.error(f"❌ Driver installer not found: {driver_path}")
        return False
    
    logger.info(f"Installing interception driver from: {driver_path}")
    try:
        result = subprocess.run(
            [str(driver_path), "/install"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✅ Interception driver installed successfully")
        logger.info("   Please restart your computer for changes to take effect")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install driver: {e}")
        logger.error(f"   Output: {e.stdout}")
        logger.error(f"   Error: {e.stderr}")
        return False

def main():
    """Main installation helper"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install Interception Driver for Half Sword AI")
    parser.add_argument("--driver-path", type=str, help="Path to install-interception.exe")
    parser.add_argument("--install-python", action="store_true", help="Install Python library only")
    parser.add_argument("--check-only", action="store_true", help="Only check if installed")
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     Interception Driver Installation Helper               ║
    ║     For Half Sword AI Input Control                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check platform
    if platform.system() != "Windows":
        logger.error("❌ Interception driver is only available on Windows")
        sys.exit(1)
    
    # Check admin if installing driver
    if args.driver_path and not check_admin():
        logger.error("❌ Administrator privileges required")
        logger.info("   Right-click and 'Run as administrator'")
        sys.exit(1)
    
    # Install Python library
    if args.install_python or args.driver_path:
        if not install_interception_python():
            sys.exit(1)
    
    # Install driver
    if args.driver_path:
        if not install_interception_driver(args.driver_path):
            sys.exit(1)
    
    # Check installation
    if check_interception_driver():
        logger.info("")
        logger.info("✅ Interception driver is ready!")
        logger.info("   The Half Sword AI agent can now use kernel-level input control")
    else:
        logger.info("")
        logger.info("⚠️ Interception driver not fully installed")
        logger.info("")
        logger.info("To complete installation:")
        logger.info("  1. Install Python library: pip install interception-python")
        logger.info("  2. Download driver from: https://github.com/oblitum/Interception/releases")
        logger.info("  3. Extract and run: install-interception.exe /install (as Admin)")
        logger.info("  4. Restart your computer")
        logger.info("")
        logger.info("Note: The agent will work without interception using DirectInput fallback")

if __name__ == "__main__":
    main()

