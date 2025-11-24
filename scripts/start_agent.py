#!/usr/bin/env python
"""
Half Sword AI Agent - Direct Launcher
Double-click this file or run: python start_agent.py
"""
import sys
import os
import subprocess

def main():
    """Launch the Half Sword AI Agent"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to project root directory (one level up from scripts/)
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Check if main.py exists
    main_py = os.path.join(script_dir, 'main.py')
    if not os.path.exists(main_py):
        print("ERROR: main.py not found!")
        print(f"Looking in: {script_dir}")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("=" * 80)
    print("  Half Sword AI Agent - Starting...")
    print("=" * 80)
    print()
    print("Press Ctrl+C to stop the agent")
    print("Press F8 for emergency kill switch")
    print()
    print("=" * 80)
    print()
    
    try:
        # Import and run the agent
        from half_sword_ai.core.agent import HalfSwordAgent
        
        agent = HalfSwordAgent()
        agent.initialize()
        agent.start()
        
    except KeyboardInterrupt:
        print("\n\nAgent stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()

