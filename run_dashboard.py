#!/usr/bin/env python3
"""
Wealth Analysis Dashboard Launcher

Main entry point for launching the comprehensive wealth analysis dashboard.
This script sets up the environment and starts the Streamlit application.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        print("✓ Core dependencies found")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def launch_dashboard(port=8501, debug=False):
    """Launch the Streamlit dashboard"""
    if not check_dependencies():
        return False
    
    dashboard_path = src_path / "visualization" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"✗ Dashboard file not found: {dashboard_path}")
        return False
    
    print(f"🚀 Launching Wealth Analysis Dashboard on port {port}")
    print(f"📊 Dashboard will be available at: http://localhost:{port}")
    
    cmd = [
        "streamlit", "run", str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", "false" if debug else "true",
        "--server.runOnSave", "true" if debug else "false",
        "--theme.base", "light"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to launch dashboard: {e}")
        return False
    except FileNotFoundError:
        print("✗ Streamlit not found. Please install with: pip install streamlit")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Launch the Wealth Analysis Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dashboard.py                    # Launch on default port 8501
  python run_dashboard.py --port 8080       # Launch on port 8080
  python run_dashboard.py --debug           # Launch in debug mode
        """
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to run the dashboard on (default: 8501)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Run in debug mode with auto-reload"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Only check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    if args.check_deps:
        if check_dependencies():
            print("✓ All dependencies are installed")
            sys.exit(0)
        else:
            sys.exit(1)
    
    print("=" * 60)
    print("🏦 COMPREHENSIVE WEALTH ANALYSIS FRAMEWORK")
    print("=" * 60)
    print()
    print("Features:")
    print("• Wealth Creation Models (Business, Investment, Employment)")
    print("• Wealth Distribution Analysis (Inequality, Social Mobility)")
    print("• Economic Modeling (Market Dynamics, Policy Impact)")
    print("• AI Impact Analysis (Automation, Digital Economy)")
    print("• Portfolio Management (Optimization, Risk Management)")
    print("• Lifecycle Planning (Accumulation, Retirement)")
    print("• Interactive Visualizations & Scenario Analysis")
    print()
    
    success = launch_dashboard(port=args.port, debug=args.debug)
    
    if not success:
        print("\n❌ Failed to launch dashboard")
        sys.exit(1)

if __name__ == "__main__":
    main()