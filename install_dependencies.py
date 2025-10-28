#!/usr/bin/env python3
"""
Installation script for Git-backed Cortex dependencies
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install all required dependencies"""
    packages = ["GitPython"]
    
    print("Installing dependencies for Git-backed Cortex...")
    print("=" * 50)
    
    success = all(install_package(pkg) for pkg in packages)
    
    if success:
        print("\n✅ All dependencies installed successfully!")
        print("You can now use the Git-backed Cortex module.")
    else:
        print("\n❌ Some dependencies failed to install.")
        print("Please install them manually: pip install GitPython")

if __name__ == "__main__":
    main()