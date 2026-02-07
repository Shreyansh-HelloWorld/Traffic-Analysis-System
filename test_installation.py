"""
Installation Test Script
Run this to verify all dependencies are installed correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    
    print("="*60)
    print("Testing Package Imports...")
    print("="*60)
    
    packages = {
        'streamlit': 'Streamlit',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
        'ultralytics': 'Ultralytics (YOLO)',
        'paddleocr': 'PaddleOCR',
        'paddle': 'PaddlePaddle'
    }
    
    failed = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name:25} - OK")
        except ImportError as e:
            print(f"❌ {name:25} - FAILED")
            failed.append((package, name))
    
    print("="*60)
    
    if failed:
        print("\n⚠️  Some packages failed to import:")
        for package, name in failed:
            print(f"   - {name} ({package})")
        print("\nTry running: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def test_versions():
    """Display versions of key packages"""
    
    print("\n" + "="*60)
    print("Package Versions:")
    print("="*60)
    
    try:
        import streamlit as st
        print(f"Streamlit: {st.__version__}")
    except:
        pass
    
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except:
        pass
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except:
        pass
    
    try:
        import pandas as pd
        print(f"Pandas: {pd.__version__}")
    except:
        pass
    
    try:
        import ultralytics
        print(f"Ultralytics: {ultralytics.__version__}")
    except:
        pass
    
    try:
        import paddle
        print(f"PaddlePaddle: {paddle.__version__}")
    except:
        pass
    
    print("="*60)


def test_models():
    """Check if model files exist"""
    
    print("\n" + "="*60)
    print("Checking Model Files...")
    print("="*60)
    
    import os
    from pathlib import Path
    
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ 'models' directory not found!")
        print("   Please create a 'models' folder and add your .pt files")
        return False
    
    required_models = {
        "models/best2.pt": "ANPR Model (Plate Detection)",
        "models/best.pt": "Vehicle Classification Model"
    }
    
    all_found = True
    
    for model_path, description in required_models.items():
        if Path(model_path).exists():
            size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {description:35} - Found ({size:.1f} MB)")
        else:
            print(f"❌ {description:35} - NOT FOUND")
            all_found = False
    
    print("="*60)
    
    if not all_found:
        print("\n⚠️  Some model files are missing!")
        print("   Please add your YOLO model weights to the 'models' folder")
        return False
    
    return True


def test_python_version():
    """Check Python version"""
    
    print("\n" + "="*60)
    print("Python Version:")
    print("="*60)
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python {version_str}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("⚠️  Warning: Python 3.9 or higher is recommended")
        return False
    else:
        print("✅ Python version is compatible")
        return True
    
    print("="*60)


def main():
    """Run all tests"""
    
    print("\n")
    print("🔧 TRAFFIC ANALYSIS SYSTEM - INSTALLATION TEST")
    print("="*60)
    
    # Test Python version
    python_ok = test_python_version()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test versions
    if imports_ok:
        test_versions()
    
    # Test models
    models_ok = test_models()
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if python_ok and imports_ok and models_ok:
        print("✅ All tests passed!")
        print("\nYou're ready to run the application:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        
        if not python_ok:
            print("\n1. Install Python 3.9 or higher")
        
        if not imports_ok:
            print("\n2. Install dependencies:")
            print("   pip install -r requirements.txt")
        
        if not models_ok:
            print("\n3. Add model files to 'models' folder:")
            print("   - models/best2.pt (ANPR model)")
            print("   - models/best.pt (Vehicle model)")
    
    print("="*60)


if __name__ == "__main__":
    main()
