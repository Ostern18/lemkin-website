#!/usr/bin/env python3
"""
Validation script for lemkin-video
"""

import sys
import os
from pathlib import Path
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is 3.10+"""
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required, found {sys.version}")
        return False
    print(f"✅ Python version {sys.version} OK")
    return True

def check_module_import():
    """Check if module can be imported"""
    try:
        import lemkin_video
        print(f"✅ Module lemkin-video imported successfully")
        print(f"   Version: {lemkin_video.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import lemkin-video: {e}")
        return False

def check_cli_available():
    """Check if CLI is available"""
    try:
        result = subprocess.run(["lemkin-video", "--version"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ CLI lemkin-video available")
            print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ CLI lemkin-video failed with code {result.returncode}")
            return False
    except FileNotFoundError:
        print(f"❌ CLI lemkin-video not found")
        return False

def check_core_functionality():
    """Check core functionality"""
    try:
        from lemkin_video.core import *
        print("✅ Core functionality imports OK")

        # TODO: Add module-specific core checks
        # Example:
        # analyzer = CoreAnalyzer()
        # result = analyzer.validate()
        # if result.is_valid:
        #     print("✅ Core validation passed")
        #     return True

        return True
    except Exception as e:
        print(f"❌ Core functionality check failed: {e}")
        return False

def check_dependencies():
    """Check critical dependencies"""
    dependencies = [
        "pydantic",
        "typer",
        "rich",
        "loguru"
    ]

    all_ok = True
    for dep in dependencies:
        spec = importlib.util.find_spec(dep)
        if spec is not None:
            print(f"✅ Dependency {dep} OK")
        else:
            print(f"❌ Missing dependency: {dep}")
            all_ok = False

    return all_ok

def main():
    """Run all validation checks"""
    print("🔍 Validating Video installation...")
    print("=" * 50)

    checks = [
        ("Python Version", check_python_version),
        ("Module Import", check_module_import),
        ("CLI Availability", check_cli_available),
        ("Core Functionality", check_core_functionality),
        ("Dependencies", check_dependencies)
    ]

    passed = 0
    total = len(checks)

    for name, check_func in checks:
        print(f"\n🧪 {name}:")
        if check_func():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Validation Summary: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All validation checks passed! Video is ready to use.")
        return 0
    else:
        print("❌ Some validation checks failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
