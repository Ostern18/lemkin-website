#!/usr/bin/env python3
"""
Script to create all missing scripts and notebooks for Lemkin modules.
"""

import os
from pathlib import Path
import json

# Base directory
BASE_DIR = Path(__file__).parent

# Template for setup.sh scripts
SETUP_SCRIPT_TEMPLATE = '''#!/bin/bash
# Setup script for {module_name}

set -e

echo "Setting up {module_title}..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )(.+)')
required_version="3.10"
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "Error: Python 3.10+ is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "Installing {module_name} in development mode..."
pip install -e ".[dev]"

# Verify installation
echo "Verifying installation..."
{module_cli} --version

# Run basic tests if they exist
if [ -f "tests/test_core.py" ]; then
    echo "Running basic tests..."
    pytest tests/test_core.py -v
    echo "✓ Tests passed"
fi

echo ""
echo "🎉 {module_title} setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the CLI:"
echo "  {module_cli} --help"
echo ""
echo "To run tests:"
echo "  make test"
echo ""
echo "To see all available commands:"
echo "  make help"
'''

# Template for Jupyter notebooks
NOTEBOOK_TEMPLATE = '''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {module_title} Demo\\n",
    "\\n",
    "This notebook demonstrates the core functionality of the {module_title} module.\\n",
    "\\n",
    "## 🔐 Safety Notice\\n",
    "\\n",
    "⚠️ **IMPORTANT**: This tool is designed for legitimate legal investigation purposes only.\\n",
    "- Use responsibly and within legal boundaries\\n",
    "- Respect privacy and confidentiality\\n",
    "- Verify results before drawing conclusions\\n",
    "- Maintain evidence integrity throughout analysis\\n",
    "\\n",
    "## 📋 Prerequisites\\n",
    "\\n",
    "- Python 3.10+\\n",
    "- {module_name} package installed\\n",
    "- Required dependencies (see requirements)\\n",
    "\\n",
    "## 🚀 Setup"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Install package if not already installed\\n",
    "# !pip install {module_name}\\n",
    "\\n",
    "# Import required libraries\\n",
    "import sys\\n",
    "import os\\n",
    "from pathlib import Path\\n",
    "import json\\n",
    "from datetime import datetime\\n",
    "\\n",
    "# Add src to path for development\\n",
    "sys.path.insert(0, str(Path.cwd().parent / 'src'))\\n",
    "\\n",
    "# Import module components\\n",
    "try:\\n",
    "    from {module_import} import __version__\\n",
    "    from {module_import}.core import *\\n",
    "    print(f\\"✅ Successfully imported {module_name} v{{__version__}}\\")\\n",
    "except ImportError as e:\\n",
    "    print(f\\"❌ Import error: {{e}}\\")\\n",
    "    print(\\"Please ensure the module is properly installed\\")\\n",
    "    sys.exit(1)"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 🔍 Basic Functionality\\n",
    "\\n",
    "Let's start with basic operations to verify everything is working."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Test basic functionality\\n",
    "print(\\"Testing basic {module_name} functionality...\\")\\n",
    "\\n",
    "# TODO: Add module-specific basic tests\\n",
    "# Example:\\n",
    "# analyzer = CoreAnalyzer()\\n",
    "# result = analyzer.basic_test()\\n",
    "# print(f\\"Result: {{result}}\\")\\n",
    "\\n",
    "print(\\"✅ Basic functionality test completed\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 📊 {demo_section_1}\\n",
    "\\n",
    "Demonstrate key features and capabilities."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# {demo_section_1} demonstration\\n",
    "print(\\"Demonstrating {demo_section_1_lower}...\\")\\n",
    "\\n",
    "# TODO: Add specific demonstration code\\n",
    "# This will vary by module - examples:\\n",
    "# - For audio: transcribe a sample file\\n",
    "# - For images: detect manipulation in a sample image\\n",
    "# - For timeline: extract temporal references from text\\n",
    "# - For integrity: generate hash for a sample file\\n",
    "\\n",
    "print(\\"✅ {demo_section_1} demonstration completed\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 🔧 {demo_section_2}\\n",
    "\\n",
    "Advanced features and configuration options."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# {demo_section_2} demonstration\\n",
    "print(\\"Demonstrating {demo_section_2_lower}...\\")\\n",
    "\\n",
    "# TODO: Add advanced feature demonstration\\n",
    "# Show configuration options, advanced parameters, etc.\\n",
    "\\n",
    "print(\\"✅ {demo_section_2} demonstration completed\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 📈 Results Analysis\\n",
    "\\n",
    "Analyze and visualize the results."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Results analysis and visualization\\n",
    "print(\\"Analyzing results...\\")\\n",
    "\\n",
    "# TODO: Add result analysis code\\n",
    "# Include visualizations, metrics, summaries\\n",
    "\\n",
    "print(\\"✅ Results analysis completed\\")"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 🏁 Summary\\n",
    "\\n",
    "This notebook demonstrated the core capabilities of {module_title}:\\n",
    "\\n",
    "- ✅ Basic functionality verification\\n",
    "- ✅ {demo_section_1} demonstration\\n",
    "- ✅ {demo_section_2} showcase\\n",
    "- ✅ Results analysis\\n",
    "\\n",
    "## 📚 Next Steps\\n",
    "\\n",
    "1. **Explore CLI**: Try `{module_cli} --help`\\n",
    "2. **Read Documentation**: Check the README.md file\\n",
    "3. **Run Tests**: Execute `make test`\\n",
    "4. **Integration**: Combine with other Lemkin modules\\n",
    "\\n",
    "## ⚖️ Legal Compliance\\n",
    "\\n",
    "Remember to:\\n",
    "- Use only for legitimate legal purposes\\n",
    "- Maintain evidence integrity\\n",
    "- Protect sensitive information\\n",
    "- Follow applicable laws and regulations\\n",
    "\\n",
    "---\\n",
    "\\n",
    "**{module_title}** - Part of the Lemkin AI Legal Investigation Platform\\n",
    "\\n",
    "🔗 [GitHub Repository](https://github.com/lemkin-org/lemkin-ai)  \\n",
    "📧 [Support](mailto:support@lemkin.org)  \\n",
    "📖 [Documentation](https://docs.lemkin.org)"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "codemirror_mode": {{
    "name": "ipython",
    "version": 3
   }},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}'''

# Template for validation scripts
VALIDATION_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""
Validation script for {module_name}
"""

import sys
import os
from pathlib import Path
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is 3.10+"""
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required, found {{sys.version}}")
        return False
    print(f"✅ Python version {{sys.version}} OK")
    return True

def check_module_import():
    """Check if module can be imported"""
    try:
        import {module_import}
        print(f"✅ Module {module_name} imported successfully")
        print(f"   Version: {{{module_import}.__version__}}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {{e}}")
        return False

def check_cli_available():
    """Check if CLI is available"""
    try:
        result = subprocess.run(["{module_cli}", "--version"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ CLI {module_cli} available")
            print(f"   Output: {{result.stdout.strip()}}")
            return True
        else:
            print(f"❌ CLI {module_cli} failed with code {{result.returncode}}")
            return False
    except FileNotFoundError:
        print(f"❌ CLI {module_cli} not found")
        return False

def check_core_functionality():
    """Check core functionality"""
    try:
        from {module_import}.core import *
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
        print(f"❌ Core functionality check failed: {{e}}")
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
            print(f"✅ Dependency {{dep}} OK")
        else:
            print(f"❌ Missing dependency: {{dep}}")
            all_ok = False

    return all_ok

def main():
    """Run all validation checks"""
    print("🔍 Validating {module_title} installation...")
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
        print(f"\\n🧪 {{name}}:")
        if check_func():
            passed += 1

    print("\\n" + "=" * 50)
    print(f"📊 Validation Summary: {{passed}}/{{total}} checks passed")

    if passed == total:
        print("🎉 All validation checks passed! {module_title} is ready to use.")
        return 0
    else:
        print("❌ Some validation checks failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''

def get_demo_sections(module_name):
    """Get appropriate demo sections for each module"""
    demos = {
        'audio': ('Audio Transcription', 'Speaker Analysis'),
        'video': ('Video Authentication', 'Deepfake Detection'),
        'images': ('Image Verification', 'Manipulation Detection'),
        'geo': ('Coordinate Processing', 'Event Correlation'),
        'forensics': ('File Analysis', 'Network Processing'),
        'osint': ('Data Collection', 'Source Verification'),
        'integrity': ('Hash Generation', 'Chain of Custody'),
        'redaction': ('PII Detection', 'Multi-format Redaction'),
        'classifier': ('Document Classification', 'Batch Processing'),
        'ner': ('Entity Extraction', 'Entity Linking'),
        'timeline': ('Temporal Extraction', 'Timeline Construction'),
        'frameworks': ('Legal Analysis', 'Framework Mapping'),
        'ocr': ('Document Processing', 'Layout Analysis'),
        'research': ('Case Law Search', 'Citation Analysis'),
        'comms': ('Communication Analysis', 'Pattern Detection'),
        'dashboard': ('Data Visualization', 'Interactive Dashboards'),
        'reports': ('Report Generation', 'Legal Formatting'),
        'export': ('Format Conversion', 'Compliance Export')
    }

    module_key = module_name.split('-')[1] if '-' in module_name else module_name
    return demos.get(module_key, ('Core Functionality', 'Advanced Features'))

def create_setup_script(module_path, module_name):
    """Create setup.sh script"""
    scripts_dir = module_path / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    setup_file = scripts_dir / "setup.sh"

    if not setup_file.exists():
        module_title = ' '.join(word.capitalize() for word in module_name.split('-')[1:])
        module_cli = module_name

        content = SETUP_SCRIPT_TEMPLATE.format(
            module_name=module_name,
            module_title=module_title,
            module_cli=module_cli
        )

        setup_file.write_text(content)
        setup_file.chmod(0o755)  # Make executable
        print(f"✓ Created {setup_file}")
    else:
        print(f"  Already exists: {setup_file}")

def create_validation_script(module_path, module_name):
    """Create validation script"""
    scripts_dir = module_path / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    validation_file = scripts_dir / "validate_installation.py"

    if not validation_file.exists():
        module_title = ' '.join(word.capitalize() for word in module_name.split('-')[1:])
        module_import = module_name.replace('-', '_')
        module_cli = module_name

        content = VALIDATION_SCRIPT_TEMPLATE.format(
            module_name=module_name,
            module_title=module_title,
            module_import=module_import,
            module_cli=module_cli
        )

        validation_file.write_text(content)
        validation_file.chmod(0o755)  # Make executable
        print(f"✓ Created {validation_file}")
    else:
        print(f"  Already exists: {validation_file}")

def create_demo_notebook(module_path, module_name):
    """Create demo notebook"""
    notebooks_dir = module_path / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)

    notebook_file = notebooks_dir / f"{module_name}_demo.ipynb"

    if not notebook_file.exists():
        module_title = ' '.join(word.capitalize() for word in module_name.split('-')[1:])
        module_import = module_name.replace('-', '_')
        module_cli = module_name

        demo_section_1, demo_section_2 = get_demo_sections(module_name)

        content = NOTEBOOK_TEMPLATE.format(
            module_name=module_name,
            module_title=module_title,
            module_import=module_import,
            module_cli=module_cli,
            demo_section_1=demo_section_1,
            demo_section_2=demo_section_2,
            demo_section_1_lower=demo_section_1.lower(),
            demo_section_2_lower=demo_section_2.lower()
        )

        notebook_file.write_text(content)
        print(f"✓ Created {notebook_file}")
    else:
        print(f"  Already exists: {notebook_file}")

def complete_module_scripts_notebooks(module_name):
    """Complete scripts and notebooks for a module"""
    module_path = BASE_DIR / module_name

    if not module_path.exists():
        print(f"✗ Module {module_name} does not exist")
        return

    print(f"\\n=== Completing {module_name} Scripts & Notebooks ===")

    # Create setup script
    create_setup_script(module_path, module_name)

    # Create validation script
    create_validation_script(module_path, module_name)

    # Create demo notebook
    create_demo_notebook(module_path, module_name)

def main():
    """Complete all modules with missing scripts and notebooks"""

    # All Lemkin modules
    modules = [
        'lemkin-integrity', 'lemkin-redaction', 'lemkin-classifier',
        'lemkin-ner', 'lemkin-timeline', 'lemkin-frameworks',
        'lemkin-osint', 'lemkin-geo', 'lemkin-forensics',
        'lemkin-video', 'lemkin-images', 'lemkin-audio',
        'lemkin-ocr', 'lemkin-research', 'lemkin-comms',
        'lemkin-dashboard', 'lemkin-reports', 'lemkin-export'
    ]

    print("Starting scripts and notebooks completion process...")

    for module_name in modules:
        complete_module_scripts_notebooks(module_name)

    print("\\n✅ Scripts and notebooks completion process finished!")
    print("\\nNext steps:")
    print("1. Review generated files for module-specific adjustments")
    print("2. Run setup scripts: cd <module>/ && ./scripts/setup.sh")
    print("3. Validate installations: python scripts/validate_installation.py")
    print("4. Open demo notebooks in Jupyter")
    print("5. Commit changes to repository")

if __name__ == "__main__":
    main()