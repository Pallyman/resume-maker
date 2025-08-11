#!/usr/bin/env python3
"""
Quick test script to verify your setup
"""

import sys
from pathlib import Path

print("🔍 Checking AI Resume Builder Setup...")
print("=" * 50)

# Check Python version
print(f"✅ Python version: {sys.version.split()[0]}")

# Check required packages
packages_ok = True
try:
    import flask
    print(f"✅ Flask installed: {flask.__version__}")
except ImportError:
    print("❌ Flask not installed")
    packages_ok = False

try:
    import flask_cors
    print("✅ Flask-CORS installed")
except ImportError:
    print("❌ Flask-CORS not installed")
    packages_ok = False

try:
    import openai
    print(f"✅ OpenAI installed: {openai.__version__}")
except ImportError:
    print("❌ OpenAI not installed")
    packages_ok = False

try:
    import anthropic
    print(f"✅ Anthropic installed: {anthropic.__version__}")
except ImportError:
    print("❌ Anthropic not installed")
    packages_ok = False

try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    print("✅ python-dotenv installed")
    
    # Check for API keys
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key found in .env")
    elif os.getenv('ANTHROPIC_API_KEY'):
        print("✅ Anthropic API key found in .env")
    else:
        print("⚠️  No API key found - using mock mode")
        print("   Add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env file")
except ImportError:
    print("❌ python-dotenv not installed")
    packages_ok = False

# Check file structure
print("\n📁 Checking file structure...")
base_dir = Path.cwd()

files_to_check = [
    ('backend/app.py', 'Backend application'),
    ('frontend/resume-builder.html', 'Frontend HTML'),
    ('requirements.txt', 'Requirements file'),
    ('.env', 'Environment file (optional)'),
    ('main.py', 'Main entry point (optional)')
]

files_ok = True
for file_path, description in files_to_check:
    full_path = base_dir / file_path
    if full_path.exists():
        print(f"✅ {description}: {file_path}")
    else:
        if '.env' in file_path or 'main.py' in file_path:
            print(f"⚠️  {description}: {file_path} (optional)")
        else:
            print(f"❌ {description}: {file_path} NOT FOUND")
            files_ok = False

# Summary
print("\n" + "=" * 50)
if packages_ok and files_ok:
    print("✅ Setup looks good! You can run the app with:")
    print("   python backend/app.py")
    print("   or")
    print("   python main.py (if you created it)")
else:
    print("⚠️  Some issues found. Please check the messages above.")

print("\n💡 Next steps:")
print("1. Create .env file with your API key (if not done)")
print("2. Run: python backend/app.py")
print("3. Open: http://localhost:5000")
print("=" * 50)