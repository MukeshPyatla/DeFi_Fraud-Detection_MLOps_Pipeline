#!/usr/bin/env python3
"""
Quick deployment script for Streamlit Cloud.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_git_status():
    """Check if git is initialized and has commits."""
    try:
        result = subprocess.run("git status", shell=True, capture_output=True, text=True)
        if "not a git repository" in result.stderr:
            return False
        return True
    except:
        return False

def check_required_files():
    """Check if required files exist."""
    required_files = [
        "streamlit_app.py",
        "requirements_streamlit.txt",
        ".streamlit/config.toml"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    return missing_files

def setup_git():
    """Initialize git repository if not already done."""
    if not check_git_status():
        print("📁 Initializing git repository...")
        run_command("git init", "Initializing git")
        run_command("git add .", "Adding files to git")
        run_command('git commit -m "Initial commit: DeFi Fraud Detection Dashboard"', "Making initial commit")
        return True
    else:
        print("✅ Git repository already exists")
        return True

def create_streamlit_config():
    """Create .streamlit directory and config if it doesn't exist."""
    streamlit_dir = Path(".streamlit")
    if not streamlit_dir.exists():
        streamlit_dir.mkdir()
        print("✅ Created .streamlit directory")
    
    config_file = streamlit_dir / "config.toml"
    if not config_file.exists():
        print("⚠️  .streamlit/config.toml not found. Please create it manually.")
        return False
    
    return True

def main():
    """Main deployment function."""
    print("🚀 Streamlit Cloud Deployment Setup")
    print("=" * 50)
    
    # Check required files
    missing_files = check_required_files()
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present before deploying.")
        return False
    
    print("✅ All required files found")
    
    # Setup git
    if not setup_git():
        print("❌ Failed to setup git repository")
        return False
    
    # Create Streamlit config
    if not create_streamlit_config():
        print("❌ Failed to create Streamlit configuration")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Create a GitHub repository")
    print("2. Push your code to GitHub:")
    print("   git remote add origin https://github.com/yourusername/your-repo-name.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print("3. Deploy on Streamlit Cloud:")
    print("   - Go to https://share.streamlit.io")
    print("   - Sign in with GitHub")
    print("   - Click 'New app'")
    print("   - Select your repository")
    print("   - Set main file path to: streamlit_app.py")
    print("   - Click 'Deploy!'")
    print("\n📚 For detailed instructions, see STREAMLIT_CLOUD_DEPLOYMENT.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 