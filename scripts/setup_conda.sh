#!/bin/bash
set -e

# Define the persistent installation directory
INSTALL_DIR="$HOME/miniconda"

echo "🔍 Checking for Conda installation in persistent storage..."
if [ -d "$INSTALL_DIR" ]; then
    echo "✅ Conda is already installed in $INSTALL_DIR:"
    "$INSTALL_DIR/bin/conda" --version
else
    echo "⚠️ Conda not found in $INSTALL_DIR. Installing Miniconda..."

    # Detect system architecture
    ARCH=$(uname -m)
    echo "🧠 Detected architecture: $ARCH"

    # Set Miniconda URL based on architecture
    if [ "$ARCH" = "x86_64" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [ "$ARCH" = "aarch64" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
    else
        echo "❌ Unsupported architecture: $ARCH. Please install Miniconda manually."
        exit 1
    fi

    # Download Miniconda installer
    echo "📦 Downloading Miniconda for $ARCH..."
    if command -v wget &> /dev/null; then
        wget "$MINICONDA_URL" -O miniconda.sh
    elif command -v curl &> /dev/null; then
        curl -L "$MINICONDA_URL" -o miniconda.sh
    else
        echo "❌ Neither wget nor curl is installed. Please install one to continue."
        exit 1
    fi

    # Install Miniconda silently to the persistent directory
    echo "🛠 Installing Miniconda to $INSTALL_DIR..."
    bash miniconda.sh -b -p "$INSTALL_DIR"

    # Clean up installer
    rm miniconda.sh

    # Initialize Conda for the current shell
    echo "🔧 Initializing Conda..."
    "$INSTALL_DIR/bin/conda" init bash

    # Source the conda initialization script directly
    source "$INSTALL_DIR/etc/profile.d/conda.sh"

    # Verify installation
    if command -v conda &> /dev/null; then
        echo "✅ Conda installed successfully in $INSTALL_DIR:"
        conda --version
    else
        echo "❌ Conda installation failed. Please check the output for errors."
        exit 1
    fi
fi

# Add Conda Forge channel
echo "➕ Adding Conda Forge channel..."
conda config --add channels conda-forge

echo "🎉 Conda setup complete! You can now use 'conda' commands."
