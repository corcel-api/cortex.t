#!/bin/bash

# Set error handling
set -e

# Define variables
EXIFTOOL_VERSION="13.19"
DOWNLOAD_URL="https://exiftool.org/Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz"
DOWNLOAD_DIR="$HOME/Downloads"
INSTALL_DIR="/usr/local/bin"

# Create Downloads directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

echo "Downloading ExifTool version ${EXIFTOOL_VERSION}..."
cd "$DOWNLOAD_DIR"
curl -O "$DOWNLOAD_URL"

echo "Unpacking ExifTool..."
gzip -dc "Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz" | tar -xf -
cd "Image-ExifTool-${EXIFTOOL_VERSION}"

# Install perl module
echo "Installing ExifTool..."
perl Makefile.PL
make
sudo make install

# Create symbolic link to make exiftool accessible globally
echo "Making exiftool globally accessible..."
if [ -f "/usr/local/bin/exiftool" ]; then
    echo "ExifTool is already installed globally"
else
    sudo cp -f exiftool "$INSTALL_DIR/"
    sudo chmod 755 "$INSTALL_DIR/exiftool"
fi

# Clean up downloaded files
echo "Cleaning up..."
cd "$DOWNLOAD_DIR"
rm -rf "Image-ExifTool-${EXIFTOOL_VERSION}"*

echo "Installation complete! You can now use 'exiftool' from any directory."
echo "Try running: exiftool -ver"