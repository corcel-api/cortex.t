#!/bin/bash

# Function to check if the user has root privileges
check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "You are not running as root. Commands requiring root will use 'sudo'."
        SUDO="sudo"
    else
        echo "You are running as root. 'sudo' is not required."
        SUDO=""
    fi
}

# Run the root check
check_root

# Update the package list
$SUDO apt update

# Install Redis
$SUDO apt install -y redis



# Verify installation
if redis-cli --version; then
    echo "Redis installed successfully."
else
    echo "Redis installation failed."
    exit 1
fi

# Attempt to start Redis with systemctl
echo "Attempting to start Redis using systemctl..."
if $SUDO systemctl start redis 2>/dev/null; then
    echo "Redis started successfully using systemctl."
else
    echo "systemctl not available or failed. Starting Redis manually..."
    if redis-server --daemonize yes; then
        echo "Redis started manually in the background."
    else
        echo "Failed to start Redis. Check your setup."
        exit 1
    fi
fi

# Enable Redis to start on boot (optional, for non-WSL environments)
if $SUDO systemctl enable redis 2>/dev/null; then
    echo "Redis enabled to start on boot using systemctl."
else
    echo "systemctl not available. Skipping boot configuration."
fi

# Test Redis
if redis-cli ping | grep -q "PONG"; then
    echo "Redis is working correctly!"
else
    echo "Redis test failed. Check the service status or logs."
fi