#!/bin/bash

# Function to update repository
update_repo() {
    echo "Checking for updates..."
    
    # Fetch latest changes without merging
    git fetch origin

    # Get current and remote hash
    LOCAL_HASH=$(git rev-parse HEAD)
    REMOTE_HASH=$(git rev-parse origin/main)

    if [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
        echo "Updates available. Updating repository..."
        
        # Stash any local changes
        git stash
        git checkout main
        # Pull latest changes
        git pull origin main
        git reset --hard origin/main
        
        # Reinstall dependencies
        uv sync --prerelease=allow
        
        pm2 restart all
        
        echo "Update completed successfully!"
    else
        echo "Repository is up to date!"
    fi
}

# Run the update function
while true; do
    update_repo
    sleep 1800  # Sleep for 30 minutes (1800 seconds)
done