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
        
        prefix="cortex"
        # Get list of PM2 processes with the cortext prefix
        processes=$(pm2 list | grep "$prefix" | awk '{print $4}')

        # Check if any matching processes were found
        if [ -z "$processes" ]; then
            echo "No cortext processes found"
            exit 0
        fi

        # Restart each matching process
        echo "Found cortext processes:"
        echo "$processes"
        echo "Restarting..."

        for process in $processes; do
            echo "Restarting $process..."
            pm2 restart "$process"
        done

        echo "Restart complete!"
        
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