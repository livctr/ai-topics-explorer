#!/bin/bash
set -e

# Usage: ./scp_copy.sh user@host:/path/to/destination

echo "WARNING: UNTESTED"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 remote_destination (format: user@host:/path/on/remote)"
    exit 1
fi

REMOTE_DESTINATION="$1"
SSH_KEY="$HOME/.ssh/cs-topics-explorer-server-key.pem"

# Determine the script directory and then the project root (parent of the script's directory)
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"
echo "Copying files from source directory: $SOURCE_DIR"

# Parse the remote destination into user@host and remote path.
REMOTE_USER_HOST="${REMOTE_DESTINATION%%:*}"
REMOTE_PATH="${REMOTE_DESTINATION#*:}"

# -----------------------------------------------------------------------------
# Step 1: Create the directory structure on the remote host.
#
# We use `find` to list all directories (excluding .git and node_modules),
# then for each directory, compute its relative path from SOURCE_DIR and
# create that directory remotely.
# -----------------------------------------------------------------------------
echo "Creating remote directory structure..."
while IFS= read -r -d '' dir; do
    # Compute the relative path of the directory
    rel_dir=$(realpath --relative-to="$SOURCE_DIR" "$dir")
    # Compute the full remote directory path
    remote_dir="$REMOTE_PATH/$rel_dir"
    echo "Creating remote directory: $remote_dir"
    ssh -i "$SSH_KEY" "$REMOTE_USER_HOST" "mkdir -p '$remote_dir'"
done < <(find "$SOURCE_DIR" -type d \
         \( -path '*/.git*' -o -path '*/node_modules*' \) -prune -o -print0)

# -----------------------------------------------------------------------------
# Step 2: Copy files one-by-one with scp.
#
# We use `find` to locate all files (excluding files in .git or node_modules and
# files ending with .pyc) and then, for each file, compute its relative path and
# copy it to the matching location on the remote host.
# -----------------------------------------------------------------------------
echo "Copying files..."
while IFS= read -r -d '' file; do
    # Compute the file's relative path
    rel_file=$(realpath --relative-to="$SOURCE_DIR" "$file")
    remote_file="$REMOTE_PATH/$rel_file"
    echo "Copying: $rel_file"
    scp -i "$SSH_KEY" "$file" "$REMOTE_USER_HOST:'$remote_file'"
done < <(find "$SOURCE_DIR" -type f \
         \( -path '*/.git*' -o -path '*/node_modules*' \) -prune -o \
         -name '*.pyc' -prune -o -print0)

echo "Copy complete!"
