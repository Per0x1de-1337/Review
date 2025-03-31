#!/bin/bash

# Check if a branch name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <branch-name>"
    exit 1
fi

BRANCH_NAME="$1"

# Create and switch to the new branch
git checkout -b "$BRANCH_NAME"

# Modify nothing.go with random content
echo "$(cat nothing.go) $(tr -dc 'a-zA-Z0-9' </dev/urandom | head -c 10)" > nothing.go

# Add, commit, and push changes
git add nothing.go
git commit -m "add random"
git push --set-upstream origin "$BRANCH_NAME"
