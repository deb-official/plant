#!/bin/bash

# Update package list and install Git
echo "Updating package list and installing Git..."
sudo apt update -y
sudo apt install git -y

# Verify Git installation
git --version

# Configure Git with your name and email
# (set your own: GIT_NAME="Your Name" GIT_EMAIL="you@example.com" ./gitconfig.sh)
GIT_NAME="${GIT_NAME:-Anonymous}"
GIT_EMAIL="${GIT_EMAIL:-anonymous@users.noreply.github.com}"
echo "Configuring Git user details..."
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

# Generate SSH key if it doesn't exist
SSH_KEY_PATH="$HOME/.ssh/id_ed25519"
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "Generating new SSH key..."
    ssh-keygen -t ed25519 -C "$GIT_EMAIL" -f "$SSH_KEY_PATH" -N ""
    echo "SSH key generated at $SSH_KEY_PATH"
else
    echo "SSH key already exists at $SSH_KEY_PATH"
fi

# Set proper permissions for SSH key
chmod 600 "$SSH_KEY_PATH"
chmod 644 "$SSH_KEY_PATH.pub"

# Start SSH agent and add the key
eval "$(ssh-agent -s)"
ssh-add "$SSH_KEY_PATH"

# Display the public key to copy to your Git hosting service
echo "Here is your SSH public key. Copy it and add it to your Git hosting service (e.g., GitHub, GitLab):"
cat "$SSH_KEY_PATH.pub"

# Test SSH connection to GitHub (optional, comment out if not using GitHub)
echo "Testing SSH connection to GitHub..."
ssh -T git@github.com || echo "SSH test failed. Ensure the key is added to your Git host."

echo "Git and SSH setup complete! You can now clone, push, and pull repositories."
