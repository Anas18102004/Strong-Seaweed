#!/bin/bash
# AWS Server Bootstrap Script
# Run this ONCE on a fresh Ubuntu 20.04+ AWS EC2 instance
# Usage: curl -fsSL https://raw.githubusercontent.com/Anas18102004/Strong-Seaweed/beta/AWS_BOOTSTRAP.sh | bash

set -e

echo "🚀 Strong Seaweed AWS Server Bootstrap"
echo "======================================="

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Update system
echo -e "${BLUE}[1/8] Updating system packages...${NC}"
sudo apt update
sudo apt upgrade -y
sudo apt install -y curl wget git build-essential

# Step 2: Install Node.js
echo -e "${BLUE}[2/8] Installing Node.js 20...${NC}"
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Step 3: Install Python 3.11 (via deadsnakes PPA for Ubuntu 24.04)
echo -e "${BLUE}[3/8] Installing Python 3.11...${NC}"
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Step 4: Install MongoDB
echo -e "${BLUE}[4/8] Installing MongoDB...${NC}"
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt update
sudo apt install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod

# Step 5: Install Nginx
echo -e "${BLUE}[5/8] Installing Nginx...${NC}"
sudo apt install -y nginx
sudo systemctl enable nginx

# Step 6: Clone repository
echo -e "${BLUE}[6/8] Cloning Strong Seaweed repository...${NC}"
cd ~
if [ ! -d "Strong Seaweed" ]; then
  git clone https://github.com/Anas18102004/Strong-Seaweed.git "Strong Seaweed"
else
  echo "Repository already exists, updating..."
  cd "Strong Seaweed"
  git fetch origin
  git checkout beta
  git pull origin beta
  cd ~
fi

cd "Strong Seaweed"
git checkout beta
git pull origin beta

# Step 7: Copy systemd services
echo -e "${BLUE}[7/8] Setting up systemd services...${NC}"
sudo cp "Strong Seaweed/Strong-Seaweed-main/.systemd/ml-api.service" /etc/systemd/system/ 2>/dev/null || true
sudo cp "Strong Seaweed/blue-weave-aqua-main/.systemd/python-agents.service" /etc/systemd/system/ 2>/dev/null || true
sudo cp "Strong Seaweed/blue-weave-aqua-main/.systemd/blueweave-backend.service" /etc/systemd/system/ 2>/dev/null || true

sudo systemctl daemon-reload
sudo systemctl enable ml-api python-agents blueweave-backend 2>/dev/null || true

# Step 8: Setup Nginx
echo -e "${BLUE}[8/8] Configuring Nginx...${NC}"
sudo cp "Strong Seaweed/blue-weave-aqua-main/.nginx/blueweave.conf" /etc/nginx/sites-available/ 2>/dev/null || true
sudo ln -sf /etc/nginx/sites-available/blueweave.conf /etc/nginx/sites-enabled/blueweave.conf
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# Summary
echo ""
echo -e "${GREEN}✅ Bootstrap Complete!${NC}"
echo "======================================="
echo ""
echo "📋 Next Steps:"
echo "1. Add these GitHub Secrets (Settings → Secrets and variables → Actions):"
echo "   - MONGODB_URI: Your MongoDB Atlas connection string"
echo "   - JWT_SECRET: Your JWT secret (generate random)"
echo ""
echo "2. Test deploy workflow:"
echo "   - Go to GitHub Actions"
echo "   - Select 'Deploy to AWS'"
echo "   - Choose service: 'all'"
echo "   - Check 'Restart service': Yes"
echo "   - Click 'Run workflow'"
echo ""
echo "3. Monitor logs:"
echo "   ssh ubuntu@13.48.123.136"
echo "   sudo journalctl -u ml-api -n 50 -f"
echo ""
echo "📊 Verify services running:"
echo "   sudo systemctl status ml-api python-agents blueweave-backend nginx"
echo ""
echo "🩺 Health check:"
echo "   curl http://localhost/health"
echo ""
echo "🚀 Ready for CI/CD automation!"
