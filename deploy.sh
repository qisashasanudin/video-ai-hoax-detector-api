#!/bin/bash
# Fast deployment script for Oracle Cloud

set -e

echo "🚀 Video AI Hoax Detector API - Oracle Cloud Deployment"
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Oracle Cloud
if [ ! -f "/opt/oracle-cloud" ]; then
    echo -e "${YELLOW}⚠️  Note: This script is designed for Oracle Cloud VMs${NC}"
fi

# Step 1: Update system
echo -e "\n${YELLOW}📦 Updating system packages...${NC}"
sudo apt-get update && sudo apt-get upgrade -y

# Step 2: Install Docker
if ! command -v docker &> /dev/null; then
    echo -e "\n${YELLOW}🐳 Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
else
    echo -e "${GREEN}✓ Docker already installed${NC}"
fi

# Step 3: Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "\n${YELLOW}📋 Installing Docker Compose...${NC}"
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    echo -e "${GREEN}✓ Docker Compose already installed${NC}"
fi

# Step 4: Install Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "\n${YELLOW}🧠 Installing Ollama...${NC}"
    curl -fsSL https://ollama.ai/install.sh | sh
    sudo systemctl start ollama
    sudo systemctl enable ollama
else
    echo -e "${GREEN}✓ Ollama already installed${NC}"
fi

# Step 5: Pull Gemma model
echo -e "\n${YELLOW}📥 Pulling Gemma model (this may take a few minutes)...${NC}"
ollama pull gemma:7b

# Step 6: Create data directory
echo -e "\n${YELLOW}📁 Creating data directory...${NC}"
mkdir -p data/jobs logs

# Step 7: Build Docker image
echo -e "\n${YELLOW}🔨 Building Docker image...${NC}"
docker-compose build

# Step 8: Start services
echo -e "\n${YELLOW}▶️  Starting services...${NC}"
docker-compose up -d

# Step 9: Verify services
echo -e "\n${YELLOW}🔍 Verifying services...${NC}"
sleep 5

if curl -s http://localhost:8000/docs > /dev/null; then
    echo -e "${GREEN}✓ API is running at http://localhost:8000${NC}"
else
    echo -e "${RED}✗ API failed to start. Check logs with: docker-compose logs api${NC}"
    exit 1
fi

if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo -e "${GREEN}✓ Ollama is running at http://localhost:11434${NC}"
else
    echo -e "${RED}✗ Ollama failed to start. Check with: sudo systemctl status ollama${NC}"
fi

# Step 10: Display summary
echo -e "\n${GREEN}========================================================${NC}"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo -e "${GREEN}========================================================${NC}"
echo ""
echo "📍 API URL: http://$(hostname -I | awk '{print $1}'):8000"
echo "📍 API Docs: http://$(hostname -I | awk '{print $1}'):8000/docs"
echo ""
echo "🔧 Useful Commands:"
echo "   docker-compose logs -f api        # View API logs"
echo "   docker-compose ps                 # Check services"
echo "   docker-compose down               # Stop services"
echo "   docker-compose up -d              # Start services"
echo ""
echo "📚 Next Steps:"
echo "   1. Update your frontend API URL to: http://$(hostname -I | awk '{print $1}'):8000"
echo "   2. Open firewall port 8000 in Oracle Cloud console"
echo "   3. Test the API at http://$(hostname -I | awk '{print $1}'):8000/docs"
echo ""
