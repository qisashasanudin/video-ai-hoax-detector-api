# Deployment Guide: Oracle Cloud Always Free

This guide walks you through deploying the Video AI Hoax Detector API to Oracle Cloud Always Free tier.

## Prerequisites

- Oracle Cloud Always Free account (sign up at https://www.oracle.com/cloud/free/)
- SSH client installed on your machine
- GitHub access token (optional, for private repos)

## Step 1: Create Oracle Cloud VM Instance

1. **Sign in to Oracle Cloud Console** → https://www.oracle.com/cloud/sign-in/
2. **Navigate to**: Compute → Instances
3. **Click**: Create Instance
4. **Configure**:
   - **Name**: `video-ai-hoax-detector-api`
   - **Image**: Ubuntu 22.04 (always free eligible)
   - **Shape**: Ampere A1 (ARM-based, 4 OCPUs, 24GB RAM - always free)
   - **Networking**: Default VCN
   - **SSH Key**: Download and save the private key (e.g., `ssh-key.key`)
   - **Public IP**: Assign
5. **Click**: Create

**Wait 2-3 minutes for the instance to start.**

## Step 2: SSH into Your Instance

```bash
# Make key readable
chmod 600 ~/Downloads/ssh-key.key

# SSH into instance (replace with your instance IP)
ssh -i ~/Downloads/ssh-key.key ubuntu@<INSTANCE_IP>
```

## Step 3: Install Docker & Docker Compose

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version

# Log out and back in for group changes to take effect
exit
ssh -i ~/Downloads/ssh-key.key ubuntu@<INSTANCE_IP>
```

## Step 4: Clone Repository

```bash
# Clone the API repository
git clone https://github.com/qisashasanudin/video-ai-hoax-detector-api.git
cd video-ai-hoax-detector-api
```

## Step 5: Configure Environment

```bash
# Create .env file (if needed)
cat > .env << EOF
OLLAMA_MODEL_NAME=gemma4
MISINFO_LLM_MODEL=ollama://gemma4
EOF
```

## Step 6: Setup Ollama on the Host

Ollama must run on the host machine, not in Docker (for GPU/performance).

```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama

# Pull Gemma4 model (takes ~5-10 min on first run)
ollama pull gemma:7b  # or gemma:13b for larger model

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

## Step 7: Build and Run with Docker Compose

```bash
# Build the Docker image
docker-compose build

# Start the API container
docker-compose up -d

# Check logs
docker-compose logs -f api

# Verify API is running
curl http://localhost:8000/docs
```

## Step 8: Configure Firewall Rules

In Oracle Cloud Console:

1. Go to **Networking → Virtual Cloud Networks**
2. Select your VCN
3. Find **Security Lists**
4. Edit the default security list
5. Add **Ingress Rule**:
   - Protocol: TCP
   - Source: 0.0.0.0/0 (any IP)
   - Destination Port: 8000
6. Save

## Step 9: Update Frontend API URL

Update your Next.js frontend to point to your Oracle Cloud instance:

**web/src/lib/mockApi.ts:**

```typescript
const API_BASE_URL = "http://<INSTANCE_IP>:8000";
```

Or use a domain name if you have one:

```typescript
const API_BASE_URL = "https://api.yourdomain.com";
```

## Step 10: Setup PM2 for Auto-Restart (Optional)

```bash
# Install PM2 for process management
sudo npm install -g pm2

# Create PM2 config for docker-compose
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'api',
    script: 'docker-compose',
    args: 'up',
    autorestart: true,
    max_memory_restart: '1G',
    error_file: './logs/pm2-error.log',
    out_file: './logs/pm2-out.log',
  }]
};
EOF

# Start with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

## Monitoring & Logs

```bash
# View API logs
docker-compose logs -f api

# Check Docker container status
docker ps

# View system resources
free -h  # Memory
df -h    # Disk space
```

## Maintenance

### Update Code

```bash
cd ~/video-ai-hoax-detector-api
git pull origin main
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup Data

```bash
# Backup job data
tar -czf backup-jobs-$(date +%Y%m%d).tar.gz data/jobs/
```

### Stop/Restart

```bash
# Stop container
docker-compose down

# Restart container
docker-compose up -d
```

## Troubleshooting

### API not responding

```bash
# Check container logs
docker-compose logs api

# Ensure firewall rule allows port 8000
# Check from local machine: curl http://<INSTANCE_IP>:8000/docs
```

### Ollama connection issues

```bash
# Verify Ollama is running
sudo systemctl status ollama

# Check Ollama endpoint
curl http://localhost:11434/api/tags

# Restart Ollama if needed
sudo systemctl restart ollama
```

### Out of disk space

```bash
# Check disk usage
df -h

# Clean Docker images/containers
docker system prune -a
docker volume prune
```

## Cost

- **Oracle Cloud Always Free**: $0/month (forever free)
- No hidden charges or surprise billing

## Resources

- Oracle Cloud Free Tier: https://www.oracle.com/cloud/free/
- Docker Documentation: https://docs.docker.com/
- Ollama Documentation: https://ollama.ai/
- FastAPI Documentation: https://fastapi.tiangolo.com/
