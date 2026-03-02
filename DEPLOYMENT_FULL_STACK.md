# Complete Full Stack Deployment Guide

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│ AWS Server: ubuntu@13.48.123.136                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────┐                                 │
│  │ Nginx :80              │ ← Public entry point            │
│  │ - Serves frontend      │                                 │
│  │ - Proxies /api → 4000  │                                 │
│  └──────────┬─────────────┘                                 │
│             │                                                │
│  ┌──────────▼──────────────────────────────────────────┐    │
│  │ Blue Weave Node.js Backend :4000                    │    │
│  │ - Express API                                       │    │
│  │ - Calls ML API (8000) & Python Agents (8101)       │    │
│  │ - Connects to MongoDB (27017)                      │    │
│  └─────┬───────────────┬────────────────────┬──────────┘    │
│        │               │                    │               │
│  ┌─────▼──────┐  ┌────▼────────┐  ┌────────▼──────────┐    │
│  │ ML API     │  │ Python      │  │ MongoDB           │    │
│  │ :8000      │  │ Agents      │  │ :27017            │    │
│  │ XGBoost    │  │ :8101       │  │ blueweave DB      │    │
│  │ Species    │  │ LangChain   │  │                   │    │
│  │ Kappaphycus│  │ CrewAI      │  │                   │    │
│  └────────────┘  └─────────────┘  └───────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## GitHub Repository Secrets

Go to: **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Value | Description |
|------------|-------|-------------|
| `AWS_SSH_KEY` | Contents of `mohammad_anas.pem` | Private SSH key |
| `AWS_HOST` | `13.48.123.136` | AWS instance IP |
| `AWS_USER` | `ubuntu` | SSH username |
| `DEPLOY_PATH` | `/home/ubuntu/Strong Seaweed` | Repo path on server |
| `MONGODB_URI` | `mongodb://127.0.0.1:27017/blueweave` | MongoDB connection (or Atlas URL) |
| `JWT_SECRET` | `<generate random string>` | JWT signing secret |

---

## AWS Server Initial Setup

### 1. SSH to Server

```bash
ssh -i "C:\Users\Anas\Downloads\mohammad_anas.pem" ubuntu@13.48.123.136
```

### 2. Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git rsync nginx curl

# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installations
python3 --version
node --version
npm --version
```

### 3. Install MongoDB

**Option A: Local MongoDB**

```bash
# Import MongoDB public GPG key
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Install MongoDB
sudo apt update
sudo apt install -y mongodb-org

# Start and enable MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod
sudo systemctl status mongod
```

**Option B: MongoDB Atlas (Cloud)**

1. Create free cluster at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Get connection string
3. Add to GitHub secret `MONGODB_URI`
4. Skip local MongoDB installation

### 4. Clone Repository

```bash
cd ~
git clone <your-repo-url> "Strong Seaweed"
cd "Strong Seaweed"
```

### 5. Setup ML Backend

```bash
cd "Strong Seaweed/Strong-Seaweed-main"

# Create virtual environment
python3 -m venv .venv_model
source .venv_model/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-model-api.txt

deactivate
cd ../..
```

### 6. Setup Python Agent Gateway

```bash
cd "Strong Seaweed/blue-weave-aqua-main/server/agents_python"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

deactivate
cd ../../../..
```

### 7. Setup Node.js Backend

```bash
cd "Strong Seaweed/blue-weave-aqua-main/server"

# Install dependencies
npm ci --production

# Create .env file
cat > .env << 'ENVFILE'
PORT=4000
MONGODB_URI=mongodb://127.0.0.1:27017/blueweave
JWT_SECRET=CHANGE_ME_IN_PRODUCTION
CORS_ORIGINS=http://13.48.123.136
MODEL_API_URL=http://127.0.0.1:8000
LANGCHAIN_API_URL=http://127.0.0.1:8101
LANGGRAPH_API_URL=http://127.0.0.1:8101
CREWAI_API_URL=http://127.0.0.1:8101
AI_ROUTER_MODE=hybrid
VOICE_TTS_PROVIDER=browser
ENVFILE

cd ../../..
```

### 8. Build Frontend

```bash
cd "Strong Seaweed/blue-weave-aqua-main"

# Install dependencies
npm ci

# Build for production
npm run build

cd ../..
```

### 9. Install Systemd Services

```bash
cd "Strong Seaweed"

# Copy ML API service
sudo cp "Strong Seaweed/Strong-Seaweed-main/.systemd/ml-api.service" /etc/systemd/system/

# Copy Python Agents service
sudo cp "Strong Seaweed/blue-weave-aqua-main/server/.systemd/python-agents.service" /etc/systemd/system/

# Copy Node.js Backend service
sudo cp "Strong Seaweed/blue-weave-aqua-main/server/.systemd/blueweave-backend.service" /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable ml-api
sudo systemctl enable python-agents
sudo systemctl enable blueweave-backend

# Start all services
sudo systemctl start ml-api
sudo systemctl start python-agents
sudo systemctl start blueweave-backend

# Check statuses
sudo systemctl status ml-api
sudo systemctl status python-agents
sudo systemctl status blueweave-backend
```

### 10. Configure Nginx

```bash
# Copy nginx configuration
sudo cp "Strong Seaweed/blue-weave-aqua-main/.nginx/blueweave.conf" /etc/nginx/sites-available/blueweave

# Enable site
sudo ln -s /etc/nginx/sites-available/blueweave /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Start nginx
sudo systemctl enable nginx
sudo systemctl restart nginx
```

### 11. Configure Firewall

```bash
# Allow HTTP, HTTPS, and SSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
sudo ufw enable

# Check status
sudo ufw status
```

---

## Uploading Trained Models

### Via GitHub Actions Workflow

1. Commit your models to `releases/<release_tag>/`
2. Push to GitHub
3. Go to Actions → "Sync Models to AWS"
4. Enter release tag
5. Run workflow

### Via SCP (Manual)

```powershell
# From your local PC
scp -i "C:\Users\Anas\Downloads\mohammad_anas.pem" -r `
  "Strong Seaweed\Strong-Seaweed-main\releases\<RELEASE_TAG>" `
  ubuntu@13.48.123.136:"/home/ubuntu/Strong Seaweed/Strong Seaweed/Strong-Seaweed-main/releases/"
```

---

## CI/CD Workflows

### 1. CI (Automatic on Push)

Runs 4 parallel jobs:
- ✓ Blue Weave Frontend: lint → test → build
- ✓ Blue Weave Backend: syntax check
- ✓ ML Backend: syntax → lint → import validation
- ✓ Python Agents: syntax → import validation

### 2. Deploy to AWS (Manual)

**Usage:**
1. Go to Actions → "Deploy to AWS"
2. Select service:
   - `ml-apis` - Only ML backend
   - `blueweave-backend` - Only Node.js backend
   - `blueweave-frontend` - Only frontend (rebuild + nginx reload)
   - `python-agents` - Only Python agent gateway
   - `all` - Everything
3. Check "Restart service after deployment"
4. Run workflow

### 3. Sync Models to AWS (Manual)

**Usage:**
1. Go to Actions → "Sync Models to AWS"
2. Enter release tag (e.g., `kappa_india_gulf_v2_prod_ready_v4_with_v2labels`)
3. Run workflow

---

## Service Management Commands

### View Logs

```bash
# ML API
sudo journalctl -u ml-api -f

# Python Agents
sudo journalctl -u python-agents -f

# Node.js Backend
sudo journalctl -u blueweave-backend -f

# Nginx
sudo journalctl -u nginx -f

# MongoDB
sudo journalctl -u mongod -f
```

### Restart Services

```bash
sudo systemctl restart ml-api
sudo systemctl restart python-agents
sudo systemctl restart blueweave-backend
sudo systemctl restart nginx
```

### Check Status

```bash
sudo systemctl status ml-api
sudo systemctl status python-agents
sudo systemctl status blueweave-backend
sudo systemctl status mongod
sudo systemctl status nginx
```

---

## Testing Your Deployment

### 1. Test ML API Directly

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"lat": 8.5, "lon": 77.5}'
```

### 2. Test Python Agents Directly

```bash
curl http://localhost:8101/health
```

### 3. Test Node.js Backend Directly

```bash
curl http://localhost:4000/api/health
```

### 4. Test Full Stack Through Nginx

```bash
# From AWS server
curl http://localhost/api/health

# From your PC
curl http://13.48.123.136/api/health
```

### 5. Access Frontend

Open browser: `http://13.48.123.136`

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
sudo journalctl -u <service-name> -n 100

# Check if port is already in use
sudo netstat -tulpn | grep :<port>

# Test manually
cd /home/ubuntu/Strong\ Seaweed/Strong\ Seaweed/<backend-path>
source .venv/bin/activate  # For Python services
node src/index.js  # For Node.js
python serve_species_api.py  # For ML API
```

### MongoDB Connection Failed

```bash
# Check if MongoDB is running
sudo systemctl status mongod

# Check logs
sudo journalctl -u mongod -n 50

# Test connection
mongosh --eval "db.adminCommand('ping')"
```

### Nginx 502 Bad Gateway

```bash
# Check if backends are running
sudo systemctl status blueweave-backend

# Check nginx error log
sudo tail -f /var/log/nginx/error.log

# Verify proxy ports in nginx config
sudo nginx -t
```

### Import Errors

```bash
# Reinstall Python dependencies
cd "Strong Seaweed/Strong-Seaweed-main"
source .venv_model/bin/activate
pip install -r requirements-model-api.txt --force-reinstall
```

---

## Production Checklist

- [ ] MongoDB secured (authentication enabled)
- [ ] JWT_SECRET is strong random string
- [ ] Firewall configured (ufw)
- [ ] Nginx configured
- [ ] All services enabled and running
- [ ] Log rotation configured
- [ ] Backups configured (MongoDB + models)
- [ ] SSL/TLS certificate installed (Let's Encrypt)
- [ ] Domain name configured (optional)
- [ ] Environment variables secured (.env files not committed)
- [ ] CORS origins configured correctly
- [ ] Tested full user flow

---

## SSL Setup (Optional but Recommended)

```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Get certificate (replace yourdomain.com)
sudo certbot --nginx -d yourdomain.com

# Auto-renew
sudo systemctl enable certbot.timer
```

---

## Monitoring & Maintenance

### Disk Space

```bash
df -h
du -sh /home/ubuntu/Strong\ Seaweed/*
```

### Memory Usage

```bash
free -h
htop
```

### Process Status

```bash
ps aux | grep python
ps aux | grep node
```

### Database Size

```bash
mongosh --eval "db.stats()"
```

---

## Development Workflow

1. **Local Development**
   - Code changes
   - Test locally
   - Train models (if ML-related)

2. **Commit & Push**
   ```bash
   git add .
   git commit -m "Description"
   git push origin main
   ```

3. **CI Runs Automatically**
   - Wait for all checks to pass

4. **Deploy**
   - If models changed: Run "Sync Models to AWS"
   - Run "Deploy to AWS" → Select service → Deploy

5. **Verify**
   - Check service logs
   - Test endpoints
   - Monitor for errors

---

## Rollback Procedure

```bash
# SSH to server
ssh -i "C:\Users\Anas\Downloads\mohammad_anas.pem" ubuntu@13.48.123.136

# View commit history
cd /home/ubuntu/Strong\ Seaweed
git log --oneline -10

# Roll back to specific commit
git reset --hard <commit-hash>

# Rebuild if needed
cd "Strong Seaweed/blue-weave-aqua-main"
npm ci
npm run build

# Restart services
sudo systemctl restart blueweave-backend
sudo systemctl restart ml-api
sudo systemctl restart python-agents
sudo systemctl reload nginx
```
