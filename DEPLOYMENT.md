# CI/CD Deployment Guide

## GitHub Repository Secrets Setup

Navigate to your GitHub repository → Settings → Secrets and variables → Actions → New repository secret

Add the following secrets:

| Secret Name | Value | Description |
|------------|-------|-------------|
| `AWS_SSH_KEY` | Contents of `mohammad_anas.pem` | Private SSH key for AWS access |
| `AWS_HOST` | `13.48.123.136` | AWS instance IP address |
| `AWS_USER` | `ubuntu` | SSH username |
| `DEPLOY_PATH` | `/home/ubuntu/Strong Seaweed` | Deployment directory on AWS |

---

## AWS Server Initial Setup

### 1. SSH into your AWS instance

```bash
ssh -i "C:\Users\Anas\Downloads\mohammad_anas.pem" ubuntu@13.48.123.136
```

### 2. Clone the repository

```bash
cd ~
git clone <your-repo-url> "Strong Seaweed"
cd "Strong Seaweed/Strong Seaweed/Strong-Seaweed-main"
```

### 3. Install Python and dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git rsync

# Create virtual environment
python3 -m venv .venv_model
source .venv_model/bin/activate
pip install --upgrade pip
pip install -r requirements-model-api.txt
```

### 4. Upload your trained models

**Option A: Using GitHub workflow (recommended)**

1. Commit your models to the `releases/` folder
2. Go to Actions → "Sync Models to AWS" → Run workflow
3. Enter your release tag (e.g., `kappa_india_gulf_v2_prod_ready_v4_with_v2labels`)

**Option B: Manual upload via SCP**

```powershell
# From your local PC
scp -i "C:\Users\Anas\Downloads\mohammad_anas.pem" -r "Strong Seaweed\Strong-Seaweed-main\releases\<release_tag>" ubuntu@13.48.123.136:"/home/ubuntu/Strong Seaweed/Strong Seaweed/Strong-Seaweed-main/releases/"
```

### 5. Install systemd services

```bash
cd "/home/ubuntu/Strong Seaweed/Strong Seaweed/Strong-Seaweed-main"

# Copy service files to systemd
sudo cp .systemd/kappaphycus-api.service /etc/systemd/system/
sudo cp .systemd/species-api.service /etc/systemd/system/

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable kappaphycus-api
sudo systemctl enable species-api

# Start services
sudo systemctl start kappaphycus-api
sudo systemctl start species-api

# Check status
sudo systemctl status kappaphycus-api
sudo systemctl status species-api
```

### 6. Configure firewall (if needed)

```bash
sudo ufw allow 8000/tcp
sudo ufw allow 22/tcp
sudo ufw enable
```

---

## CI/CD Workflows

### 1. **CI Workflow** (`.github/workflows/ci.yml`)

**Triggers:** Every push or pull request to `main` or `develop` branches

**Jobs:**
- **Frontend**: Lint → Test → Build
- **Backend**: Syntax check → Lint with ruff → Import validation

### 2. **Deploy Workflow** (`.github/workflows/deploy.yml`)

**Triggers:** Manual (workflow_dispatch)

**Steps:**
1. Select service to deploy (kappaphycus / species / both)
2. Choose whether to restart after deployment
3. Runs:
   - Pull latest code from GitHub
   - Install Python dependencies
   - Restart selected service(s)
   - Health check

**Usage:**
- Go to Actions → "Deploy to AWS" → Run workflow
- Select service and options
- Click "Run workflow"

### 3. **Sync Models Workflow** (`.github/workflows/sync-models.yml`)

**Triggers:** Manual (workflow_dispatch)

**Steps:**
1. Enter release tag to sync
2. Uploads model files and data files to AWS via rsync
3. Verifies deployment

**Usage:**
- Go to Actions → "Sync Models to AWS" → Run workflow
- Enter release tag (e.g., `kappa_india_gulf_v2_prod_ready_v4_with_v2labels`)
- Click "Run workflow"

---

## Common Commands

### View service logs
```bash
# Live logs
sudo journalctl -u kappaphycus-api -f
sudo journalctl -u species-api -f

# Recent logs
sudo journalctl -u kappaphycus-api -n 100
```

### Restart services
```bash
sudo systemctl restart kappaphycus-api
sudo systemctl restart species-api
```

### Stop services
```bash
sudo systemctl stop kappaphycus-api
sudo systemctl stop species-api
```

### Manual testing
```bash
# Test kappaphycus API
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"lat": 8.5, "lon": 77.5}'

# Test species API
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"lat": 8.5, "lon": 77.5}'
```

---

## Development Workflow

### Local Development
1. Train new model on your PC
2. Test locally with `python serve_kappaphycus_api.py`
3. Commit code changes to GitHub
4. Commit trained model files to `releases/<new_tag>/`

### Deployment
1. Push code → CI runs automatically
2. If CI passes:
   - Run "Sync Models to AWS" workflow with your release tag
   - Run "Deploy to AWS" workflow to update code and restart services

### Rollback
```bash
# SSH to server
ssh -i "C:\Users\Anas\Downloads\mohammad_anas.pem" ubuntu@13.48.123.136

# Roll back to previous commit
cd "/home/ubuntu/Strong Seaweed"
git log --oneline -10  # Find commit hash
git reset --hard <commit_hash>

# Restart services
sudo systemctl restart kappaphycus-api
sudo systemctl restart species-api
```

---

## Troubleshooting

### Service won't start
```bash
# Check logs
sudo journalctl -u kappaphycus-api -n 50

# Check file permissions
ls -la "/home/ubuntu/Strong Seaweed/Strong Seaweed/Strong-Seaweed-main"

# Test manually
cd "/home/ubuntu/Strong Seaweed/Strong Seaweed/Strong-Seaweed-main"
source .venv_model/bin/activate
python serve_kappaphycus_api.py
```

### Import errors
```bash
# Reinstall dependencies
source .venv_model/bin/activate
pip install --upgrade pip
pip install -r requirements-model-api.txt --force-reinstall
```

### GitHub Actions SSH fails
- Check that `AWS_SSH_KEY` secret contains the full PEM file content
- Verify `AWS_HOST` and `AWS_USER` secrets are correct
- Ensure the PEM key has correct permissions on AWS side:
  ```bash
  chmod 400 ~/.ssh/authorized_keys
  ```
