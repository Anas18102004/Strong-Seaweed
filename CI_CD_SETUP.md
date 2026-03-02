# CI/CD Setup Summary

## ✅ Files Created

### GitHub Actions Workflows
- **`.github/workflows/ci.yml`** - Automated testing on push/PR
  - Frontend: lint, test, build
  - Backend: syntax check, ruff lint, import validation
  
- **`.github/workflows/deploy.yml`** - Manual deployment to AWS
  - Pull latest code
  - Install dependencies
  - Restart services
  - Health check
  
- **`.github/workflows/sync-models.yml`** - Manual model sync to AWS
  - Upload trained models
  - Sync data files
  - Verify deployment

### Systemd Service Files
- **`Strong Seaweed/Strong-Seaweed-main/.systemd/kappaphycus-api.service`**
  - Runs `serve_kappaphycus_api.py` on port 8000
  - Auto-restart on failure
  - Logs to `_kappaphycus_api_stdout.log` and `_kappaphycus_api_stderr.log`

- **`Strong Seaweed/Strong-Seaweed-main/.systemd/species-api.service`**
  - Runs `serve_species_api.py` on port 8000
  - Auto-restart on failure
  - Logs to `_species_api_stdout.log` and `_species_api_stderr.log`

### Configuration & Documentation
- **`Strong Seaweed/Strong-Seaweed-main/ruff.toml`** - Python linting config
- **`DEPLOYMENT.md`** - Complete deployment guide
- **`QUICK_REFERENCE.md`** - Quick command reference

## 🚀 Next Steps

### 1. Set up GitHub Secrets

Go to your GitHub repo → Settings → Secrets and variables → Actions

Add these secrets:

```
AWS_SSH_KEY = <paste full contents of mohammad_anas.pem>
AWS_HOST = 13.48.123.136
AWS_USER = ubuntu
DEPLOY_PATH = /home/ubuntu/Strong Seaweed
```

### 2. Initial AWS Server Setup

SSH into your server:
```bash
ssh -i "C:\Users\Anas\Downloads\mohammad_anas.pem" ubuntu@13.48.123.136
```

Follow the setup steps in `DEPLOYMENT.md` section "AWS Server Initial Setup"

Key steps:
- Clone repository
- Install Python + dependencies
- Install systemd services
- Start services

### 3. Commit and Push

```powershell
cd "C:\Users\Anas\Desktop\Strong Seaweed"
git add .
git commit -m "Add CI/CD workflows and deployment configuration"
git push origin main
```

### 4. Test CI

The CI workflow will run automatically on push. Check:
- Go to your GitHub repository
- Click "Actions" tab
- Watch the CI workflow run

### 5. Deploy to AWS

Once CI passes:
1. Go to Actions → "Sync Models to AWS" → Run workflow
   - Enter your release tag (e.g., `kappa_india_gulf_v2_prod_ready_v4_with_v2labels`)
2. Go to Actions → "Deploy to AWS" → Run workflow
   - Select service to deploy
   - Check "Restart service after deployment"

## 📊 Workflow Overview

```
┌─────────────────────────────────────────────────────────┐
│ Local Development                                       │
│  - Train models on your PC                             │
│  - Test with: python serve_kappaphycus_api.py          │
│  - Commit code + models to GitHub                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ GitHub CI (Automatic on Push)                          │
│  ✓ Frontend: lint → test → build                       │
│  ✓ Backend: syntax → lint → import validation          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼ (if CI passes)
┌─────────────────────────────────────────────────────────┐
│ Manual Deploy Actions                                   │
│  1. Sync Models to AWS (upload trained models)          │
│  2. Deploy to AWS (update code + restart services)      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ AWS Production Server (13.48.123.136)                  │
│  - Kappaphycus API running on http://localhost:8000    │
│  - Species API running on http://localhost:8000         │
│  - Systemd manages auto-restart + logging              │
└─────────────────────────────────────────────────────────┘
```

## 🔍 Verification

After deployment, verify everything works:

```bash
# SSH to server
ssh -i "C:\Users\Anas\Downloads\mohammad_anas.pem" ubuntu@13.48.123.136

# Check service status
sudo systemctl status kappaphycus-api
sudo systemctl status species-api

# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"lat": 8.5, "lon": 77.5}'

# View logs
sudo journalctl -u kappaphycus-api -n 50
```

## 📝 Notes

- **No Docker**: Simplified approach, dependencies managed via virtualenv
- **Model Training**: Done locally on your PC, not in CI/CD
- **Model Sync**: Via GitHub Actions workflow or manual SCP
- **Port**: Both APIs use port 8000 (run one at a time, or configure different ports)
- **Logs**: Stored in working directory and accessible via journalctl
- **Auto-restart**: Systemd restarts services on failure
- **Health Checks**: Already implemented in both API servers at `/health`

## 🛠️ Customization

### Run both APIs on different ports

Edit the service files:

**kappaphycus-api.service**:
```ini
ExecStart=... python serve_kappaphycus_api.py --port 8000
```

**species-api.service**:
```ini
ExecStart=... python serve_species_api.py --port 8001
```

(Note: You'll need to modify the Python files to accept a --port argument)

### Add email notifications on deployment

Add to deploy workflow:
```yaml
- name: Send notification
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    # ... add email config
```

See `DEPLOYMENT.md` for complete documentation.
