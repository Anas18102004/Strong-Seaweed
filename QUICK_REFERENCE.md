# Quick Reference: CI/CD Commands

## GitHub Secrets Required

```
AWS_SSH_KEY = <contents of mohammad_anas.pem>
AWS_HOST = 13.48.123.136
AWS_USER = ubuntu
DEPLOY_PATH = /home/ubuntu/Strong Seaweed
```

## Local Commands

### SSH to AWS
```powershell
ssh -i "C:\Users\Anas\Downloads\mohammad_anas.pem" ubuntu@13.48.123.136
```

### Upload models manually
```powershell
scp -i "C:\Users\Anas\Downloads\mohammad_anas.pem" -r `
  "Strong Seaweed\Strong-Seaweed-main\releases\<RELEASE_TAG>" `
  ubuntu@13.48.123.136:"/home/ubuntu/Strong Seaweed/Strong Seaweed/Strong-Seaweed-main/releases/"
```

## AWS Server Commands

### Service Management
```bash
# Start
sudo systemctl start kappaphycus-api
sudo systemctl start species-api

# Stop
sudo systemctl stop kappaphycus-api
sudo systemctl stop species-api

# Restart
sudo systemctl restart kappaphycus-api
sudo systemctl restart species-api

# Status
sudo systemctl status kappaphycus-api
sudo systemctl status species-api

# View logs (live)
sudo journalctl -u kappaphycus-api -f
sudo journalctl -u species-api -f
```

### Manual Testing
```bash
# Kappaphycus API
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"lat": 8.5, "lon": 77.5}'

# Species API  
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"lat": 8.5, "lon": 77.5}'

# Health check
curl http://localhost:8000/health
```

### Update Code
```bash
cd "/home/ubuntu/Strong Seaweed"
git pull origin main
cd "Strong Seaweed/Strong-Seaweed-main"
source .venv_model/bin/activate
pip install -r requirements-model-api.txt
sudo systemctl restart kappaphycus-api species-api
```

## GitHub Actions Workflows

### 1. CI (Auto on push/PR)
- Runs frontend lint, test, build
- Runs backend syntax check, lint, import validation

### 2. Deploy to AWS (Manual)
1. Go to Actions → "Deploy to AWS"
2. Select service: kappaphycus / species / both
3. Choose restart: yes / no
4. Run workflow

### 3. Sync Models (Manual)
1. Go to Actions → "Sync Models to AWS"
2. Enter release tag (e.g., `kappa_india_gulf_v2_prod_ready_v4_with_v2labels`)
3. Run workflow

## Development Workflow

```
1. Train model locally → test locally
2. Commit code + models to GitHub
3. CI runs automatically
4. If CI passes:
   a. Run "Sync Models to AWS" workflow
   b. Run "Deploy to AWS" workflow
5. Verify with health check/logs
```
