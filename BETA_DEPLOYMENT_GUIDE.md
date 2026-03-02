# Beta Branch CI/CD Pipeline - Complete Setup Guide

## 🎯 Overview

The **beta** branch is now configured as the testing/staging environment for all 4 applications:
1. Blue Weave Frontend (React + Vite)
2. Blue Weave Node.js Backend (Express + MongoDB)
3. Python Agent Gateway (FastAPI + LangChain)
4. ML Backend (XGBoost APIs)

---

## 📋 What Was Updated

### ✅ GitHub Actions Workflows (Updated)

| Workflow | Branch | Trigger | Purpose |
|----------|--------|---------|---------|
| `ci.yml` | beta, main | Push/PR | Tests all 4 applications |
| `deploy.yml` | beta, main | Manual (workflow_dispatch) | Deploy to AWS from beta |
| `sync-models.yml` | beta or main | Manual (selectable) | Sync ML models to AWS |

---

## 🚀 Step 1: Set GitHub Repository Secrets

Navigate to: **GitHub → Your Repo → Settings → Secrets and variables → Actions**

Add these secrets:

```
AWS_SSH_KEY = [Contents of mohammad_anas.pem]
AWS_HOST = 13.48.123.136
AWS_USER = ubuntu
DEPLOY_PATH = /home/ubuntu/Strong Seaweed
MONGODB_URI = mongodb://127.0.0.1:27017/blueweave (or MongoDB Atlas URL)
JWT_SECRET = [Generate a random strong string]
```

---

## 🔄 Step 2: Test CI/CD Pipeline

### Test 1: Trigger CI on Beta Branch

```bash
# Make a small change and push to beta
git checkout beta
echo "# Test CI" >> TEST.md
git add TEST.md
git commit -m "Test CI pipeline"
git push origin beta
```

**Check:** Go to GitHub → Actions tab → Watch the CI workflow run

Expected results:
- ✅ blueweave-frontend: lint, test, build
- ✅ blueweave-backend: syntax check
- ✅ ml-backend: syntax, lint, imports
- ✅ python-agents: syntax, imports

### Test 2: Test Deploy Workflow (Manual)

```bash
git checkout beta
# Make a meaningful change
git add .
git commit -m "Ready for beta deployment testing"
git push origin beta
```

**On GitHub:**
1. Go to Actions → "Deploy to AWS"
2. Click "Run workflow"
3. Select:
   - Service: `blueweave-frontend` (start with just frontend)
   - Restart: `true`
4. Click "Run workflow"
5. Monitor the deployment logs

---

## 📁 Repository Structure (Updated)

```
Strong Seaweed/                                              ← Git root repo
├── .github/
│   └── workflows/
│       ├── ci.yml                    ← Runs on beta/main push
│       ├── deploy.yml                ← Deploys from beta
│       └── sync-models.yml           ← Syncs models from beta/main
│
├── Strong Seaweed/                                          ← Apps folder
│   ├── blue-weave-aqua-main/
│   │   ├── src/                      ← Frontend React code
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   ├── .nginx/
│   │   │   └── blueweave.conf        ← Nginx config
│   │   └── server/
│   │       ├── src/index.js          ← Node.js backend
│   │       ├── package.json
│   │       ├── .systemd/
│   │       │   ├── blueweave-backend.service
│   │       │   └── python-agents.service
│   │       └── agents_python/
│   │           ├── agent_gateway.py  ← Python agents
│   │           └── requirements.txt
│   │
│   └── Strong-Seaweed-main/
│       ├── serve_species_api.py      ← ML APIs
│       ├── serve_kappaphycus_api.py
│       ├── requirements-model-api.txt
│       ├── .systemd/
│       │   └── ml-api.service
│       └── ruff.toml
│
├── DEPLOYMENT_FULL_STACK.md          ← AWS server setup
├── CI_CD_SETUP.md                    ← Initial setup guide
├── FULL_ANALYSIS_SUMMARY.md          ← Architecture overview
└── QUICK_REFERENCE.md                ← Command reference
```

---

## 🔧 Step 3: Setup AWS Server (First Time Only)

Follow [DEPLOYMENT_FULL_STACK.md](./DEPLOYMENT_FULL_STACK.md) step-by-step:

### Essential Steps:
1. SSH to server
2. Install system dependencies (Python3, Node.js, MongoDB, nginx)
3. Clone repository and checkout beta branch
4. Setup virtual environments
5. Install systemd services
6. Configure nginx
7. Start all services

**Key Command:**
```bash
ssh -i "C:\Users\Anas\Downloads\mohammad_anas.pem" ubuntu@13.48.123.136
cd ~
git clone https://github.com/Anas18102004/Strong-Seaweed.git "Strong Seaweed"
cd "Strong Seaweed"
git checkout beta
# Then follow DEPLOYMENT_FULL_STACK.md...
```

---

## 📊 CI/CD Workflow Breakdown

### Workflow 1: CI (`.github/workflows/ci.yml`)

**Triggers:** 
- Push to `beta` or `main` branch
- Pull requests to `beta` or `main`

**Jobs (Parallel):**

#### Job 1: blueweave-frontend
```yaml
Path: Strong Seaweed/blue-weave-aqua-main/
Tasks:
  - npm ci
  - npm run lint (ESLint)
  - npm test (vitest)
  - npm run build
```

#### Job 2: blueweave-backend
```yaml
Path: Strong Seaweed/blue-weave-aqua-main/server/
Tasks:
  - npm ci
  - node --check src/index.js (syntax validation)
```

#### Job 3: ml-backend
```yaml
Path: Strong Seaweed/Strong-Seaweed-main/
Tasks:
  - pip install -r requirements-model-api.txt
  - python -m py_compile *.py
  - ruff check . (Python linting)
  - Validate imports (serve_*.py, project_paths)
```

#### Job 4: python-agents
```yaml
Path: Strong Seaweed/blue-weave-aqua-main/server/agents_python/
Tasks:
  - pip install -r requirements.txt
  - python -m py_compile agent_gateway.py
  - Validate imports
```

---

### Workflow 2: Deploy to AWS (`.github/workflows/deploy.yml`)

**Trigger:** Manual (workflow_dispatch)

**Options:**
- Service: `ml-apis` | `blueweave-backend` | `blueweave-frontend` | `python-agents` | `all`
- Restart: `true` | `false`

**Process:**
1. Setup SSH key from secrets
2. SSH to AWS server
3. Checkout beta branch (`git checkout beta`)
4. Pull latest code (`git reset --hard origin/beta`)
5. Setup selected service(s):
   - **ml-apis**: Create venv, install requirements
   - **python-agents**: Create venv, install requirements
   - **blueweave-backend**: npm ci --production
   - **blueweave-frontend**: npm ci + npm run build
6. Restart systemd services (if restart=true)
7. Health check

---

### Workflow 3: Sync Models (`.github/workflows/sync-models.yml`)

**Trigger:** Manual (workflow_dispatch)

**Options:**
- Release tag: `kappa_india_gulf_v2_prod_ready_v4_with_v2labels` (or any release tag)
- Branch: `beta` (default) | `main`

**Process:**
1. Checkout selected branch
2. Find release folder: `Strong Seaweed/Strong-Seaweed-main/releases/{RELEASE_TAG}/`
3. Sync via rsync:
   ```bash
   rsync -avz {RELEASE_TAG}/ ubuntu@13.48.123.136:...
   ```
4. Sync data files (netcdf, feature matrix)
5. Verify on remote

---

## 🧪 Development & Deployment Workflow

### For Beta Testing:

1. **Create feature branch from beta:**
   ```bash
   git checkout beta
   git checkout -b feature/my-feature
   ```

2. **Make changes:**
   - Frontend: `Strong Seaweed/blue-weave-aqua-main/src/`
   - Backend: `Strong Seaweed/blue-weave-aqua-main/server/src/`
   - ML APIs: `Strong Seaweed/Strong-Seaweed-main/serve_*.py`
   - Agents: `Strong Seaweed/blue-weave-aqua-main/server/agents_python/`

3. **Test locally:**
   ```bash
   # Frontend
   cd "Strong Seaweed/blue-weave-aqua-main"
   npm run dev
   
   # Backend
   cd server
   npm run dev
   
   # ML APIs
   cd ../../../Strong-Seaweed-main
   python serve_species_api.py
   ```

4. **Commit to feature branch:**
   ```bash
   git add .
   git commit -m "Add feature X"
   git push origin feature/my-feature
   ```

5. **Create Pull Request:**
   - Base branch: `beta`
   - Head branch: `feature/my-feature`
   - CI runs automatically

6. **Once CI passes:**
   - Merge to beta
   - CI runs again on beta
   - Deploy manually: GitHub → Actions → "Deploy to AWS"

---

## 🔍 Monitoring Deployments

### Check Deployment Status

```bash
# SSH to server
ssh -i mohammad_anas.pem ubuntu@13.48.123.136

# Check service status
sudo systemctl status ml-api
sudo systemctl status python-agents
sudo systemctl status blueweave-backend
sudo systemctl status nginx

# View logs
sudo journalctl -u ml-api -f
sudo journalctl -u python-agents -f
sudo journalctl -u blueweave-backend -f
sudo journalctl -u nginx -f

# Test frontends/APIs
curl http://localhost:8000/health        # ML API
curl http://localhost:8101/health        # Python agents
curl http://localhost:4000/api/health    # Node backend
curl http://localhost/api/health         # Through nginx
```

---

## 📈 Promoting Beta to Main

When beta is **stable and tested:**

1. **Ensure beta has all changes:**
   ```bash
   git checkout beta
   git log --oneline -5  # Verify latest commits
   ```

2. **Create PR: beta → main:**
   - GitHub → New Pull Request
   - Base: `main`
   - Compare: `beta`
   - Create PR
   - Wait for CI to pass

3. **Merge PR:**
   - Merge strategy: Create a merge commit (for history)
   - Or squash if you prefer clean history

4. **Update main deploy configurations (if needed)**

5. **Main branch CI/CD now runs** and deploys production

---

## 🚨 Troubleshooting

### CI Workflow Fails

**Check logs:**
- GitHub → Actions → Failed workflow → Click job → View output

**Common issues:**
- Missing dependencies: Run `npm ci` or `pip install`
- Syntax errors: Check Python/JavaScript files
- Import errors: Verify module paths with Strong Seaweed prefix

### Deploy Fails

1. **Check SSH connection:**
   ```bash
   ssh -i mohammad_anas.pem ubuntu@13.48.123.136 "echo OK"
   ```

2. **Verify secrets are set:**
   - GitHub → Settings → Secrets → Check all 4 secrets exist

3. **Check server state:**
   ```bash
   ssh ubuntu@13.48.123.136
   cd /home/ubuntu/"Strong Seaweed"
   git status
   git branch
   ```

### Service Won't Start

1. **Check logs:**
   ```bash
   sudo journalctl -u ml-api -n 50
   ```

2. **Test manually:**
   ```bash
   cd /path/to/service
   source .venv/bin/activate  # For Python
   python serve_species_api.py  # For ML API
   ```

---

## ✅ Deployment Checklist

- [ ] GitHub secrets configured (AWS_SSH_KEY, AWS_HOST, AWS_USER, DEPLOY_PATH, MONGODB_URI, JWT_SECRET)
- [ ] AWS server setup complete (DEPLOYMENT_FULL_STACK.md followed)
- [ ] Beta branch has all application code
- [ ] CI workflow tests pass
- [ ] Manual deploy workflow tested (at least frontend)
- [ ] All 4 services running on AWS
- [ ] Frontend accessible at http://13.48.123.136
- [ ] Backend health checks passing
- [ ] MongoDB connected
- [ ] Logs being generated without errors
- [ ] Monitoring/alerting configured (optional)

---

## 📚 Documentation Files

- **[DEPLOYMENT_FULL_STACK.md](./DEPLOYMENT_FULL_STACK.md)** - Complete AWS server setup
- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - Command cheat sheet
- **[FULL_ANALYSIS_SUMMARY.md](./FULL_ANALYSIS_SUMMARY.md)** - Architecture overview
- **[CI_CD_SETUP.md](./CI_CD_SETUP.md)** - Initial CI/CD intro

---

## 🎓 Next Steps

1. **Set GitHub secrets** (Step 1 above)
2. **Setup AWS server** (follow DEPLOYMENT_FULL_STACK.md)
3. **Test CI** by pushing to beta
4. **Test Deploy** workflow manually
5. **Monitor logs** during deployment
6. **Once stable:** Create PR beta → main for production

---

**Beta branch is ready! 🚀**
