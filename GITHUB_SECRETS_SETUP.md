# GitHub Secrets Configuration Guide

## Overview

Your CI/CD pipeline requires 6 secrets to be set in GitHub. Without these, **deploy and sync-models workflows will fail**, but **CI workflow will pass** (it doesn't use secrets).

---

## 📍 Where to Set Secrets

1. Go to your GitHub repository: https://github.com/Anas18102004/Strong-Seaweed
2. Click: **Settings** → **Secrets and variables** → **Actions**
3. Click: **New repository secret**
4. Enter Name and Secret value → **Add secret**

---

## 🔐 Required Secrets (6 Total)

### 1. `AWS_SSH_KEY` (REQUIRED - for deployment)

**What it is:** Private SSH key to authenticate with AWS server

**Where to get:**
```powershell
# On your Windows PC, read the pem file
Get-Content "C:\Users\Anas\Downloads\mohammad_anas.pem"
```

**Example format:**
```
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA1234567890...
[entire contents of .pem file]
...more lines...
-----END RSA PRIVATE KEY-----
```

**⚠️ Important:**
- Copy the **ENTIRE contents** (from `-----BEGIN` to `-----END`)
- Include all the lines in between
- Do NOT modify or truncate
- This is a secret - never share or commit to repo

---

### 2. `AWS_HOST` (REQUIRED - for deployment)

**What it is:** IP address of your AWS server

**Value:**
```
13.48.123.136
```

---

### 3. `AWS_USER` (REQUIRED - for deployment)

**What it is:** SSH username for AWS server

**Value:**
```
ubuntu
```

---

### 4. `DEPLOY_PATH` (REQUIRED - for deployment)

**What it is:** Path to repository directory on AWS server

**Value:**
```
/home/ubuntu/Strong Seaweed
```

---

### 5. `MONGODB_URI` (REQUIRED - for Node.js backend)

**What it is:** MongoDB connection string for the Node.js backend database

**Using MongoDB Atlas (Cloud - Recommended) ✅**

1. Go to https://www.mongodb.com/cloud/atlas
2. Create free account
3. Create a cluster (free tier available)
4. In "Database" section, click "Connect"
5. Select "Drivers" → Choose your language
6. Copy the connection string (looks like):
   ```
   mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/blueweave?retryWrites=true&w=majority
   ```
7. Replace `username` and `password` with your Atlas credentials
8. Replace `cluster0` with your actual cluster name

**Example Atlas URI:**
```
mongodb+srv://myuser:mypassword@cluster0.abc123.mongodb.net/blueweave?retryWrites=true&w=majority
```

**⚠️ Important:**
- Keep your credentials secure
- Never commit this URI to git
- Change password annually
- Set IP whitelist to allow your AWS server IP (13.48.123.136)

---

### 6. `JWT_SECRET` (REQUIRED - for Node.js authentication)

**What it is:** Secret key for signing JWT tokens

**Generate a strong random string:**

**Option 1: Use PowerShell (Windows)**
```powershell
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Maximum 256 }))
```

**Option 2: Use OpenSSL (if installed)**
```bash
openssl rand -base64 32
```

**Option 3: Use online generator**
Visit: https://generate-random.org/encryption-key-generator?count=1&bytes=32&uppercase=true&numbers=true&lowercase=true&symbols=true

**Example value:**
```
aB3cDeFgHiJkLmNoPqRsTuVwXyZ0123456789!@#$%^&*
```

⚠️ **Make it complex** - use uppercase, lowercase, numbers, special characters

---

## 📋 Quick Setup Checklist

| # | Secret Name | Value | Status |
|---|------------|-------|--------|
| 1 | `AWS_SSH_KEY` | Contents of mohammad_anas.pem | ❌ TODO |
| 2 | `AWS_HOST` | 13.48.123.136 | ❌ TODO |
| 3 | `AWS_USER` | ubuntu | ❌ TODO |
| 4 | `DEPLOY_PATH` | /home/ubuntu/Strong Seaweed | ❌ TODO |
| 5 | `MONGODB_URI` | mongodb+srv://user:pass@cluster.mongodb.net/blueweave | ❌ TODO |
| 6 | `JWT_SECRET` | [Generate strong random string] | ❌ TODO |

---

## 🔍 How to Set Each Secret

### Step 1: Copy AWS_SSH_KEY

**Windows PowerShell:**
```powershell
# Read the file and copy to clipboard
Get-Content "C:\Users\Anas\Downloads\mohammad_anas.pem" | Set-Clipboard

# Or display it to copy manually
Get-Content "C:\Users\Anas\Downloads\mohammad_anas.pem"
```

**Then in GitHub:**
1. Secrets → New repository secret
2. Name: `AWS_SSH_KEY`
3. Secret: Paste the entire contents (Ctrl+V)
4. Click "Add secret"

### Step 2-4: Copy Simple Values

**In GitHub for each of these:**

```
Name: AWS_HOST
Secret: 13.48.123.136
```

```
Name: AWS_USER
Secret: ubuntu
```

```
Name: DEPLOY_PATH
Secret: /home/ubuntu/Strong Seaweed
```

### Step 5: Set MongoDB URI (MongoDB Atlas)

**From your MongoDB Atlas connection string:**
```
Name: MONGODB_URI
Secret: mongodb+srv://your_username:your_password@cluster0.xxxxx.mongodb.net/blueweave?retryWrites=true&w=majority
```

**Example:**
```
Name: MONGODB_URI
Secret: mongodb+srv://anas:mySecurePassword123@seaweed-cluster.abc123.mongodb.net/blueweave?retryWrites=true&w=majority
```

### Step 6: Set JWT Secret

**Generate strong random string:**

**PowerShell method:**
```powershell
-join ((65..90) + (97..122) + (48..57) + (33,64,35,36,37,94,38,42) | Get-Random -Count 48 | ForEach-Object {[char]$_})
```

**Then in GitHub:**
```
Name: JWT_SECRET
Secret: [paste your generated string]
```

---

## ✅ Verification After Setting Secrets

### Check Secrets Are Set:

1. Go to GitHub → Settings → Secrets and variables → Actions
2. You should see all 6 secrets listed:
   - ✅ AWS_HOST
   - ✅ AWS_USER
   - ✅ AWS_SSH_KEY
   - ✅ DEPLOY_PATH
   - ✅ MONGODB_URI
   - ✅ JWT_SECRET

### Test Secrets:

1. Make a small commit and push to beta:
   ```bash
   git checkout beta
   echo "# Test" >> TEST.md
   git add TEST.md
   git commit -m "Test CI"
   git push origin beta
   ```

2. Go to GitHub → Actions
3. Watch the **CI** workflow:
   - Should PASS ✅ (doesn't use secrets)
   - Tests all 4 apps

4. If CI passes, secrets for deployment are likely correct

---

## 🚨 Common Issues & Fixes

### Issue 1: "Repository secret not found"

**Problem:** Secret name in workflow doesn't match GitHub secret name

**Solution:**
- Secret in GitHub: `AWS_SSH_KEY`
- Referenced in workflow: `${{ secrets.AWS_SSH_KEY }}`
- These MUST match (case-sensitive)

### Issue 2: "Permission denied (publickey)"

**Problem:** SSH key is incorrect or invalid

**Solution:**
1. Verify you copied the ENTIRE .pem file
2. Check that it starts with `-----BEGIN RSA PRIVATE KEY-----`
3. Check that it ends with `-----END RSA PRIVATE KEY-----`

### Issue 3: "No such file or directory: /home/ubuntu/Strong Seaweed"

**Problem:** DEPLOY_PATH is incorrect or repo not cloned yet

**Solution:**
1. SSH to server and verify path:
   ```bash
   ssh ubuntu@13.48.123.136
   ls /home/ubuntu/
   ```
2. If folder doesn't exist, clone it first:
   ```bash
   cd ~
   git clone https://github.com/Anas18102004/Strong-Seaweed.git "Strong Seaweed"
   cd "Strong Seaweed"
   git checkout beta
   ```

### Issue 4: "mongodb connection failed"

**Problem:** MongoDB not running or MONGODB_URI wrong

**Solution:**
1. Check if MongoDB is running:
   ```bash
   sudo systemctl status mongod
   ```
2. Start if not running:
   ```bash
   sudo systemctl start mongod
   ```
3. Test connection:
   ```bash
   mongosh --eval "db.adminCommand('ping')"
   ```

---

## 📊 Workflow Success Criteria

| Workflow | Requires Secrets | Status |
|----------|------------------|--------|
| CI (ci.yml) | ❌ NO | Works without secrets |
| Deploy (deploy.yml) | ✅ YES | Needs all 4 AWS secrets |
| Sync Models (sync-models.yml) | ✅ YES | Needs all 4 AWS secrets |

**Test order:**
1. ✅ Set all 6 secrets
2. ✅ Push to beta → CI should pass
3. ✅ Run Deploy workflow → Should work
4. ✅ Run Sync Models workflow → Should work

---

## 🔐 Security Best Practices

✅ **DO:**
- Store SSH key securely (delete local copy after uploading to GitHub)
- Use strong random JWT_SECRET
- Rotate JWT_SECRET periodically
- Use different secrets for dev/staging/prod (if multiple environments)

❌ **DON'T:**
- Commit secrets to git
- Share secrets in chat/email
- Use same secret across multiple repos
- Hardcode secrets in code
- Post screenshots showing secret values

---

## 📝 Example Secrets Setup

Here's what your GitHub Secrets page should look like after setting all:

```
Repository secrets
────────────────────────────────────────

AWS_HOST                   Updated 5 minutes ago
AWS_SSH_KEY                Updated 5 minutes ago  
AWS_USER                   Updated 5 minutes ago
DEPLOY_PATH                Updated 5 minutes ago
JWT_SECRET                 Updated 5 minutes ago
MONGODB_URI                Updated 5 minutes ago
```

---

## ✅ Full Setup Checklist

- [ ] Read AWS_SSH_KEY from `C:\Users\Anas\Downloads\mohammad_anas.pem`
- [ ] Created GitHub secret `AWS_SSH_KEY` with entire .pem contents
- [ ] Created GitHub secret `AWS_HOST` = `13.48.123.136`
- [ ] Created GitHub secret `AWS_USER` = `ubuntu`
- [ ] Created GitHub secret `DEPLOY_PATH` = `/home/ubuntu/Strong Seaweed`
- [ ] Created GitHub secret `MONGODB_URI` = `mongodb://127.0.0.1:27017/blueweave`
- [ ] Generated strong random JWT_SECRET
- [ ] Created GitHub secret `JWT_SECRET` = your random string
- [ ] Verified all 6 secrets appear in GitHub Settings
- [ ] Test push to beta branch
- [ ] CI workflow passes
- [ ] Ready to run Deploy workflow

---

## 🎯 Next Steps After Setting Secrets

1. **Push test commit to beta:**
   ```bash
   git checkout beta
   echo "# Secrets configured" >> TEST.md
   git add TEST.md
   git commit -m "Ready for CI testing"
   git push origin beta
   ```

2. **Watch CI run:** GitHub → Actions → Select CI workflow

3. **Once CI passes, test Deploy:**
   - Go to Actions → "Deploy to AWS"
   - Select service: `blueweave-frontend` (start small)
   - Check "Restart service": Yes
   - Click "Run workflow"

4. **Monitor deployment logs**

---

## 📞 Troubleshooting Commands

If deployment fails, SSH to server and debug:

```bash
# SSH to server
ssh -i "C:\Users\Anas\Downloads\mohammad_anas.pem" ubuntu@13.48.123.136

# Check repository
cd /home/ubuntu/"Strong Seaweed"
git status
git branch
git log --oneline -3

# Check secrets in environment
echo $MONGODB_URI
echo $JWT_SECRET

# Check service status
sudo systemctl status blueweave-backend
sudo journalctl -u blueweave-backend -n 20
```

---

**Your CI/CD pipeline is ready! Configure these secrets and you're good to go! 🚀**
