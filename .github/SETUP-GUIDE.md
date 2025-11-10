# GitHub Actions CI/CD Setup Guide

This guide will walk you through setting up GitHub Actions for your LG-Urban project.

## 📋 What the Workflow Does

The CI/CD pipeline includes:

1. **Change Detection**: Only runs relevant tests based on what files changed
2. **Modal Function Tests**: Tests your Modal runtime integration
3. **Database Migration Tests**: Validates Alembic migrations with Docker Postgres
4. **API Health Tests**: Tests core API endpoints
5. **Code Linting**: Checks code quality with ruff and black
6. **Automatic Modal Deployment**: Deploys Modal functions when changes are pushed to main

## 🔐 Step 1: Add GitHub Secrets

Go to your repository on GitHub:
1. Click **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** for each of these:

### Required Secrets

```bash
# Modal credentials (required for Modal tests and deployment)
MODAL_TOKEN_ID          # Your Modal token ID
MODAL_TOKEN_SECRET      # Your Modal token secret

# AWS credentials (required for S3 tests)
AWS_ACCESS_KEY_ID       # Your AWS access key
AWS_SECRET_ACCESS_KEY   # Your AWS secret key
AWS_REGION              # e.g., eu-central-1
S3_BUCKET               # Your S3 bucket name (e.g., lg-urban-prod)

# Encryption (required for API tests with user API keys)
ENCRYPTION_KEY          # Fernet encryption key for API key storage

# OpenAI (optional, for API tests that need LLM)
OPENAI_API_KEY          # Your OpenAI API key (can use a limited-quota key for tests)
```

### How to Generate ENCRYPTION_KEY

```bash
python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

Copy the output and add it as a GitHub secret. This key encrypts user API keys in the database.

### How to Get Your Modal Tokens

```bash
# If you have Modal CLI configured locally:
cat ~/.modal.toml

# Or create new tokens at: https://modal.com/settings/tokens
```

## 🧪 Step 2: Test Locally (Optional but Recommended)

Before pushing to GitHub, you can test parts of the workflow locally:

### Test API endpoints with local Docker Postgres:

```bash
# Start Postgres
cd infra
docker compose up -d db

# Wait for it to be ready
docker exec chat_pg pg_isready -U postgres

# Run migrations
cd ..
export ALEMBIC_DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/chat"
alembic upgrade head

# Run API tests
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/chat"
pytest tests/api/ -v

# Cleanup
cd infra
docker compose down -v
```

### Test Modal functions:

```bash
# Make sure you have Modal tokens in .env
export MODAL_TOKEN_ID="your-token-id"
export MODAL_TOKEN_SECRET="your-token-secret"

# Run Modal tests
pytest backend/modal_runtime/tests/ -v
```

### Test migrations:

```bash
cd infra
docker compose up -d db

# Wait and run migrations
cd ..
export ALEMBIC_DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/chat"
alembic upgrade head
alembic current

# Test downgrade/upgrade cycle
alembic downgrade -1
alembic upgrade head

# Cleanup
cd infra
docker compose down -v
```

## 📤 Step 3: Push to GitHub

Once secrets are added, commit and push your workflow:

```bash
git add .github/
git add tests/api/
git commit -m "Add GitHub Actions CI/CD pipeline"
git push origin main  # or your current branch
```

## 👀 Step 4: Monitor the Workflow

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. You should see your workflow running
4. Click on the running workflow to see detailed logs

## 🎯 How Different Triggers Work

### On Pull Request:
- ✅ Runs tests for changed components only
- ❌ Does NOT deploy to Modal
- Fast feedback for code review

### On Push to Main:
- ✅ Runs all tests
- ✅ Deploys Modal functions (only if Modal files changed)
- ✅ Railway auto-deploys backend (separate from this workflow)

## 🔧 Workflow Behavior

### Smart Change Detection

The workflow only runs tests for the parts you changed:

- **Modal tests** run if: `backend/modal_runtime/**` changed
- **Migration tests** run if: `backend/db/alembic/versions/**` or `backend/db/models.py` changed
- **API tests** run if: `backend/**` or `requirements.txt` changed
- **All tests** run on push to main (safety check)

### Modal Deployment

Modal functions are deployed **only when**:
1. Push is to `main` branch (not PRs)
2. Files in `backend/modal_runtime/` have changed
3. All Modal tests pass

This prevents unnecessary deployments and keeps Modal in sync with your backend.

## 🐛 Troubleshooting

### "Modal tokens not configured"

Add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` to GitHub secrets.

### "Migration tests failing"

Check that your migration files are valid:
```bash
alembic check  # locally
```

### "API tests failing"

Ensure the database schema is up to date:
- Check if you need to create a new migration
- Verify all models are properly imported in `backend/db/models.py`

### "Workflow not appearing"

- Make sure the file is at `.github/workflows/ci-cd.yml`
- Check for YAML syntax errors: https://www.yamllint.com/

### "Docker Compose failing in CI"

The workflow uses `infra/docker-compose.yml`. If you change the compose file, make sure it still works with default values (no custom .env needed in CI).

## 📊 Viewing Test Results

GitHub Actions shows test results in the UI:
- Green checkmark ✅ = All tests passed
- Red X ❌ = Some tests failed
- Click on a failed job to see which test failed and why

## 🔄 Next Steps

After the workflow is running:

1. **Badge**: Add a status badge to your README:
   ```markdown
   ![CI/CD](https://github.com/YOUR_USERNAME/LG-Urban/workflows/CI%2FCD%20Pipeline/badge.svg)
   ```

2. **Branch Protection**: Require tests to pass before merging:
   - Settings → Branches → Add rule for `main`
   - Enable "Require status checks to pass"

3. **Notifications**: Set up Slack/Discord notifications:
   - Add a notification step to the workflow
   - Use GitHub Actions marketplace integrations

## 💡 Tips

- Run tests locally before pushing to save CI minutes
- Use draft PRs to test workflow changes without triggering full CI
- Check the Actions tab regularly to catch issues early
- Modal deployment happens automatically - no manual `modal deploy` needed!

## 🆘 Need Help?

- Check workflow run logs in the Actions tab
- GitHub Actions docs: https://docs.github.com/en/actions
- Modal docs: https://modal.com/docs

