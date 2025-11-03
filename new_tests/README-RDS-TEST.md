# RDS Quick Start

## TL;DR - Run Tests

```bash
cd ~/LG-Urban

# Set environment variables
export DATABASE_URL='postgresql+asyncpg://postgres:m4tt30f4lc@localhost:5432/lgurban'
export S3_BUCKET=lg-urban-prod
export AWS_REGION=eu-central-1

# Run tests - tunnel starts automatically! 🚀
PYTHONPATH=$(pwd) pytest new_tests/artifacts/
```

**That's it!** The SSH tunnel to RDS will start automatically if needed.

---

## How It Works

- ✅ Tests check if localhost:5432 is accessible
- ✅ If not, automatically starts SSH tunnel via bastion host
- ✅ Tunnel stays running after tests (faster subsequent runs)
- ✅ Works seamlessly in background

---

## Manual Tunnel (for Alembic, psql, etc.)

```bash
~/start-rds-tunnel.sh
```

Leave that terminal open, use another terminal for your work.

---

## Stop the Tunnel

```bash
# Find and kill the tunnel process
pkill -f "ssh.*5432.*lg-urban"
```

Or just restart your laptop.

---

## Check Tunnel Status

```bash
# Quick check
nc -zv localhost 5432

# Or
ps aux | grep "ssh.*5432.*lg-urban"
```

---

## Full Documentation

See `RDS-CONNECTION-GUIDE.md` for complete details, troubleshooting, and architecture.

---

## Database Details

- **Host:** localhost:5432 (via tunnel)
- **Database:** lgurban
- **User:** postgres
- **Bastion:** 3.122.52.220
- **RDS:** lg-urban-prod1.cji2iikug9u5.eu-central-1.rds.amazonaws.com

