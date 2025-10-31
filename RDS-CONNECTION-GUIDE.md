# RDS Connection Guide

## Overview

Our RDS database is **private** (not publicly accessible), which is the secure AWS best practice. To connect from laptop, we must use an **SSH tunnel** through a bastion host.

Tests automatically start tunnel if needed (see [Quick start in tests](./new_tests/QUICK-START-RDS.md)).

---

## Architecture

```
Your Laptop                AWS VPC
┌─────────────┐          ┌──────────────────────────────────┐
│             │          │                                  │
│  Tests/Code │          │  ┌──────────┐      ┌─────────┐   │
│      ↓      │  SSH     │  │ Bastion  │      │   RDS   │   │
│ localhost:  │─────────→│  │   EC2    │─────→│ Postgres│   │
│    5432     │  Tunnel  │  │          │      │         │   │
│             │          │  └──────────┘      └─────────┘   │
└─────────────┘          │                                  │
                         └──────────────────────────────────┘
```

---

## AWS Resources Created

### EC2 Bastion Host
- **Instance ID**: i-03509b694daca01ef (find in EC2 console)
- **Public IP**: 3.77.151.181  <- ⚠️ may change at restart! see later
- **Type**: t2.micro
- **Cost**: ~$8.50/month (or stop when not using)
- **Security Group**: bastion-ssh-access (SSH from your IP)

### RDS Database
- **Instance ID**: lg-urban-prod1
- **Endpoint**: lg-urban-prod1.cji2iikug9u5.eu-central-1.rds.amazonaws.com
- **Type**: db.t4g.micro
- **Database**: lgurban
- **User**: postgres
- **Publicly Accessible**: No ✅
- **Security Group**: ec2-rds-admin (allows bastion → RDS)

---

## (!) Cost Management (!)

### Current Monthly Cost
- **Bastion (t2.micro)**: ~$8.50/month
- **RDS (db.t4g.micro)**: ~$12-15/month
- **Total**: ~$20-24/month

### How to Reduce Costs

1. **Stop bastion when not using** (saves ~90% of bastion cost)
   ```bash
   aws ec2 stop-instances --instance-ids i-03509b694daca01ef
   aws ec2 start-instances --instance-ids i-03509b694daca01ef  # when needed
   ```
   
   ⚠️ **Note**: After restart, bastion's Public IP may change. Update IP in `~/start-rds-tunnel.sh` and `new_tests/conftest.py`. Find it in [console](https://eu-central-1.console.aws.amazon.com/ec2/home?region=eu-central-1#InstanceDetails:instanceId=i-03509b694daca01ef) under **Public IPv4 address**

3. **Stop RDS in dev environments** (production should stay running)

   ```bash
   aws rds stop-db-instance --db-instance-identifier lg-urban-prod1 # Saves money while stopped (only pay for storage, not compute)
   aws rds start-db-instance --db-instance-identifier lg-urban-prod1  # Start when you need it again (takes ~5 min to start)
   ```

3. **Use t4g.nano bastion** (~$3/month instead of $8.50)

---

## Quick Start

### Option 1: Automatic Tunnel (Recommended for Tests)

The tests will **automatically start the tunnel** if needed! Just run:

```bash
cd ~/LG-Urban

```
Set environment variables in `.env` file
```python
DATABASE_URL='postgresql+asyncpg://postgres:m4tt30f4lc@localhost:5432/lgurban'
ALEMBIC_DATABASE_URL='postgresql+psycopg2://postgres:m4tt30f4lc@localhost:5432/lgurban'
S3_BUCKET=lg-urban-prod
AWS_REGION=eu-central-1
```

Run tests - tunnel starts automatically!
```bash
PYTHONPATH=$(pwd) pytest new_tests/artifacts/
```

The pytest configuration will:
- ✅ Check if tunnel is already running
- ✅ Start it automatically if not
- ✅ **Leave it running for subsequent test runs** (faster!)

### Option 2: Manual Tunnel (For Other Work)

For non-test work (Alembic, psql, etc.), start the tunnel manually:

**Terminal 1:**
```bash
~/start-rds-tunnel.sh
```

**Leave this terminal window open.** The tunnel is running as long as you see the cursor hanging.

**Terminal 2:**
```bash
cd ~/LG-Urban

# Set environment variables
export DATABASE_URL='postgresql+asyncpg://postgres:m4tt30f4lc@localhost:5432/lgurban'
export ALEMBIC_DATABASE_URL='postgresql+psycopg2://postgres:m4tt30f4lc@localhost:5432/lgurban'

# Run Alembic migrations
alembic upgrade head

# Connect with psql
psql "$DATABASE_URL"
```

---

## Detailed Information

### RDS Instance Details

- **Instance ID**: `lg-urban-prod1`
- **Endpoint**: `lg-urban-prod1.cji2iikug9u5.eu-central-1.rds.amazonaws.com`
- **Port**: 5432
- **Database Name**: `lgurban`
- **Username**: `postgres`
- **Publicly Accessible**: No (secure)

### Bastion Host Details

- **Public IP**: `3.77.151.181`
- **Instance Type**: t2.micro
- **VPC**: `vpc-051e64accf516ba40`
- **Security Group**: `bastion-ssh-access`
- **SSH Key**: `~/.ssh/rds-bastion-key.pem`

### Tunnel Details

- **Local Port**: 5432 (on your laptop)
- **Remote Port**: 5432 (on RDS)
- **Connection**: `localhost:5432` → Bastion → RDS

---

## Resources

- **Quick Start**: `QUICK-START-RDS.md`
- **Full Guide**: `RDS-CONNECTION-GUIDE.md`
- **AWS Console**: https://console.aws.amazon.com/
- **Our RDS**: https://eu-central-1.console.aws.amazon.com/rds/home?region=eu-central-1#database:id=lg-urban-prod1

---

## Managing the Tunnels

### Automatic Tunnel 

The `new_tests/conftest.py` contains a pytest fixture that:
1. Checks if `DATABASE_URL` uses `localhost`
2. Tests if port 5432 is already open (tunnel running)
3. If not, starts SSH tunnel in background
4. Waits up to 10 seconds for connection
5. Leaves tunnel running after tests (for speed)

### Stopping the Automatic Tunnel

The tunnel runs in the background and won't show up in your terminal. To stop it:

```bash
# Find the SSH tunnel process
ps aux | grep "ssh.*5432.*lg-urban"

# Kill it (replace <PID> with the process ID)
kill <PID>
```
Or just restart your laptop - the tunnel will close automatically.

### Manual Tunnel (for Alembic, psql, etc.)

```bash
~/start-rds-tunnel.sh
```
Leave that terminal open, use another terminal for your work.

### Stop the Manual Tunnel
As for the automatic one:
```bash
# Find and kill the tunnel process
pkill -f "ssh.*5432.*lg-urban"
```
Or just restart your laptop.

### Checking Tunnel Status (both)

```bash
# Quick check: can we connect to localhost:5432?
nc -zv localhost 5432

# See if tunnel process is running
ps aux | grep "ssh.*5432.*lg-urban"
```

---

## Troubleshooting

### "Permission denied (publickey)"

Check SSH key permissions:
```bash
chmod 400 ~/.ssh/rds-bastion-key.pem
```

### "Connection refused" on localhost:5432

- Check that the tunnel is running (first terminal)
- Verify nothing else is using port 5432:
  ```bash
  sudo lsof -i :5432
  ```
- If you have local PostgreSQL, stop it or use a different port

### "Connection timeout" to RDS

- Verify bastion can reach RDS:
  ```bash
  ssh -i ~/.ssh/rds-bastion-key.pem ec2-user@3.77.151.181 # ⚠️ url may change at restart!
  timeout 5 bash -c 'cat < /dev/null > /dev/tcp/lg-urban-prod1.cji2iikug9u5.eu-central-1.rds.amazonaws.com/5432'
  exit
  ```

### Tests Still Timeout

- Make sure the tunnel is actually running
- Verify `DATABASE_URL` uses `localhost`, not the RDS hostname
- Check your laptop's firewall isn't blocking localhost connections

---

## Cost Management

### Bastion Host Costs

- **t2.micro**: ~$8.50/month (always on)
- **t4g.nano**: ~$3.00/month (ARM-based, cheaper)

### Savings Tips

- **Stop the bastion when not using it** (saves ~90% of costs)
- You can stop/start EC2 instances - the IP will change, update the script
- Consider using an **Elastic IP** ($0/month when attached) to keep a fixed IP

---

## Best Practices

### Daily Workflow

1. **Morning**: Start tunnel → `~/start-rds-tunnel.sh`
2. **Work**: New terminal → Set env vars → Code/test
3. **Evening**: Stop tunnel → `Ctrl+C` in tunnel terminal

### Security

- ✅ RDS is private (not exposed to internet)
- ✅ Bastion only allows SSH from your IP
- ✅ SSH key is password-protected
- ✅ All traffic is encrypted (SSH + TLS for PostgreSQL)

### Environment Variables

For simplicity we could also add to `.bashrc` or `.zshrc`:

```bash
# RDS tunnel alias
alias rds-tunnel='~/start-rds-tunnel.sh'

# Database connection helper
function rds-connect() {
    export DATABASE_URL='postgresql+asyncpg://postgres:m4tt30f4lc@localhost:5432/lgurban'
    export ALEMBIC_DATABASE_URL='postgresql+psycopg2://postgres:m4tt30f4lc@localhost:5432/lgurban'
    echo "✅ RDS environment variables set"
    echo "   DATABASE_URL=$DATABASE_URL"
}
```

Then just run:
```bash
rds-tunnel  # Start tunnel (in one terminal)
rds-connect # Set env vars (in another terminal)
```



