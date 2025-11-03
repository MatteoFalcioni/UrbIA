import sys
import socket
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv
import os
import pytest

# Ensure imports like `backend.*` work regardless of where pytest is launched from
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Load .env early so backend/db/session.py sees DATABASE_URL at import time
env_path = repo_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is open and accepting connections."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def ensure_rds_tunnel():
    """
    Automatically start SSH tunnel to RDS if not already running.
    
    This checks if localhost:5432 is accessible. If not, it starts
    the SSH tunnel in the background using the start-rds-tunnel.sh script.
    
    The tunnel is left running after tests complete for convenience.
    """
    # Skip tunnel management if DATABASE_URL doesn't point to localhost
    db_url = os.getenv("DATABASE_URL", "")
    if "localhost" not in db_url and "127.0.0.1" not in db_url:
        # Not using tunnel, skip
        yield
        return
    # Check if tunnel already running
    if is_port_open("localhost", 5432, timeout=2.0):
        print("\n✅ RDS tunnel already running (localhost:5432 is accessible)")
        yield
        return
    
    # Start the tunnel
    print("\n🚀 Starting RDS tunnel...")
    
    tunnel_script = Path.home() / "start-rds-tunnel.sh"
    if not tunnel_script.exists():
        print(f"⚠️  Warning: {tunnel_script} not found. Please start tunnel manually.")
        yield
        return
    
    # Start tunnel in background
    try:
        # SSH tunnel as background process
        ssh_key = Path.home() / ".ssh" / "rds-bastion1.pem"
        bastion_ip = "3.77.151.181"
        rds_endpoint = "lg-urban-prod1.cji2iikug9u5.eu-central-1.rds.amazonaws.com"
        
        tunnel_process = subprocess.Popen(
            [
                "ssh",
                "-i", str(ssh_key),
                "-L", f"5432:{rds_endpoint}:5432",
                "-N",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=60",
                "-o", "ServerAliveCountMax=3",
                f"ec2-user@{bastion_ip}"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Detach from parent
        )
        
        # Wait for tunnel to be ready (max 10 seconds)
        print("⏳ Waiting for tunnel to connect...", end="", flush=True)
        for i in range(20):  # 20 * 0.5s = 10s timeout
            time.sleep(0.5)
            if is_port_open("localhost", 5432, timeout=1.0):
                print(" ✅ Connected!")
                break
            print(".", end="", flush=True)
        else:
            print(" ❌ Timeout")
            print("⚠️  Tunnel may still be connecting. If tests fail, start manually:")
            print(f"    ~/start-rds-tunnel.sh")
        
        yield
        
        # Note: We intentionally leave the tunnel running after tests
        # This is better for developer experience (no need to restart for each test run)
        # The tunnel will close when the SSH connection drops or laptop restarts
        print("\n💡 RDS tunnel left running for convenience (close with Ctrl+C in tunnel terminal)")
        
    except Exception as e:
        print(f"\n⚠️  Failed to start tunnel automatically: {e}")
        print("Please start tunnel manually:")
        print(f"    ~/start-rds-tunnel.sh")
        yield



