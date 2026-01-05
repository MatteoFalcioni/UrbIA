import modal
import json
from typing import Dict, Any

# Import the Modal app and image from app.py
from .app import image
from .session import volume_name, session_base_dir


class SandboxExecutor:
    """Manages per-session Modal Sandboxes with persistent state."""

    def __init__(self, session_id: str, env: dict[str, str] | None = None):

        self.session_id = session_id
        self.env = env or {}
        
        print(f"[EXECUTOR] Initializing for session {session_id}")
        
        # Create per-session volume for persistent workspace
        print("[EXECUTOR] Getting volume...")
        self.volume = modal.Volume.from_name(volume_name(), create_if_missing=True)
        base_dir = session_base_dir(session_id)
        
        # Only load AWS secrets if S3 upload is not disabled
        secrets = []
        if self.env.get("S3_DISABLE_UPLOAD") != "1":
            secrets.append(modal.Secret.from_name("aws-credentials-IAM"))
        
        # Important: fixed sandbox creation by hydrating the Modal App inline (required outside Modal containers).
        print("[EXECUTOR] Looking up app...")
        hydrated_app = modal.App.lookup("lg-urban-executor", create_if_missing=True)
        
        print(f"[EXECUTOR] Creating sandbox (image: {image})...")
        self.sandbox = modal.Sandbox.create(
            app=hydrated_app,
            image=image,
            timeout=60 * 60 * 2,  # 2 hours session timeout
            idle_timeout=60 * 10,  # 10 min idle timeout
            volumes={"/workspace": self.volume},  # link it to above volume
            workdir=base_dir,  # NEW: per session cwd
            secrets=secrets,  # AWS creds for S3 uploads (optional)
        )
        print(f"[EXECUTOR] Sandbox created: {self.sandbox.object_id}")

        # ensure per-session dir exists before starting the driver
        print("[EXECUTOR] Creating base dir...")
        self.sandbox.exec("mkdir", "-p", base_dir).wait()

        # Start driver with optional environment variables
        print("[EXECUTOR] Starting driver process...")
        self.process = self.sandbox.exec(
            "python",
            "-u",  # Force unbuffered output to prevent hangs in CI
            "/root/driver.py",
            bufsize=0,  # CRITICAL: bufsize=0 for unbuffered I/O
            workdir=base_dir,  # Set working directory for driver process
            env=self.env,  # Pass custom env vars to driver process
        )
        print("[EXECUTOR] Driver started.")

    def execute(self, code: str, timeout: int = 120) -> Dict[str, Any]:
        """Execute code and return results.

        Args:
            code (str): The code to execute.
            timeout (int): The timeout for the execution.

        Returns:
            Dict[str, Any]: The result of the execution.
                - stdout (str): The standard output of the execution.
                - stderr (str): The standard error of the execution.
                - artifacts (list): The artifacts of the execution.
                - error (str): The error message if the execution fails.

        NOTE: The driver handles all artifact scanning and S3 upload, so we just need to send the code and return the response.
        """
        try:
            print(f"[EXECUTOR] Executing code (len={len(code)})...")
            # Send command to driver
            command = json.dumps({"code": code})
            command_with_newline = command + "\n"

            # Write in chunks to avoid buffer overflow for large datasets
            chunk_size = 8192  # 8KB chunks - safe size for Modal's buffer
            print(f"[EXECUTOR] Writing {len(command_with_newline)} bytes to stdin...")
            for i in range(0, len(command_with_newline), chunk_size):
                chunk = command_with_newline[i : i + chunk_size]
                self.process.stdin.write(chunk)
                self.process.stdin.drain()  # Flush after each chunk to prevent buffer overflow
            
            print("[EXECUTOR] Waiting for response from stdout...")
            # Read response line - use iter() to get next line from stdout
            result_line = next(iter(self.process.stdout), None)
            print(f"[EXECUTOR] Got response line: {result_line[:50] if result_line else 'None'}...")

            if not result_line:
                # Try to read stderr to see why the driver terminated
                print("[EXECUTOR] Stream closed, reading stderr...")
                stderr_lines = []
                try:
                    # Read all available stderr lines (non-blocking)
                    for line in self.process.stderr:
                        stderr_lines.append(line)
                        if len(stderr_lines) >= 50:  # Limit to prevent hanging
                            break
                except Exception:
                    pass
                
                stderr_output = "".join(stderr_lines) if stderr_lines else "No stderr output captured"
                print(f"[EXECUTOR] Stderr captured: {stderr_output}")
                
                # Check if process is still running
                try:
                    returncode = self.process.returncode
                    process_status = f"Process returncode: {returncode}"
                except Exception:
                    process_status = "Process status unknown"
                
                return {
                    "stdout": "",
                    "stderr": f"Driver process terminated unexpectedly.\n{process_status}\nDriver stderr:\n{stderr_output}",
                    "artifacts": [],
                }

            result = json.loads(result_line.strip())
            # Driver already handled artifacts, just return result
            return result

        except json.JSONDecodeError as e:
            return {
                "stdout": "",
                "stderr": f"Invalid JSON response from driver: {e}",
                "artifacts": [],
            }
        except Exception as e:
            print(f"[EXECUTOR] Exception during execution: {e}")
            return {
                "stdout": "",
                "stderr": f"Execution failed: {str(e)}",
                "artifacts": [],
            }

    def terminate(self):
        """Clean up sandbox and persist volume."""
        try:
            # Signal EOF to driver instead of closing
            self.process.stdin.write_eof()
            self.process.stdin.drain()  # Ensure EOF is sent

            # Wait for process to finish gracefully
            self.process.wait()

        except Exception as e:
            print(f"Error during graceful termination: {e}")

        finally:
            # Always terminate sandbox
            try:
                self.sandbox.terminate()
                self.sandbox.wait(raise_on_termination=False)
            except Exception as e:
                print(f"Error terminating sandbox: {e}")
