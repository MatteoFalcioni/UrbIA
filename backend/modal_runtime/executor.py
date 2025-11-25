import modal
import json
from typing import Dict, Any

# Import the Modal app and image from app.py
from .app import image
from .session import volume_name, session_base_dir

class SandboxExecutor:
    """Manages per-session Modal Sandboxes with persistent state."""
    
    def __init__(self, session_id: str):

        self.session_id = session_id
        # Create per-session volume for persistent workspace
        self.volume = modal.Volume.from_name(
            volume_name(), 
            create_if_missing=True
        )
        base_dir = session_base_dir(session_id)
        # Important: fixed sandbox creation by hydrating the Modal App inline (required outside Modal containers).
        hydrated_app = modal.App.lookup("lg-urban-executor", create_if_missing=True)
        self.sandbox = modal.Sandbox.create(
            app=hydrated_app,
            image=image,
            timeout=60*60*2,  # 2 hours session timeout
            idle_timeout=60*10,  # 10 min idle timeout
            volumes={"/workspace": self.volume},  # link it to above volume
            workdir=base_dir,  # NEW: per session cwd
            secrets=[modal.Secret.from_name("aws-credentials-IAM")]  # AWS creds for S3 uploads
        )

        # ensure per-session dir exists before starting the driver
        self.sandbox.exec("mkdir", "-p", base_dir).wait()
        
        self.process = self.sandbox.exec(
            "python", "/root/driver.py",
            bufsize=1,  # CRITICAL: bufsize=1 for line buffering!
            workdir=base_dir  # Set working directory for driver process
        )
    
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
            # Send command to driver
            command = json.dumps({"code": code})
            command_with_newline = command + "\n"
            
            # Write in chunks to avoid buffer overflow for large datasets
            chunk_size = 8192  # 8KB chunks - safe size for Modal's buffer
            for i in range(0, len(command_with_newline), chunk_size):
                chunk = command_with_newline[i:i + chunk_size]
                self.process.stdin.write(chunk)
                self.process.stdin.drain()  # Flush after each chunk to prevent buffer overflow
            
            # Read response line - use iter() to get next line from stdout
            result_line = next(iter(self.process.stdout), None)

            if not result_line:
                return {
                    "stdout": "",
                    "stderr": "Driver process terminated unexpectedly",
                    "artifacts": []
                }

            result = json.loads(result_line.strip())
            # Driver already handled artifacts, just return result
            return result
            
        except json.JSONDecodeError as e:
            return {
                "stdout": "",
                "stderr": f"Invalid JSON response from driver: {e}",
                "artifacts": []
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Execution failed: {str(e)}",
                "artifacts": []
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
