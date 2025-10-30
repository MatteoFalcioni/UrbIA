import modal
import json
from typing import Dict, Any

# Import the Modal app and image from app.py
from .app import app, image

class SandboxExecutor:
    """Manages per-session Modal Sandboxes with persistent state."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        
        # Create per-session volume for persistent workspace
        self.volume = modal.Volume.from_name(
            f"lg-urban-session-{session_id}", 
            create_if_missing=True
        )
        
        self.sandbox = modal.Sandbox.create(
            app=app,
            image=image,
            timeout=60*60*2,  # 2 hours session timeout
            idle_timeout=60*10,  # 10 min idle timeout
            volumes={"/workspace": self.volume},
            workdir="/workspace"
        )
        
        # CRITICAL: bufsize=1 for line buffering!
        self.process = self.sandbox.exec(
            "python", "/root/driver.py",
            bufsize=1  # ← Line buffering essenziale!
        )
    
    def execute(self, code: str, timeout: int = 120) -> Dict[str, Any]:
        """Execute code and return results.
        
        The driver handles all artifact scanning and S3 upload,
        so we just need to send the code and return the response.
        """
        try:
            # Send command to driver
            command = json.dumps({"code": code})
            self.process.stdin.write(command + "\n")
            self.process.stdin.drain()  # Ensure data is flushed - only use drain() instead of flush()
            
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
