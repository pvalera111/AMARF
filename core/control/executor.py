# core/control/executor.py
import subprocess
import shlex
import sys

class AMARFExecutor:
    def __init__(self):
        self.is_windows = sys.platform.startswith("win")
        self.telemetry = AMARFTelemetryEngine()

    def execute(self, validated_command: str, timeout: float = 7.0) -> dict:
        start_time = time.perf_counter()
        try:
            args = shlex.split(validated_command, posix=not self.is_windows)
            print(f"[AMARF-CONTROL] Spawning process: '{validated_command}'...")
            
            result = subprocess.run(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                text=True, timeout=timeout, shell=False
            )
            
            elapsed_time = (time.perf_counter() - start_time) * 1000
            status = "success" if result.returncode == 0 else "failed"
            
            self.telemetry.log_event(
                module_name="CONTROL", status_code=status, 
                execution_time_ms=elapsed_time,
                payload={"command": validated_command, "return_code": result.returncode}
            )
            return {
                "status": status, "return_code": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip(), 
                "error_context": None
            }
        except subprocess.TimeoutExpired:
            elapsed_time = (time.perf_counter() - start_time) * 1000
            self.telemetry.log_event(
                module_name="CONTROL", status_code="timeout", 
                execution_time_ms=elapsed_time, payload={"command": validated_command}
            )
            return {"status": "timeout", "return_code": -1, "stdout": "", 
                    "stderr": "", "error_context": "Timeout"}
        except Exception as e:
            elapsed_time = (time.perf_counter() - start_time) * 1000
            self.telemetry.log_event(
                module_name="CONTROL", status_code="error", 
                execution_time_ms=elapsed_time, 
                payload={"command": validated_command, "error": str(e)}
            )
            return {"status": "error", "return_code": -1, "stdout": "", 
                    "stderr": "", "error_context": str(e)}
