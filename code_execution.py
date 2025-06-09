import subprocess
import tempfile
import os
import uuid
import signal
from contextlib import contextmanager
from typing import List, Dict, Any


class TimeoutException(Exception):
    """Raised when code execution exceeds the time limit."""
    pass


@contextmanager
def time_limit(seconds: int):
    """Context manager to limit execution time."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def execute_python_code(code: str, test_cases: List[Dict[str, Any]] | None = None, timeout: int = 5) -> Dict[str, Any]:
    """Execute Python code in a temporary sandbox."""
    execution_id = str(uuid.uuid4())
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, f"{execution_id}.py")
        with open(file_path, "w") as f:
            f.write(code)

        try:
            with time_limit(timeout):
                proc = subprocess.run([
                    "python",
                    file_path,
                ], capture_output=True, text=True, timeout=timeout)
            result = {
                "output": proc.stdout,
                "errors": proc.stderr,
                "exit_code": proc.returncode,
            }
        except TimeoutException:
            return {"output": "", "errors": "Execution timed out", "exit_code": -1}
        except Exception as e:
            return {"output": "", "errors": str(e), "exit_code": -1}

        if test_cases:
            test_results: List[Dict[str, Any]] = []
            for test in test_cases:
                test_file = os.path.join(tmp_dir, f"test_{uuid.uuid4()}.py")
                with open(test_file, "w") as f:
                    f.write(f"import sys\nsys.path.append('{tmp_dir}')\n")
                    f.write(f"import {execution_id}\n\n")
                    f.write(test.get("code", ""))
                try:
                    test_proc = subprocess.run([
                        "python",
                        test_file,
                    ], capture_output=True, text=True, timeout=timeout)
                    test_results.append({
                        "name": test.get("name", "Unnamed Test"),
                        "passed": test_proc.returncode == 0,
                        "output": test_proc.stdout,
                        "errors": test_proc.stderr,
                    })
                except Exception as e:
                    test_results.append({
                        "name": test.get("name", "Unnamed Test"),
                        "passed": False,
                        "error": str(e),
                    })
            result["test_results"] = test_results
        return result


def execute_code(code: str, language: str = "python", test_cases: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Dispatch execution based on language."""
    language = (language or "python").lower()
    if language == "python":
        return execute_python_code(code, test_cases)
    return {
        "output": "",
        "errors": f"Language '{language}' not supported yet",
        "exit_code": -1,
    }
