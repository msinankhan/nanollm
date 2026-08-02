import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

# -----------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of executing Python code in a sandbox."""
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False


# The guard runs in the subprocess before the untrusted code. It applies the
# resource limits and disables destructive functions to protect against
# accidents (a fork bomb, deleting files, killing other processes, ...).
# It is trivially bypassable by adversarial code, see docstring above.
GUARD = r"""
import faulthandler, builtins, os, shutil, subprocess, sys
maximum_memory_bytes = {maximum_memory_bytes}
if maximum_memory_bytes is not None and sys.platform != "darwin":
    # (the resource limit calls seem to fail on macOS, skip them there)
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
faulthandler.disable()
builtins.exit = None
builtins.quit = None
builtins.help = None
os.environ["OMP_NUM_THREADS"] = "1"
for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
             "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
             "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
             "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
    setattr(os, name, None)
for name in ("rmtree", "move", "chown"):
    setattr(shutil, name, None)
subprocess.Popen = None
for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
    sys.modules[name] = None
"""


def execute_code(
    code: str,
    timeout: float = 5.0, # 5 seconds default
    maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024, # 256MB default
) -> ExecutionResult:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute as a string
        timeout: Maximum execution time in seconds (default: 5.0)
        maximum_memory_bytes: Memory limit in bytes (default: 256MB, None to disable)

    Returns:
        ExecutionResult with success status, stdout/stderr, and error information

    Example:
        >>> result = execute_code("print('hello world')")
        >>> result.success
        True
        >>> result.stdout
        'hello world\\n'
    """
    # the guard runs first, then the untrusted code (with fresh globals, as a repr'd literal)
    guard = GUARD.format(maximum_memory_bytes=maximum_memory_bytes)
    program = guard + f"\nexec(compile({code!r}, '<llm>', 'exec'), {{'__name__': '__main__'}})\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            process = subprocess.run(
                [sys.executable, "-c", program],
                cwd=tmpdir, # writes land in the tempdir, deleted afterwards
                env={"PATH": "/usr/bin:/bin"}, # scrub the environment
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # subprocess.run kills the child process on timeout
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error="Execution timed out (process killed)",
                timeout=True,
            )

    success = process.returncode == 0
    stderr = process.stderr
    # the last line of the traceback identifies the exception, e.g. "TypeError: ..."
    error = None if success else (stderr.strip().splitlines() or ["Execution failed"])[-1]
    memory_exceeded = "MemoryError" in stderr
    result = ExecutionResult(
        success=success,
        stdout=process.stdout,
        stderr=stderr,
        error=error,
        memory_exceeded=memory_exceeded,
    )
    return result