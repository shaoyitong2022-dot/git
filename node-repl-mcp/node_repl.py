"""Core Node REPL wrapper — pure stdlib, no FastMCP dependency.
Exposes the same tool semantics as the node_repl MCP server.
"""
import subprocess
import asyncio
import os
import sys

_dir_blackhole = os.devnull if os.name == "nt" else "/dev/null"

class NodeKernel:
    """Persistent Node.js REPL kernel."""

    def __init__(self):
        self._proc = None
        self._module_dirs = []

    def _spawn(self):
        env = os.environ.copy()
        if self._module_dirs:
            sep = ";" if os.name == "nt" else ":"
            existing = env.get("NODE_PATH", "")
            node_path = sep.join(self._module_dirs)
            if existing:
                node_path = node_path + sep + existing
            env["NODE_PATH"] = node_path
        self._proc = subprocess.Popen(
            ["node", "--interactive"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=0,
        )
        # drain banner
        try:
            self._read_until_prompt(timeout=2)
        except asyncio.TimeoutError:
            pass

    def _read_until_prompt(self, timeout: float = 30.0) -> str:
        """Read all output up to (but not including) the next REPL prompt."""
        buf = []
        start = asyncio.get_event_loop().time()
        prompt_seen = set()
        while True:
            ch = None
            try:
                ch = self._proc.stdout.read(1)
            except Exception:
                break
            if ch == "" or ch is None:
                break
            buf.append(ch)
            prompt_seen = set(p for p in prompt_seen if not ("".join(buf)).endswith(p))
            for p in ["> ", "... "]:
                if "".join(buf).endswith(p):
                    prompt_seen.add(p)
            if "> " in prompt_seen:
                break
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                raise asyncio.TimeoutError(f"No prompt after {timeout}s")
        result = "".join(buf)
        # strip trailing prompts
        for p in ["> ", "... "]:
            if result.endswith(p):
                result = result[:-len(p)]
        return result.strip()

    async def eval_js(self, code: str, timeout_ms: int = 30000) -> str:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()
        self._read_until_prompt(timeout=0.5)  # clear stray output
        self._proc.stdin.write(code + "\n")
        self._proc.stdin.flush()
        await asyncio.sleep(0.2)
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._read_until_prompt, 25.0),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            return f"Execution timed out after {timeout_ms}ms"
        return result or "(no output)"

    def add_module_dir(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            raise ValueError(f"Not a valid directory: {abs_path}")
        if abs_path not in self._module_dirs:
            self._module_dirs.append(abs_path)
            if self._proc:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                self._proc = None
            return True
        return False

    def reset(self) -> str:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        return "Kernel reset"

_kernel = None

def get_kernel() -> NodeKernel:
    global _kernel
    if _kernel is None:
        _kernel = NodeKernel()
    return _kernel

# ---- Tool functions (same names as MCP server) ----

async def js(code: str, timeout_ms: int = 30000, title: str = "") -> str:
    return await get_kernel().eval_js(code, timeout_ms)

async def js_add_node_module_dir(path: str) -> bool:
    return get_kernel().add_module_dir(path)

async def js_reset() -> str:
    return get_kernel().reset()

