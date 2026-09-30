"""
Serve the Unsloth API (no web UI) from a Jupyter notebook and expose it via a Cloudflare
quick tunnel. Works on Google Colab and on other Linux Jupyter environments such as the
Jupyter Docker Stacks (non-root, no sudo, curl optional).

The code always runs inside Linux (x86_64 or arm64): on Colab, or in a Linux container on a
Linux, Windows or macOS host (see compose.yaml / compose.gpu.yaml). Python >= 3.10 and outbound
internet access are required. NVIDIA GPUs are usable on Linux and Windows hosts; Docker on
macOS has no GPU access, so Unsloth runs GGUF models on the CPU there.
check_environment() reports architecture, GPU, memory and disk and warns about common issues.
Logs go to /content/logs on Colab and to ./logs elsewhere.

Design: no module-level state and only local imports. Shared logic lives in small helper
functions (prefixed with "_"); to move a public function elsewhere, copy it together with
the helpers it calls. State (API keys, ports, processes) is passed via return values.

Model specs are plain dicts:
    {"repo": "<hf repo id>", "file": "<file or glob to download>", "variant": "<GGUF quant tag>"}
"file" and "variant" are only needed for GGUF repos.

----------------------------------------------------------------------------------------------
Everything at once (recommended):

    stack = run_llms(
        model={"repo": "unsloth/Qwen3.5-4B-GGUF",
               "file": "Qwen3.5-4B-UD-Q4_K_XL.gguf", "variant": "UD-Q4_K_XL"},
        embedding_model={"repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
                         "file": "nomic-embed-text-v1.5.f16.gguf", "variant": "f16"},
    )
    healthcheck(stack)                               # call as often as you like
    test_chat(stack["public_url"], stack["api_key"])
    test_embeddings(stack["public_url"], stack["api_key"],
                    model=stack["embedding_model_name"])
    shutdown(stack)

----------------------------------------------------------------------------------------------
Step by step:

    install_unsloth(); install_hf(); install_cloudflared()

    # Unsloth GGUF chat model (llama.cpp)
    download_model("unsloth/Qwen3.5-4B-GGUF", include="Qwen3.5-4B-UD-Q4_K_XL.gguf")
    server = serve_unsloth("unsloth/Qwen3.5-4B-GGUF", "UD-Q4_K_XL")      # port 8900 by default

    # Non-Unsloth safetensors model (transformers, 4-bit by default, needs a GPU): no file/variant
    #   download_model("Qwen/Qwen3.5-4B"); server = serve_unsloth("Qwen/Qwen3.5-4B")

    # Embedding model as the only loaded model
    #   server = serve_unsloth("nomic-ai/nomic-embed-text-v1.5-GGUF", "f16")

    # Chat model plus embedding model side by side
    #   server = serve_unsloth("unsloth/Qwen3.5-4B-GGUF", "UD-Q4_K_XL",
    #                          embedding_model={"repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
    #                                           "variant": "f16"})

    tunnel = expose_with_cloudflared(server["port"], server["api_key"])
    test_chat(tunnel["url"], server["api_key"])

    configure_model_switching(server["port"], server["api_key"])        # "Switch model by request"
    load_model(server["port"], server["api_key"], "Qwen/Qwen3.5-4B")    # swap model, no restart
    stop_process(tunnel["process"]); stop_process(server["process"])
"""


# =========================================================================== helpers

def _find_exe(name: str) -> str | None:
    """Locate an executable on PATH or in ~/.local/bin (where the installers put them)."""
    import os
    import shutil
    from pathlib import Path

    return shutil.which(name, path=f"{Path.home() / '.local' / 'bin'}:{os.environ['PATH']}")


def _require_exe(name: str, install_hint: str) -> str:
    exe = _find_exe(name)
    if not exe:
        raise RuntimeError(f"'{name}' not found - run {install_hint}() first.")
    return exe


def _log_path(name: str) -> str:
    """Default log file: /content/logs on Colab, ./logs elsewhere."""
    import importlib.util
    from pathlib import Path

    on_colab = importlib.util.find_spec("google.colab") is not None
    base = Path("/content") if on_colab else Path.cwd()
    return str(base / "logs" / f"{name}.log")


def _fetch(url: str, dest) -> str:
    """Download `url` to `dest` with the standard library (no curl/wget needed)."""
    import shutil
    import urllib.request
    from pathlib import Path

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "unsloth-jupyter/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)
    return str(dest)


def _run(cmd: list[str], env: dict | None = None) -> None:
    """Run a command to completion; on failure raise with the last lines of output."""
    import os
    import subprocess

    r = subprocess.run(cmd, text=True, capture_output=True, env={**os.environ, **(env or {})})
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-30:])
        raise RuntimeError(f"Command failed (exit {r.returncode}): {' '.join(cmd)}\n{tail}")


def _install(name: str, install_fn, is_complete=None) -> str:
    """Call `install_fn()` unless `name` is already available (and `is_complete()` agrees).
    Returns the executable path."""
    ready = lambda: _find_exe(name) and (is_complete is None or is_complete())
    if ready():
        exe = _find_exe(name)
        print(f"✅ {name} already installed: {exe}")
        return exe
    if _find_exe(name):
        print(f"⚠️ Incomplete {name} installation found - running the installer again.")
    print(f"⏳ Installing {name}...")
    install_fn()
    if not ready():
        raise RuntimeError(f"Installer for {name} finished, but the installation is incomplete.")
    exe = _find_exe(name)
    print(f"✅ {name} installed: {exe}")
    return exe


def _start_background(cmd: list[str], log_file: str, ready_regex: str,
                      timeout: float, env: dict | None = None):
    """Start a long-running process, mirror its output to `log_file` and block until a line
    matches `ready_regex`. Output keeps being drained afterwards, so the process never stalls
    on a full pipe. Returns (process, match)."""
    import os
    import re
    import subprocess
    import threading
    from pathlib import Path

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd, text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})},
    )
    pattern, ready, found = re.compile(ready_regex), threading.Event(), {}

    def pump():
        with open(log_path, "w") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if not ready.is_set() and (m := pattern.search(line)):
                    found["match"] = m
                    ready.set()
        ready.set()  # process ended -> wake up the waiter

    threading.Thread(target=pump, daemon=True).start()

    if not ready.wait(timeout):
        raise TimeoutError(f"Not ready after {timeout}s - see {log_path} (process still running).")
    if "match" not in found:
        tail = "\n".join(log_path.read_text().splitlines()[-30:])
        raise RuntimeError(f"Process exited with code {proc.poll()} before becoming ready:\n{tail}")
    return proc, found["match"]


def _request(method: str, url: str, api_key: str | None = None, payload: dict | None = None,
             timeout: float = 300) -> dict:
    """JSON request against the Unsloth API (standard library only); raises on HTTP errors."""
    import json
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"User-Agent": "unsloth-jupyter/1.0"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        body = exc.read()[:300].decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from None


def _http_status(url: str, api_key: str | None = None, timeout: float = 10) -> int:
    """Status code of a GET request (errors are returned, not raised)."""
    import urllib.error
    import urllib.request

    headers = {"User-Agent": "unsloth-jupyter/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _embedding_env(embedding_model: str | dict | None) -> dict:
    """Environment variables that configure Unsloth's own embedder (served by /v1/embeddings
    while a non-embedding model is loaded)."""
    if not embedding_model:
        return {}
    if isinstance(embedding_model, str):
        return {"RAG_EMBEDDING_MODEL": embedding_model}
    repo = embedding_model["repo"]
    env = {"RAG_EMBEDDING_MODEL": embedding_model.get("name") or repo.removesuffix("-GGUF")}
    if repo.endswith("-GGUF"):
        env["RAG_EMBED_GGUF_REPO"] = repo
    if embedding_model.get("variant"):
        env["RAG_EMBED_GGUF_VARIANT"] = embedding_model["variant"]
    return env


def _print_client_settings(url: str, api_key: str | None) -> None:
    print("\n✅ Setup complete. Use these on your client machine:")
    print("-" * 60)
    print(f"export LLM_BASE_URL='{url}'")
    print(f"export API_KEY='{api_key or 'YOUR_API_KEY'}'")
    print("-" * 60)
    print("Note: quick tunnels don't pass streaming (SSE) reliably; use stream=false.")


# =========================================================================== installers

def install_unsloth() -> str:
    """Install Unsloth Studio (own venv + prebuilt llama.cpp) with the official installer.
    Needs no root: everything goes to ~/.unsloth and ~/.local/bin."""
    import tempfile
    from pathlib import Path

    def install():
        script = _fetch("https://unsloth.ai/install.sh", Path(tempfile.gettempdir()) / "unsloth-install.sh")
        _run(["sh", script], env={"UNSLOTH_SKIP_AUTOSTART": "1"})

    def is_complete():
        # The `unsloth` launcher is created early; the prebuilt llama.cpp server comes late
        # and is what the API needs, so its presence marks a finished installation.
        llama = Path.home() / ".unsloth" / "llama.cpp"
        return any((d / "llama-server").is_file() for d in (llama, llama / "build" / "bin"))

    return _install("unsloth", install, is_complete)


def install_hf() -> str:
    """Install the Hugging Face CLI (`hf`) into ~/.local/bin.
    Uses the official installer when curl exists (it needs curl), otherwise a private venv."""
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    def install():
        if shutil.which("curl"):
            script = _fetch("https://hf.co/cli/install.sh", Path(tempfile.gettempdir()) / "hf-install.sh")
            _run(["bash", script])
        else:
            venv = Path.home() / ".local" / "share" / "hf-cli"
            _run([sys.executable, "-m", "venv", str(venv)])
            _run([str(venv / "bin" / "pip"), "install", "--quiet", "--upgrade", "huggingface_hub"])
            link = Path.home() / ".local" / "bin" / "hf"
            link.parent.mkdir(parents=True, exist_ok=True)
            link.unlink(missing_ok=True)
            link.symlink_to(venv / "bin" / "hf")

    return _install("hf", install)


def install_cloudflared() -> str:
    """Install cloudflared as a static binary into ~/.local/bin (no apt/sudo/curl)."""
    import platform
    from pathlib import Path

    def install():
        arch = {"x86_64": "amd64", "amd64": "amd64",
                "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
        if platform.system() != "Linux" or not arch:
            raise RuntimeError(f"Unsupported platform: {platform.system()} {platform.machine()}")
        target = Path.home() / ".local" / "bin" / "cloudflared"
        _fetch("https://github.com/cloudflare/cloudflared/releases/latest/download/"
               f"cloudflared-linux-{arch}", target)
        target.chmod(0o755)

    return _install("cloudflared", install)


# =========================================================================== models

def download_model(repo_id: str, include: str | None = None, quiet: bool = False) -> None:
    """Download into the Hugging Face cache, where Unsloth finds it automatically.
    GGUF repos: pass the file (or a glob) via `include` to avoid fetching every quant.
    Safetensors repos: leave `include` empty to get the whole model.
    Gated models: set HF_TOKEN (e.g. from Colab secrets) first."""
    import subprocess

    hf = _require_exe("hf", "install_hf")
    cmd = [hf, "download", repo_id] + (["--include", include] if include else [])
    label = f"{repo_id}{f' ({include})' if include else ''}"
    print(f"⏳ Downloading {label} ...")
    r = subprocess.run(cmd, text=True, capture_output=quiet)
    if r.returncode != 0:
        raise RuntimeError(f"Download of {label} failed (exit {r.returncode}):\n"
                           + ((r.stderr or "")[-2000:] if quiet else ""))
    print(f"✅ Downloaded {label}")


def load_model(port: int, api_key: str, repo_id: str, variant: str | None = None,
               max_seq_length: int = 0, timeout: float = 1800) -> dict:
    """Swap the model on a running server. `variant` only for GGUF repos.
    Talks to the local Unsloth port directly (not through the tunnel)."""
    payload = {"model_path": repo_id, "max_seq_length": max_seq_length, "load_in_4bit": True}
    if variant:
        payload["gguf_variant"] = variant
    print(f"⏳ Loading {repo_id}{':' + variant if variant else ''} ...")
    result = _request("POST", f"http://127.0.0.1:{port}/api/inference/load", api_key, payload, timeout)
    print("✅ Model loaded.")
    return result


def configure_model_switching(port: int, api_key: str, enabled: bool = True,
                              auto_download: bool | None = None,
                              idle_unload_seconds: int | None = None) -> dict:
    """Set Unsloth's "Switch model by request" (Settings > API in the web UI).

    enabled:             a request whose "model" names another downloaded model loads it
                         (replacing the current one; the first such request waits for the load)
    auto_download:       also start downloading a GGUF model named in a request that is not
                         downloaded yet (needs `enabled`); None leaves the setting unchanged
    idle_unload_seconds: unload the model after this many idle seconds (0 = off, minimum 60);
                         None leaves the setting unchanged

    The setting is stored by Unsloth and survives server restarts."""
    payload: dict = {"enabled": enabled}
    if auto_download is not None:
        payload["auto_download_model"] = auto_download
    if idle_unload_seconds is not None:
        payload["auto_unload_idle_seconds"] = idle_unload_seconds
    result = _request("PUT", f"http://127.0.0.1:{port}/api/settings/openai-auto-switch",
                      api_key, payload, timeout=30)
    print(f"✅ Switch model by request: {'on' if result.get('enabled') else 'off'}"
          f" (auto-download {'on' if result.get('auto_download_model') else 'off'})")
    return result


# =========================================================================== serving

def serve_unsloth(model: str, variant: str | None = None, port: int = 8900,
                  embedding_model: str | dict | None = None, extra_args: list[str] | None = None,
                  log_file: str | None = None, timeout: float = 1800) -> dict:
    """Start the headless Unsloth API with `model` loaded and block until it is ready.

    - GGUF repo: pass the quant tag as `variant` (e.g. "UD-Q4_K_XL").
    - Safetensors repo (e.g. "Qwen/Qwen3.5-4B"): leave `variant` as None.
    - `embedding_model`: Unsloth's own embedder, used by /v1/embeddings while a
      non-embedding model is loaded. A model spec dict {"repo", "variant"} or a model name.

    Binds to localhost only (cloudflared is the only way in) and disables server-side
    tools, because the endpoint will be public. The default port 8900 avoids Jupyter's 8888.
    Returns {"api_key", "port", "process"}."""
    import re
    from pathlib import Path

    exe = _require_exe("unsloth", "install_unsloth")
    log_file = log_file or _log_path("unsloth")
    model_arg = f"{model}:{variant}" if variant else model
    cmd = [exe, "run", "--model", model_arg, "--api-only", "--silent",
           "-H", "127.0.0.1", "-p", str(port), "--disable-tools", *(extra_args or [])]

    print(f"⏳ Starting Unsloth API with {model_arg} (downloads the model if needed)...")
    proc, match = _start_background(cmd, log_file, r"API Key:\s+(\S+)", timeout,
                                    _embedding_env(embedding_model))

    # The server moves to the next free port if the requested one is taken.
    ports = re.findall(r"http://(?:127\.0\.0\.1|localhost):(\d+)", Path(log_file).read_text())
    actual_port = int(ports[-1]) if ports else port
    print(f"✅ Unsloth API running on http://127.0.0.1:{actual_port}/v1")
    return {"api_key": match.group(1), "port": actual_port, "process": proc}


def expose_with_cloudflared(port: int, api_key: str | None = None,
                            log_file: str | None = None,
                            timeout: float = 120, print_settings: bool = True) -> dict:
    """Open a Cloudflare quick tunnel to localhost:`port`. Returns {"url", "process"}.
    The target does not need to be up yet; requests fail with 502 until it is."""
    exe = _require_exe("cloudflared", "install_cloudflared")
    cmd = [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}",
           "--http-host-header", f"localhost:{port}"]

    print("⏳ Starting cloudflared tunnel...")
    proc, match = _start_background(cmd, log_file or _log_path("cloudflared"),
                                    r"https://[a-z0-9-]+\.trycloudflare\.com", timeout)
    url = match.group(0)
    print(f"✅ Tunnel established: {url}")
    if print_settings:
        _print_client_settings(url, api_key)
    return {"url": url, "process": proc}


def stop_process(process) -> None:
    """Stop a process returned by serve_unsloth() or expose_with_cloudflared()."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except Exception:
            process.kill()
    print(f"Stopped process {process.pid}.")


# =========================================================================== environment

def check_environment(verbose: bool = True) -> dict:
    """Describe the Linux environment the code runs in and warn about common problems.
    Returns {"arch", "host", "gpus", "memory_gb", "disk_free_gb", "warnings"}."""
    import importlib.util
    import platform
    import shutil
    import subprocess
    from pathlib import Path

    arch = platform.machine().lower()
    kernel = platform.release().lower()
    if "microsoft" in kernel or "wsl" in kernel:
        host = "Windows (Docker Desktop / WSL2)"
    elif "linuxkit" in kernel:
        host = "Docker Desktop (macOS or Windows)"
    elif importlib.util.find_spec("google.colab") is not None:
        host = "Google Colab"
    else:
        host = "Linux"

    gpus = []
    if smi := shutil.which("nvidia-smi"):
        r = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True)
        gpus = [line.strip() for line in r.stdout.splitlines() if line.strip()] if r.returncode == 0 else []

    def memory_gb():
        # A container memory limit (cgroup v2) wins over the VM/host total.
        try:
            limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            if limit != "max":
                return int(limit) / 1024**3
        except OSError:
            pass
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024**2
        return 0.0

    mem = round(memory_gb(), 1)
    disk = round(shutil.disk_usage(Path.home()).free / 1024**3, 1)

    warnings = []
    if arch not in ("x86_64", "amd64", "aarch64", "arm64"):
        warnings.append(f"Unsupported CPU architecture {arch}.")
    if not gpus:
        hint = ("Docker on macOS cannot use the Mac GPU" if arch in ("aarch64", "arm64")
                else "start the container with GPU access if the host has an NVIDIA GPU")
        warnings.append(f"No NVIDIA GPU visible - GGUF models run on the CPU; {hint}.")
    if mem < 8:
        warnings.append(f"Only {mem} GB RAM available - raise the limit in Docker Desktop "
                        "(Settings > Resources) or .wslconfig on Windows; 4B models need ~8 GB.")
    if disk < 20:
        warnings.append(f"Only {disk} GB free disk - Unsloth plus models need roughly 20 GB.")

    info = {"arch": arch, "host": host, "gpus": gpus, "memory_gb": mem,
            "disk_free_gb": disk, "warnings": warnings}
    if verbose:
        print(f"🖥️  {host}, {arch}, {mem} GB RAM, {disk} GB free disk, "
              f"GPU: {', '.join(gpus) if gpus else 'none'}")
        for w in warnings:
            print(f"⚠️  {w}")
    return info


# =========================================================================== all-in-one

def run_llms(model: dict, embedding_model: dict | None = None,
             predownload: dict[str, dict] | None = None, port: int = 8900,
             expose: bool = True, switch_model_by_request: bool = True) -> dict:
    """Install, download, serve and expose everything, running independent steps in parallel.

    model:           spec of the chat model that gets loaded, {"repo", "file", "variant"}
    embedding_model: optional spec of the embedding model served next to it
    predownload:     optional {name: spec} of further models, only downloaded so that
                     load_model() can switch to them quickly later
    port:            local port of the Unsloth server
    expose:          open a Cloudflare quick tunnel
    switch_model_by_request:
                     let API requests load another downloaded model by naming it
                     (Unsloth's "Switch model by request"; on by default)

    Parallel plan:
      1. install unsloth | hf | cloudflared                     (all at once)
      2. downloads start as soon as hf is ready (all at once, while unsloth still installs)
         tunnel starts as soon as cloudflared is ready (it only needs the port number)
      3. server starts once unsloth and all downloads are done, then model switching is set

    Returns a "stack" dict that healthcheck() and shutdown() understand."""
    from concurrent.futures import ThreadPoolExecutor

    environment = check_environment()
    if not environment["gpus"] and not model.get("variant"):
        print(f"⚠️  {model['repo']} is not a GGUF model; without a GPU use a GGUF variant instead.")

    stack = {"environment": environment, "switch_model_by_request": switch_model_by_request,
             "server": None, "tunnel": None, "public_url": None, "api_key": None,
             "model": model, "embedding_model": embedding_model,
             "embedding_model_name": _embedding_env(embedding_model).get("RAG_EMBEDDING_MODEL"),
             "predownload": predownload or {}}

    specs = [model] + ([embedding_model] if embedding_model else []) + list((predownload or {}).values())
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            f_unsloth = pool.submit(install_unsloth)
            f_hf = pool.submit(install_hf)
            f_cloudflared = pool.submit(install_cloudflared) if expose else None

            f_hf.result()
            downloads = [pool.submit(download_model, s["repo"], s.get("file"), True) for s in specs]

            f_tunnel = None
            if expose:
                f_cloudflared.result()
                f_tunnel = pool.submit(expose_with_cloudflared, port, print_settings=False)

            f_unsloth.result()
            for d in downloads:
                d.result()

            stack["server"] = serve_unsloth(model["repo"], model.get("variant"), port,
                                            embedding_model=embedding_model)
            stack["api_key"] = stack["server"]["api_key"]
            try:
                configure_model_switching(stack["server"]["port"], stack["api_key"],
                                          enabled=switch_model_by_request)
            except Exception as exc:
                print(f"⚠️  Could not set 'Switch model by request': {exc}")

            if f_tunnel:
                stack["tunnel"] = f_tunnel.result()
                # The server may have moved to another port if the requested one was taken.
                if stack["server"]["port"] != port:
                    stop_process(stack["tunnel"]["process"])
                    stack["tunnel"] = expose_with_cloudflared(stack["server"]["port"],
                                                              print_settings=False)
                stack["public_url"] = stack["tunnel"]["url"]
    except BaseException:
        print("❌ Setup failed - stopping everything that was started.")
        shutdown(stack)
        raise

    if stack["public_url"]:
        _print_client_settings(stack["public_url"], stack["api_key"])
    return stack


def shutdown(stack: dict) -> None:
    """Stop tunnel and server of a stack returned by run_llms()."""
    for part in ("tunnel", "server"):
        if stack.get(part):
            stop_process(stack[part]["process"])
            stack[part] = None


# =========================================================================== monitoring

def healthcheck(stack: dict, timeout: float = 10, check_embeddings: bool = False,
                verbose: bool = True) -> dict:
    """Check every component of a stack from run_llms() without generating text.

    Checks: server process alive, /api/health, a model resident in memory
    (/api/inference/loaded-models, no disk scan), the server rejecting a wrong API key,
    the "Switch model by request" setting,
    tunnel process alive, and the full public path Cloudflare -> Unsloth.
    `check_embeddings=True` also embeds one short word (tiny load, off by default).

    Returns {"ok": bool, "checks": {name: {"ok", "detail", "ms"}}}."""
    import time

    checks: dict = {}

    def run(name, fn):
        t0 = time.monotonic()
        try:
            ok, detail = True, fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}"
        checks[name] = {"ok": ok, "detail": detail, "ms": round((time.monotonic() - t0) * 1000)}

    def alive(proc):
        if proc.poll() is not None:
            raise RuntimeError(f"exited with code {proc.returncode}")
        return f"pid {proc.pid}"

    def loaded_models(base, key):
        data = _request("GET", f"{base}/api/inference/loaded-models", key, timeout=timeout)
        ids = [m.get("id") for m in data.get("data", [])]
        if not ids:
            raise RuntimeError("server is up, but no model is loaded")
        return ", ".join(map(str, ids))

    def health(base):
        status = _request("GET", f"{base}/api/health", timeout=timeout).get("status")
        if status != "healthy":
            raise RuntimeError(f"status is {status!r}")
        return status

    def rejects_bad_key(base):
        status = _http_status(f"{base}/api/inference/loaded-models",
                              "invalid-key-for-healthcheck", timeout)
        if status != 401:
            raise RuntimeError(f"expected 401 for a wrong key, got {status}")
        return "401 as expected"

    server, tunnel = stack.get("server"), stack.get("tunnel")
    if not server:
        return {"ok": False, "checks": {"server": {"ok": False, "detail": "not started", "ms": 0}}}

    local = f"http://127.0.0.1:{server['port']}"
    run("server process", lambda: alive(server["process"]))
    run("server health", lambda: health(local))
    run("model loaded", lambda: loaded_models(local, server["api_key"]))
    run("auth enforced", lambda: rejects_bad_key(local))

    if "switch_model_by_request" in stack:
        def switching():
            enabled = _request("GET", f"{local}/api/settings/openai-auto-switch",
                               server["api_key"], timeout=timeout).get("enabled")
            if enabled != stack["switch_model_by_request"]:
                raise RuntimeError(f"is {'on' if enabled else 'off'}, expected "
                                   f"{'on' if stack['switch_model_by_request'] else 'off'}")
            return "on" if enabled else "off"
        run("model switching", switching)

    if tunnel:
        run("tunnel process", lambda: alive(tunnel["process"]))
        run("public endpoint", lambda: loaded_models(tunnel["url"], server["api_key"]))

    if check_embeddings and stack.get("embedding_model"):
        def embed():
            payload = {"input": ["ping"], "model": stack.get("embedding_model_name")}
            data = _request("POST", f"{local}/v1/embeddings", server["api_key"], payload, timeout)
            return f"dimension {len(data['data'][0]['embedding'])}"
        run("embeddings", embed)

    result = {"ok": all(c["ok"] for c in checks.values()), "checks": checks}
    if verbose:
        print(f"{'✅ healthy' if result['ok'] else '❌ UNHEALTHY'}  ({time.strftime('%H:%M:%S')})")
        for name, c in checks.items():
            print(f"  {'✅' if c['ok'] else '❌'} {name:<17} {c['ms']:>5} ms  {c['detail']}")
    return result


# =========================================================================== tests

def test_chat(base_url: str, api_key: str, prompt: str = "Say hello in one short sentence.") -> str:
    """One non-streaming chat request. `base_url`: tunnel URL or http://127.0.0.1:<port>."""
    result = _request("POST", f"{base_url.rstrip('/')}/v1/chat/completions", api_key,
                      {"messages": [{"role": "user", "content": prompt}], "stream": False})
    answer = result["choices"][0]["message"]["content"]
    print(answer)
    return answer


def test_embeddings(base_url: str, api_key: str, texts: list[str] | None = None,
                    model: str | None = None) -> list[list[float]]:
    """One embeddings request. nomic-embed-text expects task prefixes such as
    "search_query: " and "search_document: " in front of each text."""
    texts = texts or ["search_query: What is Unsloth?",
                      "search_document: Unsloth fine-tunes and serves open models."]
    payload = {"input": texts} | ({"model": model} if model else {})
    result = _request("POST", f"{base_url.rstrip('/')}/v1/embeddings", api_key, payload)
    vectors = [item["embedding"] for item in result["data"]]
    print(f"✅ {len(vectors)} embeddings, dimension {len(vectors[0])}")
    return vectors
