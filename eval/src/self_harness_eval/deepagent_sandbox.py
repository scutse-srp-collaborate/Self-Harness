"""Task-scoped Docker sandbox backend for DeepAgents file/shell tools.

Each SWE-bench case gets its own disposable container.  Only the task workspace
is bind-mounted (read/write) into the container; the harness snapshot, dataset,
repository cache, model credentials and the rest of the host filesystem are not
mounted or passed into it.  The model API key is used only by the host-side
adapter process and is deliberately stripped from every Docker subprocess
environment.

The class implements DeepAgents' ``BaseSandbox`` protocol, so the standard
DeepAgents filesystem/shell tools operate inside the container.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from .openai_compat import MISSING_API_KEY_MESSAGE, MissingApiKeyError  # noqa: F401

TraceCallback = Callable[[dict[str, Any]], None]

class SandboxError(RuntimeError):
    """Raised when the task sandbox cannot be created or used."""


def _docker_env() -> dict[str, str]:
    """Environment for Docker CLI subprocesses: no model credentials."""

    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    for name in (
        "HOME",
        "DOCKER_HOST",
        "DOCKER_CONFIG",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "XDG_RUNTIME_DIR",
    ):
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


def redact_secrets(text: str, *, secrets: Sequence[str] = ()) -> str:
    """Remove known secret values from text before logging/persisting it."""

    result = text
    for secret in secrets:
        if secret:
            result = result.replace(secret, "<redacted>")
    return result


def docker_image_digest(image: str, *, timeout_s: float = 60.0) -> str | None:
    completed = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        timeout_s=timeout_s,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _run(
    argv: Sequence[str],
    *,
    timeout_s: float | None,
    capture_limit: int = 200_000,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in argv],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=timeout_s,
        env=_docker_env(),
    )


def _agent_extra_env() -> dict[str, str]:
    """Read non-secret container env derived from the task image probe."""

    raw = os.environ.get("SELF_HARNESS_AGENT_CONTAINER_ENV", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in parsed.items():
        name = str(key)
        upper = name.upper()
        if any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            continue
        result[name] = str(value)
    return result


@dataclass
class DockerSandboxSettings:
    image: str = "python:3.12"
    network: str = "none"
    memory: str = "8g"
    cpus: str = "4"
    pids_limit: int = 512
    tool_timeout_s: int = 120
    workspace_mount: str = "/testbed"
    inherit_image_digest: str | None = None
    readonly_mounts: tuple[tuple[str, str], ...] = ()
    container_env: Mapping[str, str] = field(default_factory=dict)
    read_only_root: bool = False
    bootstrap_workspace_from_image: bool = False
    image_repo_path: str = "/testbed"
    bootstrap_workspace_mount: str = "/workspace"

    @classmethod
    def from_environment(cls) -> "DockerSandboxSettings":
        raw_timeout = os.environ.get("SELF_HARNESS_TOOL_TIMEOUT_S", "120").strip()
        try:
            tool_timeout = int(float(raw_timeout))
        except ValueError:
            tool_timeout = 120
        if tool_timeout <= 0:
            tool_timeout = 120
        raw_read_only = os.environ.get("SELF_HARNESS_AGENT_READ_ONLY", "").strip().lower()
        return cls(
            image=os.environ.get("SELF_HARNESS_AGENT_IMAGE", "python:3.12").strip()
            or "python:3.12",
            network=os.environ.get("SELF_HARNESS_AGENT_NETWORK", "none").strip() or "none",
            memory=os.environ.get("SELF_HARNESS_AGENT_MEMORY", "8g").strip() or "8g",
            cpus=os.environ.get("SELF_HARNESS_AGENT_CPUS", "4").strip() or "4",
            tool_timeout_s=tool_timeout,
            inherit_image_digest=os.environ.get("SELF_HARNESS_AGENT_IMAGE_DIGEST", "").strip()
            or None,
            container_env=_agent_extra_env(),
            read_only_root=raw_read_only in {"1", "true", "yes", "on"},
            bootstrap_workspace_from_image=os.environ.get(
                "SELF_HARNESS_AGENT_BOOTSTRAP_FROM_IMAGE", ""
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            image_repo_path=os.environ.get("SELF_HARNESS_AGENT_IMAGE_REPO", "/testbed").strip()
            or "/testbed",
            bootstrap_workspace_mount=os.environ.get(
                "SELF_HARNESS_AGENT_BOOTSTRAP_MOUNT", "/workspace"
            ).strip()
            or "/workspace",
        )


class DockerSandbox(BaseSandbox):
    """DeepAgents sandbox backend backed by a disposable Docker container."""

    grep_exclude_dirs = (".git",)

    def __init__(
        self,
        *,
        workspace: Path,
        settings: DockerSandboxSettings | None = None,
        expected_base_commit: str | None = None,
        trace: TraceCallback | None = None,
        secrets: Sequence[str] = (),
        container_name: str | None = None,
    ) -> None:
        self.settings = settings or DockerSandboxSettings.from_environment()
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            raise SandboxError(f"task workspace does not exist: {self.workspace}")
        self.expected_base_commit = expected_base_commit
        self._trace_callback = trace
        self._secrets = tuple(secret for secret in secrets if secret)
        self._closed = False
        self._container_name = container_name or f"self-harness-{uuid.uuid4().hex[:12]}"
        self._image_digest = docker_image_digest(self.settings.image)
        if self._image_digest is None:
            raise SandboxError(
                f"container image {self.settings.image!r} is not available locally; "
                "pre-pull it or set SELF_HARNESS_AGENT_IMAGE to a pre-pulled image"
            )
        expected_digest = self.settings.inherit_image_digest
        if expected_digest and self._image_digest != expected_digest:
            raise SandboxError(
                f"container image {self.settings.image!r} digest {self._image_digest!r} does not "
                f"match the preflight digest {expected_digest!r}"
            )
        self._startup_args: tuple[str, ...] = ()
        self._bootstrap_info: dict[str, Any] | None = None
        self._bootstrap_container_id: str | None = None
        if self.settings.bootstrap_workspace_from_image:
            self._bootstrap_workspace_artifacts()
        self._container_id = self._start_container()
        self._verify_workspace()

    # -- lifecycle -----------------------------------------------------

    def _bootstrap_workspace_artifacts(self) -> None:
        """Copy build artifacts from the image repo into the mounted workspace.

        SWE-bench task images contain compiled extension modules and other
        generated files in ``/testbed``.  The agent workspace must be the host
        checkout at the task base commit, but mounting it over ``/testbed`` would
        hide those artifacts.  A short-lived helper container therefore copies
        only git-ignored/untracked files from the image repo into the workspace
        before the agent container starts.  Tracked source files are never
        overwritten.
        """

        helper_name = f"{self._container_name}-bootstrap"
        env_args: list[str] = []
        for key, value in sorted(self.settings.container_env.items()):
            env_args.extend(["--env", f"{key}={value}"])
        argv = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            helper_name,
            "--workdir",
            self.settings.bootstrap_workspace_mount,
            "--mount",
            f"type=bind,src={self.workspace},dst={self.settings.bootstrap_workspace_mount}",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,exec,size=2g",
            "--tmpfs",
            "/root:rw,size=512m",
            "--tmpfs",
            "/run:rw,size=256m",
            *env_args,
            "--env",
            "HOME=/root",
            self.settings.image,
            "sleep",
            "infinity",
        ]
        try:
            started = _run(argv, timeout_s=120, capture_limit=10_000)
            if started.returncode != 0:
                raise SandboxError(
                    "failed to start image-artifact bootstrap container: "
                    + redact_secrets(started.stderr[-1200:], secrets=self._secrets)
                )
            helper_id = started.stdout.strip()
            if not helper_id:
                raise SandboxError("bootstrap container did not return an id")
            self._bootstrap_container_id = helper_id
            script = f"""
import json, os, shutil, subprocess
src = {self.settings.image_repo_path!r}
dst = {self.settings.bootstrap_workspace_mount!r}
ignored = subprocess.check_output(
    ["git", "-C", src, "ls-files", "-o", "-i", "--exclude-standard", "-z"]
).decode().split(chr(0))
copied = 0
for rel in ignored:
    if not rel or rel.startswith(".git/"):
        continue
    source = os.path.join(src, rel)
    target = os.path.join(dst, rel)
    if not os.path.isfile(source):
        continue
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy2(source, target)
    copied += 1
print(json.dumps({{"copied": copied}}))
"""
            wrapped_script = "python - <<'PY'\n" + script + "\nPY"
            completed = _run(
                ["docker", "exec", helper_id, "bash", "-lc", wrapped_script],
                timeout_s=300,
                capture_limit=200_000,
            )
            if completed.returncode != 0:
                raise SandboxError(
                    "failed to copy task image build artifacts into the workspace: "
                    + redact_secrets(
                        (completed.stdout or "") + (completed.stderr or ""), secrets=self._secrets
                    )[-1500:]
                )
            payload: dict[str, Any] = {}
            for line in reversed((completed.stdout or "").splitlines()):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    payload = candidate
                    break
            self._bootstrap_info = {
                "container_id": helper_id,
                "image": self.settings.image,
                "image_digest": self._image_digest,
                "image_repo_path": self.settings.image_repo_path,
                "workspace_mount": self.settings.bootstrap_workspace_mount,
                "copied_files": payload.get("copied"),
                "container_args": argv,
            }
            self._trace("workspace_bootstrap", self._bootstrap_info)
        finally:
            try:
                _run(["docker", "rm", "-f", helper_name], timeout_s=60)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    def _start_container(self) -> str:
        argv = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            self._container_name,
            "--workdir",
            self.settings.workspace_mount,
            "--mount",
            f"type=bind,src={self.workspace},dst={self.settings.workspace_mount}",
            "--network",
            self.settings.network,
            "--memory",
            self.settings.memory,
            "--cpus",
            self.settings.cpus,
            "--pids-limit",
            str(self.settings.pids_limit),
        ]
        if self.settings.read_only_root:
            argv.extend(
                [
                    "--read-only",
                    "--tmpfs",
                    "/tmp:rw,exec,size=4g",
                    "--tmpfs",
                    "/root:rw,size=1g",
                    "--tmpfs",
                    "/run:rw,size=256m",
                ]
            )
        for source, target in self.settings.readonly_mounts:
            argv.extend(["--mount", f"type=bind,src={source},dst={target},readonly"])
        for key, value in sorted(self.settings.container_env.items()):
            argv.extend(["--env", f"{key}={value}"])
        argv.extend(["--env", "HOME=/root", self.settings.image, "sleep", "infinity"])
        self._startup_args = tuple(argv)
        try:
            completed = _run(argv, timeout_s=120, capture_limit=10_000)
        except subprocess.TimeoutExpired as exc:
            raise SandboxError("timed out starting the task container") from exc
        if completed.returncode != 0:
            raise SandboxError(
                "failed to start the task container: "
                + redact_secrets(completed.stderr[-1500:], secrets=self._secrets)
            )
        container_id = completed.stdout.strip()
        if not container_id:
            raise SandboxError("docker run did not return a container id")
        self._trace(
            "sandbox_created",
            {
                "image": self.settings.image,
                "image_digest": self._image_digest,
                "container_id": container_id,
                "network": self.settings.network,
                "workspace_mount": self.settings.workspace_mount,
                "host_workspace": str(self.workspace),
                "memory": self.settings.memory,
                "cpus": self.settings.cpus,
                "read_only_root": self.settings.read_only_root,
                "readonly_mounts": [list(item) for item in self.settings.readonly_mounts],
                "container_args": list(self._startup_args),
            },
        )
        return container_id

    def _verify_workspace(self) -> None:
        self.execute(
            f"git config --global --add safe.directory {self.settings.workspace_mount}",
            timeout=30,
        )
        result = self.execute("git rev-parse HEAD", timeout=self.settings.tool_timeout_s)
        head = result.output.strip().splitlines()[-1] if result.output.strip() else ""
        if result.exit_code != 0:
            self.close()
            raise SandboxError(
                "task workspace is not a usable git checkout inside the sandbox: "
                + redact_secrets(result.output[-1000:], secrets=self._secrets)
            )
        if self.expected_base_commit and head != self.expected_base_commit:
            self.close()
            raise SandboxError(
                f"inside-container workspace HEAD {head!r} does not match task base commit "
                f"{self.expected_base_commit!r}"
            )
        self._trace("sandbox_workspace_verified", {"head_commit": head})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            _run(["docker", "rm", "-f", self._container_id], timeout_s=60)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
        for path in (self._container_name,):
            self._trace("sandbox_closed", {"container": path})

    def __enter__(self) -> "DockerSandbox":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def id(self) -> str:
        return self._container_id

    @property
    def image_digest(self) -> str | None:
        return self._image_digest

    @property
    def image(self) -> str:
        return self.settings.image

    @property
    def startup_args(self) -> tuple[str, ...]:
        return self._startup_args

    # -- command execution --------------------------------------------

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        configured = max(1, int(self.settings.tool_timeout_s))
        effective = configured if timeout is None else min(max(1, int(timeout)), configured)
        started = time.time()
        argv = [
            "docker",
            "exec",
            "-w",
            self.settings.workspace_mount,
            self._container_id,
            "bash",
            "-lc",
            command,
        ]
        timed_out = False
        exit_code: int | None
        output = ""
        try:
            completed = _run(argv, timeout_s=effective)
            exit_code = completed.returncode
            output = (completed.stdout or "") + (completed.stderr or "")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout = _decode(exc.stdout)
            stderr = _decode(exc.stderr)
            output = (
                (stdout + stderr)
                + f"\n[timeout] command exceeded the {effective}s tool timeout and was "
                "terminated. Break the work into smaller commands or reduce the scope.\n"
            )
        except OSError as exc:
            exit_code = 127
            output = f"ERROR: cannot execute command in sandbox: {exc}\n"
        output = redact_secrets(output, secrets=self._secrets)
        truncated = len(output) > 100_000
        if truncated:
            output = output[:50_000] + "\n...[output truncated]...\n" + output[-50_000:]
        self._trace(
            "tool_command",
            {
                "command": redact_secrets(command[:_COMMAND_PREVIEW], secrets=self._secrets),
                "exit_code": exit_code,
                "timed_out": timed_out,
                "timeout_s": effective,
                "duration_s": round(time.time() - started, 3),
                "output_preview": output[-1500:],
                "truncated": truncated,
            },
        )
        return ExecuteResponse(output=output, exit_code=exit_code, truncated=truncated)

    # -- file transfer ------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                destination = self._validate_container_path(path)
                parent = str(Path(destination).parent)
                mkdir = self.execute(f"mkdir -p {shlex.quote(parent)}", timeout=30)
                if mkdir.exit_code != 0:
                    responses.append(FileUploadResponse(path=path, error="permission_denied"))
                    continue
                with tempfile.NamedTemporaryFile(delete=False) as handle:
                    handle.write(content)
                    temp_path = handle.name
                try:
                    completed = _run(
                        ["docker", "cp", temp_path, f"{self._container_id}:{destination}"],
                        timeout_s=60,
                    )
                finally:
                    Path(temp_path).unlink(missing_ok=True)
                if completed.returncode != 0:
                    error = (
                        "file_not_found"
                        if "no such file" in completed.stderr.lower()
                        else "permission_denied"
                    )
                    responses.append(FileUploadResponse(path=path, error=error))
                else:
                    responses.append(FileUploadResponse(path=path, error=None))
            except (ValueError, OSError) as exc:
                responses.append(FileUploadResponse(path=path, error=str(exc)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                source = self._validate_container_path(path)
            except ValueError as exc:
                responses.append(FileDownloadResponse(path=path, content=None, error=str(exc)))
                continue
            check = self.execute(
                f"test -f {shlex.quote(source)} && echo FILE || "
                f"(test -d {shlex.quote(source)} && echo DIR || echo MISSING)",
                timeout=30,
            )
            kind = check.output.strip().splitlines()[-1] if check.output.strip() else "MISSING"
            if kind == "DIR":
                responses.append(FileDownloadResponse(path=path, content=None, error="is_directory"))
                continue
            if kind != "FILE":
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue
            with tempfile.NamedTemporaryFile(delete=False) as handle:
                temp_path = handle.name
            try:
                completed = _run(
                    ["docker", "cp", f"{self._container_id}:{source}", temp_path],
                    timeout_s=60,
                )
                if completed.returncode != 0:
                    responses.append(
                        FileDownloadResponse(path=path, content=None, error="file_not_found")
                    )
                else:
                    responses.append(
                        FileDownloadResponse(
                            path=path,
                            content=Path(temp_path).read_bytes(),
                            error=None,
                        )
                    )
            finally:
                Path(temp_path).unlink(missing_ok=True)
        return responses

    def _validate_container_path(self, path: str) -> str:
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("invalid_path")
        parts = [part for part in path.split("/") if part and part != "."]
        if any(part == ".." for part in parts):
            raise ValueError("invalid_path")
        return "/" + "/".join(parts)

    # -- tracing ------------------------------------------------------

    def _trace(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self._trace_callback is None:
            return
        try:
            self._trace_callback(
                {
                    "event_type": event_type,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    **dict(payload),
                }
            )
        except Exception:  # noqa: BLE001 - tracing must never break execution
            pass


_COMMAND_PREVIEW = 400


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
