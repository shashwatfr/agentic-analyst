"""Google Drive sink, over MCP.

Tools are discovered at runtime and matched by name rather than hardcoded. MCP
servers for Drive don't agree on naming - create_file, upload_file, gdrive_create_file
all exist in the wild - so matching by pattern means swapping servers is a .env change
instead of a code change.

Worth knowing: the official @modelcontextprotocol/server-gdrive is archived and
read-only, with no create or upload tool at all, so it cannot commit a report. The
README points at mcp-google-drive, which can.
"""

from __future__ import annotations

import asyncio
import base64
import re
from pathlib import Path
from typing import Any

from ..config import DriveMCPConfig
from .base import CommitResult, ReportSink, folder_name

# Ordered by preference: an exact create_file beats a generic upload.
TOOL_PATTERNS = {
    "create_file": [r"^create_file$", r"^gdrive_create_file$", r"create.*file", r"upload.*file", r"^upload$"],
    "create_folder": [r"^create_folder$", r"create.*folder", r"^mkdir$"],
}

CONNECT_TIMEOUT = 60


def _run_async(coro: Any) -> Any:
    """Run a coroutine whether or not the caller already has an event loop.

    The CLI has no loop, so asyncio.run() is fine. Streamlit reruns the script on a
    worker thread that may already have one, and asyncio.run() raises outright in
    that case - so fall back to a dedicated thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _match_tool(tools: list[Any], patterns: list[str]) -> Any | None:
    names = {t.name: t for t in tools}
    for pattern in patterns:
        for name, tool in names.items():
            if re.search(pattern, name, re.IGNORECASE):
                return tool
    return None


def _arg_names(tool: Any) -> set[str]:
    schema = getattr(tool, "args_schema", None) or {}
    if isinstance(schema, dict):
        return set(schema.get("properties", {}))
    try:
        return set(schema.model_json_schema().get("properties", {}))
    except Exception:  # noqa: BLE001
        return set()


def _build_args(tool: Any, *, name: str, content: bytes, mime: str, parent: str, is_text: bool) -> dict[str, Any]:
    """Fit our payload to whatever argument names this server actually uses."""
    accepted = _arg_names(tool)
    args: dict[str, Any] = {}

    for key in ("name", "fileName", "filename", "title", "file_name"):
        if key in accepted:
            args[key] = name
            break
    else:
        args["name"] = name

    # Text goes as text where the server allows it; PNGs always need base64.
    payload = content.decode("utf-8") if is_text else base64.b64encode(content).decode("ascii")
    for key in ("content", "data", "fileContent", "body", "text"):
        if key in accepted:
            args[key] = payload
            break
    else:
        args["content"] = payload

    if not is_text:
        for key in ("encoding", "contentEncoding"):
            if key in accepted:
                args[key] = "base64"
                break

    for key in ("mimeType", "mime_type", "mimetype"):
        if key in accepted:
            args[key] = mime
            break

    if parent:
        for key in ("parentId", "parent_id", "folderId", "folder_id", "parents"):
            if key in accepted:
                args[key] = [parent] if key == "parents" else parent
                break

    return args


async def _commit_async(
    config: DriveMCPConfig, report_md: str, chart_paths: list[str], meta: dict[str, Any]
) -> CommitResult:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "gdrive": {
                "transport": "stdio",
                "command": config.command,
                "args": config.args,
                "env": config.env or None,
            }
        }
    )

    tools = await client.get_tools(server_name="gdrive")
    if not tools:
        return CommitResult(ok=False, destination="google drive", detail="server exposed no tools")

    creator = _match_tool(tools, TOOL_PATTERNS["create_file"])
    if creator is None:
        available = ", ".join(sorted(t.name for t in tools))
        return CommitResult(
            ok=False,
            destination="google drive",
            detail=(
                "no file-creation tool found on the MCP server "
                f"(available: {available}). The official server-gdrive is read-only; "
                "try mcp-google-drive."
            ),
        )

    parent = config.folder_id
    folder = folder_name(meta)
    folder_tool = _match_tool(tools, TOOL_PATTERNS["create_folder"])
    if folder_tool is not None:
        try:
            created = await folder_tool.ainvoke({"name": folder, **({"parentId": parent} if parent else {})})
            found = re.search(r"[-\w]{25,}", str(created))
            if found:
                parent = found.group(0)
        except Exception:  # noqa: BLE001 - fall back to writing into the parent folder
            folder = ""

    uploaded, failures = [], []

    files: list[tuple[str, bytes, str, bool]] = [
        # Prefix when we couldn't make a folder, so files stay grouped by run.
        (f"{folder + '_' if not folder_tool else ''}report.md", report_md.encode("utf-8"), "text/markdown", True)
    ]
    for chart in chart_paths:
        path = Path(chart)
        if path.exists():
            files.append((path.name, path.read_bytes(), "image/png", False))

    for name, content, mime, is_text in files:
        try:
            await creator.ainvoke(
                _build_args(creator, name=name, content=content, mime=mime, parent=parent, is_text=is_text)
            )
            uploaded.append(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")

    if not uploaded:
        return CommitResult(
            ok=False, destination="google drive", detail="; ".join(failures) or "nothing uploaded"
        )

    link = f"https://drive.google.com/drive/folders/{parent}" if parent else ""
    detail = f"Uploaded {len(uploaded)} files via '{creator.name}'"
    if failures:
        detail += f" ({len(failures)} failed: {'; '.join(failures)})"

    return CommitResult(ok=True, destination="google drive", files=uploaded, link=link, detail=detail)


class DriveMCPSink(ReportSink):
    def __init__(self, config: DriveMCPConfig, fallback: ReportSink | None = None):
        self.config = config
        self.fallback = fallback

    @property
    def name(self) -> str:
        return "google drive (mcp)"

    def commit(self, report_md: str, chart_paths: list[str], meta: dict[str, Any]) -> CommitResult:
        if not self.config.enabled:
            return self._fall_back(report_md, chart_paths, meta, "no DRIVE_MCP_COMMAND configured")

        try:
            result = _run_async(
                asyncio.wait_for(
                    _commit_async(self.config, report_md, chart_paths, meta), CONNECT_TIMEOUT
                )
            )
        except Exception as exc:  # noqa: BLE001
            return self._fall_back(report_md, chart_paths, meta, f"{type(exc).__name__}: {exc}")

        if not result.ok:
            return self._fall_back(report_md, chart_paths, meta, result.detail)
        return result

    def _fall_back(
        self, report_md: str, chart_paths: list[str], meta: dict[str, Any], reason: str
    ) -> CommitResult:
        """A Drive problem shouldn't lose the report - write it locally and say why."""
        if self.fallback is None:
            return CommitResult(ok=False, destination=self.name, detail=reason)

        result = self.fallback.commit(report_md, chart_paths, meta)
        result.detail = f"Drive unavailable ({reason}). {result.detail}"
        result.destination = f"{self.fallback.name} (fallback)"
        return result
