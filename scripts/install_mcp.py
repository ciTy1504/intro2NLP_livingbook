"""Register the Living Book MCP server with Claude Desktop and/or Claude Code.

    python scripts/install_mcp.py              # show what would be written
    python scripts/install_mcp.py --apply      # write it
    python scripts/install_mcp.py --apply --target desktop

Uses stdio, so there is nothing to host and nothing to keep running: the client
launches the server as a subprocess when it needs it and stops it afterwards. That is
also the only arrangement that works here, because the server reads the local SQLite
database, manuscript, key pool and git repository.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def server_entry() -> dict[str, object]:
    return {
        "command": sys.executable,
        "args": ["-m", "livingbook.mcp.server"],
        "cwd": str(ROOT),
        "env": {
            "PYTHONPATH": str(ROOT),
            "PYTHONIOENCODING": "utf-8",
        },
    }


def desktop_config_path() -> Path:
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "Claude" / "claude_desktop_config.json"
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / "Claude"
                / "claude_desktop_config.json")
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def install_desktop(apply: bool) -> None:
    path = desktop_config_path()
    print(f"\nClaude Desktop — {path}")

    config: dict = {}
    if path.exists():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("  ! existing config is not valid JSON; refusing to overwrite it")
            return
    else:
        print("  (no existing config; it will be created)")

    servers = config.setdefault("mcpServers", {})
    if "livingbook" in servers:
        print("  'livingbook' is already registered; it will be updated")
    servers["livingbook"] = server_entry()

    rendered = json.dumps(config, indent=2)
    if not apply:
        print("  would write:")
        print("    " + "\n    ".join(rendered.splitlines()[:18]))
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(".json.bak")
        shutil.copy2(path, backup)
        print(f"  backed up to {backup.name}")
    path.write_text(rendered, encoding="utf-8")
    print("  written — restart Claude Desktop to pick it up")


def install_code(apply: bool) -> None:
    print("\nClaude Code")
    cli = shutil.which("claude")
    entry = server_entry()

    command = [
        "claude", "mcp", "add", "livingbook",
        "--scope", "user",
        "--env", f"PYTHONPATH={ROOT}",
        "--env", "PYTHONIOENCODING=utf-8",
        "--", str(entry["command"]), "-m", "livingbook.mcp.server",
    ]
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in command))

    if not apply:
        return
    if not cli:
        print("  ! the `claude` CLI is not on PATH — run the command above manually")
        return
    try:
        result = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                                text=True, timeout=60)
        print(f"  exit {result.returncode}: "
              f"{(result.stdout or result.stderr).strip()[:300]}")
    except Exception as exc:
        print(f"  ! failed: {type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually write the configuration (default is a dry run)")
    ap.add_argument("--target", choices=["desktop", "code", "both"], default="both")
    args = ap.parse_args()

    print("Living Book MCP server")
    print(f"  python : {sys.executable}")
    print(f"  root   : {ROOT}")
    print(f"  mode   : {'APPLY' if args.apply else 'dry run (pass --apply to write)'}")

    # Fail early rather than registering a server that cannot start.
    probe = subprocess.run(
        [sys.executable, "-c", "import livingbook.mcp.server"],
        cwd=str(ROOT), capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=120)
    if probe.returncode != 0:
        print(f"\n! the server does not import:\n{probe.stderr[-800:]}")
        return 1
    print("  server imports cleanly")

    if args.target in ("desktop", "both"):
        install_desktop(args.apply)
    if args.target in ("code", "both"):
        install_code(args.apply)

    print("\nOnce registered, try asking:")
    print('  "Which parts of the NLP book would speculative decoding research affect?"')
    print('  "What is the Living Book pipeline doing right now?"')
    print('  "Show me the change waiting for my approval and the evidence behind it."')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
