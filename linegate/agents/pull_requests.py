"""Local pull requests: a branch built with git plumbing plus a JSON record. No merge capability.

Opening a pull request writes blobs, a tree, and a commit on a new branch
from HEAD through a temporary index, so the working tree and the current
branch are never touched. Any merge attempt is appended to the refusal audit
log and raised as a refusal.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from linegate.agents.runtime import ToolRefused
from linegate.dataio import DATA_DIR, ROOT

PR_DIR = DATA_DIR / "pull_requests"
REFUSALS_PATH = ROOT / "traces" / "refusals.jsonl"
IDENTITY = {"GIT_AUTHOR_NAME": "linegate policy watcher", "GIT_AUTHOR_EMAIL": "policy-watcher@linegate.local",
            "GIT_COMMITTER_NAME": "linegate policy watcher", "GIT_COMMITTER_EMAIL": "policy-watcher@linegate.local"}


class LocalPullRequests:
    def __init__(self, repo: Path = ROOT, store_dir: Path = PR_DIR, refusals_path: Path = REFUSALS_PATH):
        self.repo, self.store_dir, self.refusals_path = repo, store_dir, refusals_path

    def git(self, *args: str, stdin: str | None = None, env: dict | None = None) -> str:
        result = subprocess.run(["git", *args], cwd=self.repo, input=stdin, capture_output=True, text=True,
                                env=os.environ | IDENTITY | (env or {}), check=True)
        return result.stdout.strip()

    def build_commit(self, title: str, files: dict[str, str]) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            index = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
            self.git("read-tree", "HEAD", env=index)
            for path, content in files.items():
                blob = self.git("hash-object", "-w", "--stdin", stdin=content)
                self.git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=index)
            tree = self.git("write-tree", env=index)
        return self.git("commit-tree", tree, "-p", "HEAD", "-m", title)

    def open(self, branch: str, title: str, body: str, files: dict[str, str]) -> str:
        if self.git("branch", "--list", branch):
            raise ToolRefused(f"branch {branch} already exists")
        commit = self.build_commit(title, files)
        self.git("update-ref", f"refs/heads/{branch}", commit)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        number = len(list(self.store_dir.glob("*.json"))) + 1
        url = f"local://linegate/pull/{number}"
        record = {"number": number, "url": url, "branch": branch, "base": self.git("rev-parse", "HEAD"), "head": commit,
                  "title": title, "body": body, "files": sorted(files), "state": "open",
                  "opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (self.store_dir / f"{number}.json").write_text(json.dumps(record, indent=2) + "\n")
        return url

    def get(self, url: str) -> dict:
        path = self.store_dir / f"{url.rsplit('/', 1)[-1]}.json"
        if not path.exists():
            raise ToolRefused(f"no pull request at {url}")
        return json.loads(path.read_text())

    def refuse_merge(self, url: str, actor: str) -> None:
        self.refusals_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "actor": actor, "action": "merge",
                  "target": url, "reason": "agents cannot merge; a human reviews and merges policy pull requests"}
        with self.refusals_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        raise ToolRefused(record["reason"])
