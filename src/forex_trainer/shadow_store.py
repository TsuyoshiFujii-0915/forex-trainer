"""Single-writer, immutable local event batches for development shadow records."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from .full_period import read_json

SCHEMA = Path(__file__).with_name("shadow_record.schema.json")
VALIDATOR = Draft202012Validator(read_json(SCHEMA), format_checker=FormatChecker())


def now() -> str:
    """Return the current UTC wall-clock timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def encode(value: Any) -> bytes:
    """Encode finite canonical JSON without a trailing newline."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(content: bytes) -> str:
    """Hash exact stored bytes."""
    return hashlib.sha256(content).hexdigest()


def sync_directory(path: Path) -> None:
    """Durably persist directory entries on the supported POSIX filesystem."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(path: Path, content: bytes, staging: Path) -> None:
    """Atomically publish durable bytes without replacing an existing path.

    Args:
        path: New immutable destination on the staging filesystem.
        content: Complete serialized artifact or event batch.
        staging: Directory retaining interrupted writes for explicit recovery.
    """
    temporary = staging / f"{uuid.uuid4().hex}.pending"
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    sync_directory(path.parent)
    temporary.unlink()
    sync_directory(staging)


def artifact_path(root: Path, reference: dict[str, str]) -> Path:
    """Resolve and verify a local artifact or an explicitly pinned external file."""
    if set(reference) != {"path", "sha256"}:
        raise ValueError(f"Malformed artifact reference: {reference}")
    path = Path(reference["path"])
    if not path.is_absolute():
        path = root / path
    try:
        actual = digest(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"Cannot read artifact {path}: {exc}") from exc
    if actual != reference["sha256"]:
        raise ValueError(
            f"Artifact hash mismatch: {path}; expected sha256={reference['sha256']}; actual={actual}"
        )
    return path


def read_events(root: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Validate schema, sequence, unique IDs and the exact-byte hash chain."""
    manifest_hash = digest((root / "manifest.json").read_bytes())
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    previous: str | None = None
    for path in sorted((root / "events").iterdir()):
        if path.name != f"{len(rows):012d}.jsonl":
            raise ValueError(f"Unexpected batch name or sequence gap: {path}")
        content = path.read_bytes()
        if not content or not content.endswith(b"\n"):
            raise ValueError(f"Torn committed batch; never truncate: {path}")
        for line in content.splitlines():
            try:
                row = json.loads(line)
                encode(row)
                VALIDATOR.validate(row)
            except (ValueError, ValidationError) as exc:
                raise ValueError(f"Invalid event at {path}:{len(rows)}: {exc}") from exc
            if row["sequence"] != len(rows) or row["previous_event_sha256"] != previous:
                raise ValueError(f"Hash chain or sequence mismatch: {path}:{len(rows)}")
            if row["protocol_sha256"] != manifest_hash:
                raise ValueError(f"Manifest hash mismatch at {path}")
            if row["event_id"] in ids:
                raise ValueError(f"duplicate event ID: {row['event_id']}")
            ids.add(row["event_id"])
            previous = digest(line)
            rows.append(row)
    return rows, previous


class EventLog:
    """Exclusive append API; committed event and artifact paths are immutable."""

    def __init__(self, root: Path) -> None:
        """Bind an existing initialized store without taking a lock yet."""
        self.root = root
        self.descriptor: int | None = None
        self.events: list[dict[str, Any]] = []
        self.head: str | None = None

    def __enter__(self) -> EventLog:
        """Acquire the one writer lock and verify every committed byte."""
        self.descriptor = os.open(
            self.root / "writer.lock", os.O_RDWR | os.O_CREAT, 0o600
        )
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.descriptor)
            self.descriptor = None
            raise ValueError(f"Another writer holds the lock: {self.root}") from exc
        try:
            self.events, self.head = read_events(self.root)
            for path in (self.root / "events").iterdir():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            sync_directory(self.root / "events")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the writer lock even after an interrupted operation."""
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None

    def artifact(self, content: bytes) -> dict[str, str]:
        """Store original bytes by content hash; reuse only byte-identical content."""
        if self.descriptor is None:
            raise ValueError("Artifact writes require the writer lock")
        relative = f"artifacts/{digest(content)}.json"
        path = self.root / relative
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Existing artifact hash mismatch: {path}")
        else:
            publish(path, content, self.root / "staging")
        return {"path": relative, "sha256": digest(content)}

    def append(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Commit an entire event batch after validating its IDs and row schemas."""
        if self.descriptor is None or not records:
            raise ValueError("Nonempty append requires the writer lock")
        ids = {row["event_id"] for row in self.events}
        previous = self.head
        lines, complete = [], []
        for record in records:
            if record["event_id"] in ids:
                raise ValueError(f"duplicate event ID: {record['event_id']}")
            ids.add(record["event_id"])
            row = {
                **record,
                "schema_version": 2,
                "sequence": len(self.events) + len(lines),
                "previous_event_sha256": previous,
                "protocol_sha256": digest((self.root / "manifest.json").read_bytes()),
            }
            VALIDATOR.validate(row)
            line = encode(row)
            previous = digest(line)
            lines.append(line + b"\n")
            complete.append(row)
        publish(
            self.root / "events" / f"{len(self.events):012d}.jsonl",
            b"".join(lines),
            self.root / "staging",
        )
        self.events.extend(complete)
        self.head = previous
        return complete
