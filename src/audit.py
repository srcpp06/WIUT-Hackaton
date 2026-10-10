"""Append-only audit log of officer decisions.

Each JSONL line stores the SHA-256 of the previous line's entry, so editing or deleting any past
entry breaks the chain from that point on (tamper-evident, like a minimal blockchain).
"""
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

GENESIS = '0' * 64


def _digest(entry):
    body = {k: v for k, v in entry.items() if k != 'hash'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class AuditLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def entries(self):
        if not self.path.exists():
            return []
        with self.path.open(encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def append(self, **fields):
        with self.lock:
            entries = self.entries()
            entry = {'seq': len(entries) + 1, 'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                     **fields, 'prev_hash': entries[-1]['hash'] if entries else GENESIS}
            entry['hash'] = _digest(entry)
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            return entry

    def verify(self):
        """(ok, number of entries, seq of the first broken entry or None)."""
        prev = GENESIS
        entries = self.entries()
        for e in entries:
            if e.get('prev_hash') != prev or _digest(e) != e.get('hash'):
                return False, len(entries), e.get('seq')
            prev = e['hash']
        return True, len(entries), None

    def latest_by_alert(self):
        """alert id -> most recent decision entry."""
        return {e['alert_id']: e for e in self.entries()}
