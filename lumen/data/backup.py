from pathlib import Path
import sqlite3
import shutil
from datetime import datetime


def backup_database(db_path: Path, backup_dir: Path | None = None) -> Path:
    if backup_dir is None:
        backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"lumen_{timestamp}.db"
    src = str(db_path)
    dst = str(dest)
    # Use SQLite online backup API
    with sqlite3.connect(src) as source:
        with sqlite3.connect(dst) as target:
            source.backup(target)
    # Copy WAL/shm if present
    for ext in ("-wal", "-shm"):
        src_ext = db_path.with_suffix(db_path.suffix + ext)
        if src_ext.exists():
            shutil.copy2(str(src_ext), str(dest.with_suffix(dest.suffix + ext)))
    return dest
