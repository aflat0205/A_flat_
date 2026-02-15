import shutil
from pathlib import Path


class LocalStorage:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, source: Path, dest_name: str) -> Path:
        dest = self.base_dir / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return dest

    def load(self, name: str) -> Path:
        path = self.base_dir / name
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    def exists(self, name: str) -> bool:
        return (self.base_dir / name).exists()

    def delete(self, name: str):
        path = self.base_dir / name
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
