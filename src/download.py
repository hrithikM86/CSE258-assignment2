"""Download only Beauty 2014 reviews and metadata; standard library only."""
import gzip
import hashlib
from pathlib import Path
import urllib.request

BASE = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
FILES = ["reviews_Beauty_5.json.gz", "meta_Beauty.json.gz"]

def main():
    root = Path("data/raw")
    root.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = root / name
        if not target.exists():
            part = target.with_suffix(target.suffix + ".part")
            print("Downloading", BASE + name, flush=True)
            urllib.request.urlretrieve(BASE + name, part)
            with gzip.open(part, "rb") as stream:
                while stream.read(1024 * 1024):
                    pass
            part.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        print(digest, target)

if __name__ == "__main__":
    main()
