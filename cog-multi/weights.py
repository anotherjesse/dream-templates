import os
import subprocess
from collections import deque
import shutil
import hashlib


class WeightsDownloadCache:
    def __init__(self, min_disk_space=5 * (2**30), base_dir = "/src/weights"):  # 5GB
        self.min_disk_space = min_disk_space
        self.lru_paths = deque()
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def _remove_least_recent(self):
        oldest = self.lru_paths.popleft()
        shutil.rmtree(oldest)

    def _has_enough_space(self):
        disk_usage = shutil.disk_usage(os.path.abspath(os.sep))
        print(f"Free disk space: {disk_usage.free}")
        return disk_usage.free > self.min_disk_space

    def ensure(self, url):
        path = self.weights_path(url)

        if path in self.lru_paths:
            self.lru_paths.remove(path)
        else:
            self.download_weights(url, path)

        self.lru_paths.append(path)
        return path

    def weights_path(self, url: str):
        hashed_url = hashlib.sha256(url.encode()).hexdigest()
        short_hash = hashed_url[:16]
        return os.path.join(self.base_dir, short_hash)

    def download_weights(self, url: str, dest: str):
        print("ensuring enough disk space...")        
        while not self._has_enough_space() and len(self.lru_paths) > 0:
            self._remove_least_recent()

        print(f"Downloading weights: {url}")

        output = subprocess.check_output(['./pget', '-x', url, dest])
        print(output)
