"""Read single members of a large zip over HTTP without downloading the whole archive.

A zip's directory sits at its end, and each member can be located from it, so
HTTP range requests fetch only the directory and the members actually needed.
Used for the 34 GB PMcardio ECG Image Database, of which only the images of
PTB-XL records the classifier never trained on are wanted.
"""

from __future__ import annotations

import http.client
import io
import time
import urllib.request
import zipfile


class HttpRangeFile(io.RawIOBase):
    """A seekable, read-only file whose bytes come from HTTP range requests."""

    def __init__(self, url: str, timeout: int = 120, block: int = 1 << 20):
        self.url, self.timeout, self.block = url, timeout, block
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            self.size = int(response.headers["Content-Length"])
            self.url = response.url  # follow redirects once
        self.position = 0
        self._cache_start, self._cache = -1, b""

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, min(self.size, base + offset))
        return self.position

    def _fetch(self, start: int, end: int) -> bytes:
        """One byte range, retried until it arrives complete (dropped connections are common)."""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = response.read()
                if len(data) == end - start:
                    return data
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(min(30, 2 ** attempt))
        raise OSError(f"could not read bytes {start}-{end - 1} of {self.url}")

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        end = min(self.size, self.position + size)
        if end <= self.position:
            return b""
        cached_end = self._cache_start + len(self._cache)
        if not (self._cache_start <= self.position and end <= cached_end):
            fetch_end = min(self.size, max(end, self.position + self.block))
            self._cache_start, self._cache = self.position, self._fetch(self.position, fetch_end)
        offset = self.position - self._cache_start
        data = self._cache[offset: offset + (end - self.position)]
        self.position += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))
