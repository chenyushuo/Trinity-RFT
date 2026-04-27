"""stdout/stderr Tee 到 logs/{date}/batch_run_{HHMMSS}.log。

用法:
    with BatchStdoutTee() as log_path:
        print("...")  # 同时写终端 + 文件
"""

import os
import sys
from datetime import datetime


class _TeeIO:
    """同时把 write 转发给多个底层流。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            try:
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return self.streams[0].isatty() if self.streams else False


class BatchStdoutTee:
    """context manager: 进入时打开日志文件并替换 sys.stdout/stderr，退出时还原。"""

    def __init__(self, script_dir: str | None = None):
        self._script_dir = script_dir or os.path.dirname(os.path.abspath(sys.argv[0] or __file__))
        self._orig_out = None
        self._orig_err = None
        self._fp = None
        self.log_path: str | None = None

    def __enter__(self) -> str:
        log_base = os.path.join(self._script_dir, "logs")
        date_str = os.environ.get("RESULT_DATE_PREFIX", datetime.now().strftime("%Y%m%d"))
        ts = datetime.now().strftime("%H%M%S")
        day_dir = os.path.join(log_base, date_str)
        os.makedirs(day_dir, exist_ok=True)
        self.log_path = os.path.join(day_dir, f"batch_run_{ts}.log")
        self._fp = open(self.log_path, "w", encoding="utf-8")
        self._orig_out, self._orig_err = sys.stdout, sys.stderr
        sys.stdout = _TeeIO(self._orig_out, self._fp)
        sys.stderr = _TeeIO(self._orig_err, self._fp)
        return self.log_path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._orig_out is not None:
            sys.stdout = self._orig_out
        if self._orig_err is not None:
            sys.stderr = self._orig_err
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        self._orig_out = self._orig_err = None
