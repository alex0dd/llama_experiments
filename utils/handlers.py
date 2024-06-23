from typing import List


class MultiFileHandler:
    """
    Context manager for reading from multiple files at once, without closing them
    """

    def __init__(self, file_paths: List[str]):
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        self.file_paths = file_paths
        self.files = {}

    def __enter__(self):
        for path in self.file_paths:
            self.files[path] = open(path, "rb")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for file in self.files.values():
            file.close()

    def read_bytes(self, file_path, begin_pos, n_of_bytes):
        file = self.files[file_path]
        file.seek(begin_pos)
        return file.read(n_of_bytes)
