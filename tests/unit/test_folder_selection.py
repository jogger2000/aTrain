import sys
from types import SimpleNamespace

from aTrain.components.settings.file import transcribable_files


def test_folder_selection_skips_generated_subtitles(tmp_path, monkeypatch):
    audio = tmp_path / "interview.mp3"
    subtitle = tmp_path / "interview.srt"
    audio.touch()
    subtitle.touch()

    class Container:
        def __init__(self, path):
            self.streams = [SimpleNamespace(type="audio" if path == audio else "subtitle")]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(open=lambda path: Container(path), FFmpegError=Exception),
    )

    assert transcribable_files(tmp_path) == [audio]
