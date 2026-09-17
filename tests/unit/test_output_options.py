from pathlib import Path

import pytest

# The lean CI unit-test environment deliberately omits the numerical runtime.
pytest.importorskip("numpy")

from aTrain_core import outputs
from aTrain_core.settings import ComputeType, Device, Settings


def settings(**overrides) -> Settings:
    values = {
        "file": Path("interview.mp3"),
        "file_id": "run-1",
        "file_name": "Interview May 2026.mp3",
        "model": "tiny",
        "language": "en",
        "speaker_detection": False,
        "speaker_count": None,
        "device": Device.CPU,
        "compute_type": ComputeType.INT8,
        "timestamp": "2026-05-04 13-14-15",
        "temperature": None,
    }
    values.update(overrides)
    return Settings(**values)


def test_output_stem_uses_source_prefix_and_date():
    result = outputs.create_output_stem(
        settings(
            use_original_filename=True,
            filename_prefix="LDF-42",
            filename_suffix="reviewed",
            append_date=True,
        )
    )

    assert result == "LDF-42_Interview May 2026_reviewed_2026-05-04"


def test_srt_only_writes_only_named_srt(tmp_path, monkeypatch):
    monkeypatch.setattr(outputs, "TRANSCRIPT_DIR", tmp_path)
    (tmp_path / "run-1").mkdir()
    transcription = {"segments": [{"start": 0.0, "end": 1.0, "text": " Hello"}]}

    outputs.create_output_files(
        transcription,
        settings(srt_only=True, use_original_filename=True, filename_prefix="LDF-42"),
    )

    assert (tmp_path / "run-1" / "LDF-42_Interview May 2026.srt").is_file()
    assert list((tmp_path / "run-1").iterdir()) == [
        tmp_path / "run-1" / "LDF-42_Interview May 2026.srt"
    ]


def test_native_selection_writes_output_next_to_input(tmp_path):
    source_folder = tmp_path / "recordings"
    source_folder.mkdir()
    transcription = {"segments": [{"start": 0.0, "end": 1.0, "text": " Hello"}]}

    outputs.create_output_files(
        transcription,
        settings(
            file=source_folder / "Interview May 2026.mp3",
            output_dir=source_folder,
            srt_only=True,
            use_original_filename=True,
        ),
    )

    assert (source_folder / "Interview May 2026.srt").is_file()
