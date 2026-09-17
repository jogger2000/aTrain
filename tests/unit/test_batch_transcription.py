import asyncio
from pathlib import Path

from aTrain.utils import transcription


def test_batch_processes_every_selected_path(monkeypatch, tmp_path):
    processed: list[Path] = []
    notices: list[str] = []

    async def fake_run_pipeline(payload, show_finished=True):
        processed.append(payload.path)
        assert payload.output_dir == payload.path.parent
        assert show_finished is False
        return True

    monkeypatch.setattr(transcription, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(transcription.ui, "notify", lambda message, **_: notices.append(message))
    paths = [tmp_path / "first.mp3", tmp_path / "second.mp3"]

    asyncio.run(transcription.start_transcriptions_from_paths(paths))

    assert processed == paths
    assert notices == ["Finished transcribing 2 of 2 files"]
