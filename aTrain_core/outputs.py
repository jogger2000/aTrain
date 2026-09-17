import json
import os
import shutil
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from aTrain_core.globals import (
    LOG_FILENAME,
    METADATA_FILENAME,
    TIMESTAMP_FORMAT,
    TRANSCRIPT_DIR,
)
from aTrain_core.settings import Settings


def create_directory(file_id):
    """Creates a directory for storing transcription files."""
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    file_directory = os.path.join(TRANSCRIPT_DIR, file_id)
    os.makedirs(file_directory, exist_ok=True)


def create_file_id(file_path, timestamp):
    """Creates a unique identifier for a file composed of the file path and timestamp."""
    # Extract filename from file_path
    file_base_name = os.path.basename(file_path)
    # Use rsplit to split from the right at most once

    timestamp = timestamp.replace(" ", "-").replace("-", "")
    timestamp = timestamp[:-2]
    timestamp = timestamp[2:]

    short_base_name = file_base_name[0:7] if len(file_base_name) >= 5 else file_base_name
    file_id = timestamp + "-" + short_base_name
    return file_id


def create_output_files(result, settings: Settings):
    """Creates output files based on the transcription result."""
    file_id = settings.file_id
    speaker_detection = settings.speaker_detection
    filename_stem = create_output_stem(settings)
    output_directory = Path(settings.output_dir or Path(TRANSCRIPT_DIR) / file_id)
    output_directory.mkdir(parents=True, exist_ok=True)
    if settings.srt_only:
        create_srt_file(result, output_directory, filename_stem)
        return
    create_json_file(result, output_directory, filename_stem)
    create_txt_file(
        result,
        output_directory,
        speaker_detection,
        maxqda=False,
        timestamps=False,
        brackets=True,
        filename_stem=filename_stem,
    )
    create_txt_file(
        result,
        output_directory,
        speaker_detection,
        maxqda=False,
        timestamps=True,
        brackets=True,
        filename_stem=filename_stem,
    )
    create_txt_file(
        result,
        output_directory,
        speaker_detection,
        maxqda=False,
        timestamps=True,
        brackets=False,
        filename_stem=filename_stem,
    )  # NEW: NVivo output format
    create_txt_file(
        result,
        output_directory,
        speaker_detection,
        maxqda=True,
        timestamps=True,
        brackets=True,
        filename_stem=filename_stem,
    )
    create_srt_file(result, output_directory, filename_stem)


def create_output_stem(settings: Settings) -> str:
    """Build a readable export name without changing the archive directory."""
    stem = os.path.splitext(os.path.basename(settings.file_name))[0]
    base_name = stem if settings.use_original_filename else "transcription"
    parts = [settings.filename_prefix.strip(), base_name]
    if settings.filename_suffix.strip():
        parts.append(settings.filename_suffix.strip())
    if settings.append_date:
        date = datetime.strptime(settings.timestamp, TIMESTAMP_FORMAT).strftime("%Y-%m-%d")
        parts.append(date)
    # A prefix is user supplied; prevent it from creating subdirectories or an
    # invalid Windows filename. The source filename itself already came from a
    # valid file selected by the operating system.
    return "_".join(part.replace("/", "-").replace("\\", "-") for part in parts if part)


def create_json_file(result, output_directory: Path, filename_stem):
    """Creates a JSON file for the transcription result."""
    output_file_text = output_directory / f"{filename_stem}.json"
    with open(output_file_text, "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False)


def create_txt_file(
    result,
    output_directory: Path,
    speaker_detection,
    timestamps,
    maxqda,
    brackets=True,
    filename_stem="transcription",
):
    """Creates a TXT file for the transcription result."""
    segments = result["segments"]
    match maxqda, timestamps, brackets:
        case True, _, _:
            filename = f"{filename_stem}_maxqda.txt"
        case False, True, False:
            filename = f"{filename_stem}_nvivo.txt"  # NVivo format: timestamps without brackets
        case False, True, True:
            filename = f"{filename_stem}_timestamps.txt"
        case False, False, _:
            filename = f"{filename_stem}.txt"
    file_path = output_directory / filename
    with open(file_path, "w", encoding="utf-8") as file:
        headline = (
            f"Transcription for {file_id}"
            + ("" if maxqda and speaker_detection else "\n")
            + ("" if speaker_detection else "\n")
        )
        file.write(headline)
        current_speaker = None
        for segment in segments:
            speaker = segment["speaker"] if "speaker" in segment else "Speaker undefined"
            if speaker != current_speaker and speaker_detection:
                file.write(("\n\n" if maxqda else "\n") + speaker + "\n")
                current_speaker = speaker
            text = str(segment["text"]).lstrip()
            if timestamps:
                start_time = time.strftime("[%H:%M:%S]", time.gmtime(segment["start"]))
                text = f"{start_time} - {text}"
            file.write(text + (" " if maxqda else "\n"))


def create_srt_file(result, output_directory: Path, filename_stem="transcription"):
    """Creates a SRT file for the transcription result."""

    segments = result["segments"]
    file_path = output_directory / f"{filename_stem}.srt"
    with open(file_path, "w", encoding="utf-8") as srt_file:
        for index, segment in enumerate(segments, 1):
            srt_file.write(f"{index}\n")
            start_time = segment["start"]
            end_time = segment["end"]
            start_time_format = (
                time.strftime("%H:%M:%S", time.gmtime(start_time))
                + f",{round((start_time - int(start_time)) * 1000):03}"
            )
            end_time_format = (
                time.strftime("%H:%M:%S", time.gmtime(end_time))
                + f",{round((end_time - int(end_time)) * 1000):03}"
            )
            srt_file.write(f"{start_time_format} --> {end_time_format}\n")
            srt_file.write(f"{str(segment['text']).lstrip()}\n\n")


def transform_speakers_results(diarization_segments):
    """Transforms diarization segments to speaker results."""

    diarize_df = pd.DataFrame(diarization_segments.itertracks(yield_label=True))
    diarize_df["start"] = diarize_df[0].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df[0].apply(lambda x: x.end)
    diarize_df.rename(columns={2: "speaker"}, inplace=True)
    return diarize_df


def named_tuple_to_dict(obj):
    """Converts named tuple to dictionary."""
    if isinstance(obj, dict):
        return {key: named_tuple_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [named_tuple_to_dict(value) for value in obj]
    if is_dataclass(obj):
        return {key: named_tuple_to_dict(value) for key, value in asdict(obj).items()}
    if isnamedtupleinstance(obj) or (hasattr(obj, "_asdict") and callable(obj._asdict)):
        return {key: named_tuple_to_dict(value) for key, value in obj._asdict().items()}
    if hasattr(obj, "__dict__"):
        return {key: named_tuple_to_dict(value) for key, value in vars(obj).items()}
    if isinstance(obj, tuple):
        return tuple(named_tuple_to_dict(value) for value in obj)
    return obj


def isnamedtupleinstance(x):
    """Checks if the object is an instance of namedtuple."""
    _type = type(x)
    bases = _type.__bases__
    if len(bases) != 1 or bases[0] is not tuple:
        return False
    fields = getattr(_type, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(type(i) is str for i in fields)


def create_metadata(settings: Settings, audio_duration: int):
    """Creates metadata file for the transcription."""

    metadata_file_path = os.path.join(TRANSCRIPT_DIR, settings.file_id, METADATA_FILENAME)
    metadata = {
        "file_id": settings.file_id,
        "filename": settings.file_name,
        "audio_duration": audio_duration,
        "model": settings.model,
        "language": settings.language,
        "speaker_detection": settings.speaker_detection,
        "num_speakers": settings.speaker_count,
        "device": settings.device.value,
        "compute_type": settings.compute_type.value,
        "timestamp": settings.timestamp,
        "srt_only": settings.srt_only,
        "use_original_filename": settings.use_original_filename,
        "filename_prefix": settings.filename_prefix,
        "filename_suffix": settings.filename_suffix,
        "append_date": settings.append_date,
        "output_dir": str(settings.output_dir) if settings.output_dir else None,
    }
    with open(metadata_file_path, "w", encoding="utf-8") as metadata_file:
        yaml.dump(metadata, metadata_file)
    write_logfile("Metadata created", settings.file_id)


def write_logfile(message, file_id):
    """Writes a log message to the log file."""

    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    log_file_path = os.path.join(TRANSCRIPT_DIR, file_id, LOG_FILENAME)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] ------ {message}\n")


def add_processing_time_to_metadata(file_id):
    """Adds processing time information to metadata."""

    metadata_file_path = os.path.join(TRANSCRIPT_DIR, file_id, METADATA_FILENAME)
    with open(metadata_file_path, encoding="utf-8") as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    timestamp = metadata["timestamp"]
    start_time = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    stop_time = timestamp = datetime.now()
    processing_time = stop_time - start_time
    metadata["processing_time"] = int(processing_time.total_seconds())
    with open(metadata_file_path, "w", encoding="utf-8") as metadata_file:
        yaml.dump(metadata, metadata_file)


def delete_transcription(file_id):
    """Deletes the transcription files."""

    file_id = "" if file_id == "all" else file_id
    directory_name = os.path.join(TRANSCRIPT_DIR, file_id)
    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
    if not os.path.exists(TRANSCRIPT_DIR):
        os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    """Assigns speakers to transcribed words.
    Function is taken from whisperx -> see https://github.com/m-bain/whisperX.git
    """
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        diarize_df["intersection"] = np.minimum(diarize_df["end"], seg["end"]) - np.maximum(
            diarize_df["start"], seg["start"]
        )
        diarize_df["union"] = np.maximum(diarize_df["end"], seg["end"]) - np.minimum(
            diarize_df["start"], seg["start"]
        )
        dia_tmp = diarize_df[diarize_df["intersection"] > 0] if not fill_nearest else diarize_df
        if len(dia_tmp) > 0:
            speaker = (
                dia_tmp.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
            seg["speaker"] = speaker
        if "words" in seg:
            for word in seg["words"]:
                if "start" in word:
                    diarize_df["intersection"] = np.minimum(
                        diarize_df["end"], word["end"]
                    ) - np.maximum(diarize_df["start"], word["start"])
                    diarize_df["union"] = np.maximum(diarize_df["end"], word["end"]) - np.minimum(
                        diarize_df["start"], word["start"]
                    )
                    dia_tmp = (
                        diarize_df[diarize_df["intersection"] > 0]
                        if not fill_nearest
                        else diarize_df
                    )
                    if len(dia_tmp) > 0:
                        speaker = (
                            dia_tmp.groupby("speaker")["intersection"]
                            .sum()
                            .sort_values(ascending=False)
                            .index[0]
                        )
                        word["speaker"] = speaker
    return transcript_result
