"""Controls for choosing which transcript files aTrain writes."""

from nicegui import app, ui


def input_output_options() -> None:
    """Render export options and store them with the other transcription settings."""
    state = app.storage.general
    with ui.column().classes("gap-2"):
        ui.label("Output Files").classes("font-bold text-dark text-md")
        ui.separator()
        srt_only = ui.switch("Create SRT only", value=state.get("srt_only", False))
        srt_only.props("color=dark").mark("switch_srt_only")
        srt_only.bind_value(state, "srt_only")

        original_name = ui.switch(
            "Use original filename", value=state.get("use_original_filename", True)
        )
        original_name.props("color=dark").mark("switch_original_filename")
        original_name.bind_value(state, "use_original_filename")

        prefix = ui.input(
            label="Filename prefix (optional)",
            placeholder="e.g. LDF-123",
            value=state.get("filename_prefix", ""),
        )
        prefix.props("filled bg-color=gray-100 color=dark clearable").classes("w-full")
        prefix.bind_value(state, "filename_prefix").mark("input_filename_prefix")

        suffix = ui.input(
            label="Filename suffix (optional)",
            placeholder="e.g. reviewed",
            value=state.get("filename_suffix", ""),
        )
        suffix.props("filled bg-color=gray-100 color=dark clearable").classes("w-full")
        suffix.bind_value(state, "filename_suffix").mark("input_filename_suffix")

        append_date = ui.switch("Append date", value=state.get("append_date", False))
        append_date.props("color=dark").mark("switch_append_date")
        append_date.bind_value(state, "append_date")
