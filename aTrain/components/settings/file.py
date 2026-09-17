import os
from dataclasses import dataclass, field
from pathlib import Path

from aTrain_core.globals import FLATPAK, LINUX
from aTrain_core.settings import load_formats
from nicegui import ui


class CustomUpload(ui.upload):
    def __init__(self, selection_kind: str = "file", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_kind = selection_kind
        self.on("added", self.set_added)
        self.set_select()

    def pick_files(self):
        self.reset()
        self.set_select()
        self.run_method("pickFiles")

    def upload(self):
        self.run_method("upload")

    def set_added(self):
        self.file_text = "File added" if self.selection_kind == "file" else "Folder selected"
        self.file_icon = "file_present" if self.selection_kind == "file" else "folder"

    def set_select(self):
        self.file_text = "Select File" if self.selection_kind == "file" else "Select Folder"
        self.file_icon = "attach_file" if self.selection_kind == "file" else "folder_open"


@dataclass
class FileSelection:
    file_uploader: CustomUpload | None = None
    folder_uploader: CustomUpload | None = None
    selected_paths: list[Path] = field(default_factory=list)
    selection_kind: str | None = None

    def upload(self) -> None:
        if self.selection_kind == "file" and self.file_uploader:
            self.file_uploader.upload()
        elif self.selection_kind == "folder" and self.folder_uploader:
            self.folder_uploader.upload()
        else:
            ui.notify("Please select a file or folder first", color="negative")


def transcribable_files(folder: Path) -> list[Path]:
    """Return only files which actually contain an audio stream.

    The upstream format list also contains subtitle formats such as `.srt`.
    Checking the container prevents a previously generated transcript in the
    same input/output folder from being submitted as a new transcription.
    """
    import av

    allowed_extensions = {extension.lower() for extension in load_formats()}
    files: list[Path] = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed_extensions:
            continue
        try:
            with av.open(path) as container:
                if any(stream.type == "audio" for stream in container.streams):
                    files.append(path)
        except av.FFmpegError:
            continue
    return sorted(files)


def input_file() -> FileSelection:
    allowed_files = "".join(x for x in str(load_formats()) if x not in "[]'")
    selection = FileSelection()

    with ui.column().classes("gap-2") as file_column:
        ui.label("Select Audio").classes("font-bold text-dark text-md")
        ui.separator()
        with ui.row().classes("w-full gap-2"):
            file_button = ui.button().props("color=gray-100 text-color=dark align=left")
            file_button.props("unelevated no-caps :ripple=false").classes("flex-1")
            folder_button = ui.button("Select Folder").props(
                "color=gray-100 text-color=dark align=left"
            )
            folder_button.props("unelevated no-caps :ripple=false").classes("flex-1")

    if not (FLATPAK or LINUX):
        file_uploader = CustomUpload("file").classes("hidden")
        file_uploader.props(f"accept='{allowed_files}'")
        selection.file_uploader = file_uploader
        file_button.bind_text(file_uploader, "file_text").bind_icon(file_uploader, "file_icon")

        def pick_file() -> None:
            selection.selection_kind = "file"
            file_uploader.pick_files()

        def pick_folder() -> None:
            selection.selection_kind = "folder"
            try:
                from tkinter import Tk, filedialog

                root = Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                folder = filedialog.askdirectory(title="Select Folder")
                root.destroy()
            except Exception as exc:
                ui.notify(f"Could not open folder picker: {exc}", color="negative")
                return
            if not folder:
                return
            selection.selected_paths = transcribable_files(Path(folder))
            folder_button.text = f"{len(selection.selected_paths)} files selected"
            if not selection.selected_paths:
                ui.notify("No supported audio or video files found in this folder", color="negative")

        file_button.on_click(pick_file)
        folder_button.on_click(pick_folder)
        return selection

    with file_column:
        file_label = ui.label("No file or folder selected").classes("text-sm text-gray-500")

    def pick_file_native(directory: bool = False) -> str | None:
        try:
            import gi  # type: ignore

            gi.require_version("Gio", "2.0")
            gi.require_version("GLib", "2.0")
            from gi.repository import Gio, GLib  # type: ignore

            bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            proxy = Gio.DBusProxy.new_sync(
                bus,
                Gio.DBusProxyFlags.NONE,
                None,
                "org.freedesktop.portal.Desktop",
                "/org/freedesktop/portal/desktop",
                "org.freedesktop.portal.FileChooser",
                None,
            )

            token = f"atrain{os.getpid()}"
            options = {
                "handle_token": GLib.Variant("s", token),
                "multiple": GLib.Variant("b", False),
                "directory": GLib.Variant("b", directory),
            }

            result = proxy.call_sync(
                "OpenFile",
                GLib.Variant(
                    "(ssa{sv})", ("", "Select Folder" if directory else "Select File", options)
                ),
                Gio.DBusCallFlags.NONE,
                -1,
                None,
            )
            handle = result.unpack()[0]

            request = Gio.DBusProxy.new_sync(
                bus,
                Gio.DBusProxyFlags.NONE,
                None,
                "org.freedesktop.portal.Desktop",
                handle,
                "org.freedesktop.portal.Request",
                None,
            )

            filename: str | None = None
            loop = GLib.MainLoop()

            def on_response(_proxy, _sender, _signal, params):
                nonlocal filename
                response, results = params.unpack()
                if response == 0:
                    uris = results.get("uris")
                    if uris:
                        uri = uris[0]
                        filename = Gio.File.new_for_uri(uri).get_path()
                loop.quit()

            request.connect("g-signal", on_response)
            loop.run()
            return filename
        except Exception as exc:
            print(f"Flatpak portal file dialog failed: {exc}")
            return None

    def on_pick_file():
        path = pick_file_native()
        if not path:
            return
        selection.selected_paths = [Path(path)]
        file_label.text = Path(path).name

    def on_pick_folder():
        folder = pick_file_native(directory=True)
        if not folder:
            return
        selection.selected_paths = transcribable_files(Path(folder))
        file_label.text = (
            f"{len(selection.selected_paths)} supported files in {Path(folder).name}"
        )
        if not selection.selected_paths:
            ui.notify("No supported audio or video files found in this folder", color="negative")

    file_button.text = "Select File"
    folder_button.text = "Select Folder"
    file_button.on_click(on_pick_file)
    folder_button.on_click(on_pick_folder)
    return selection
