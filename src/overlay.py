import ctypes
import logging
import sys
import tkinter as tk
from ctypes import wintypes
from tkinter.font import Font

from detection import clamp, intersection

log = logging.getLogger(__name__)


def capture_box(box, capture_size, canvas_size):
    cw, ch = capture_size
    w, h = canvas_size
    x1, y1, x2, y2 = box
    return clamp((round(x1 * w / cw), round(y1 * h / ch),
                  round(x2 * w / cw), round(y2 * h / ch)), w, h)


def place_block(anchor, size, limit, occupied):
    x1, y1, x2, y2 = anchor
    width, height = size
    left, top, right, bottom = limit
    if width > right - left or height > bottom - top:
        return None
    for x, y in ((x1, y1), (x1, y2), (x1, y1 - height),
                 (x2, y1), (x1 - width, y1)):
        x = min(max(left, x), right - width)
        y = min(max(top, y), bottom - height)
        box = (x, y, x + width, y + height)
        if all(not intersection(box, other) for other in occupied):
            return box
    return None


def configure_dpi():
    set_dpi = getattr(ctypes.windll.user32, "SetProcessDpiAwarenessContext", None)
    if set_dpi is None:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    set_dpi.argtypes = [ctypes.c_void_p]
    set_dpi.restype = wintypes.BOOL
    if not set_dpi(-4):  # Per-monitor v2
        set_dpi(-3)      # Per-monitor on older Windows 10


class Overlay:
    def __init__(self, config):
        self.config = config
        self.settings = config["overlay"]
        self.region = config["region"]
        self.root = tk.Tk()
        self.root.title("ScreenTranslate Local")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        transparent = "#010203"
        self.root.configure(bg=transparent)
        self.root.attributes("-transparentcolor", transparent)
        self.root.geometry(f"{self.region['width']}x{self.region['height']}+0+0")
        self.canvas = tk.Canvas(self.root, bg=transparent, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.update()
        self.visible = True
        self.regions = []
        self.generation = -1
        self.setup_win32()
        self.root.after(90, self.check_hotkey)

    def setup_win32(self):
        self.user32 = user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        self.hwnd = hwnd = user32.GetAncestor(self.root.winfo_id(), 2)  # GA_ROOT
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        user32.SetWindowLongW.restype = ctypes.c_long
        style = user32.GetWindowLongW(hwnd, -20)
        ctypes.set_last_error(0)
        previous = user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20 | 0x80)
        if not previous and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())

        user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.SetWindowPos.restype = wintypes.BOOL
        if not user32.SetWindowPos(hwnd, -1, self.region["left"], self.region["top"],
                                  self.region["width"], self.region["height"], 0x0010 | 0x0020):
            raise ctypes.WinError(ctypes.get_last_error())
        self.root.update_idletasks()

        user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
        user32.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowDisplayAffinity.restype = wintypes.BOOL
        affinity = wintypes.DWORD()
        self.excluded = (
            not self.settings["hide_during_capture"] and sys.getwindowsversion().build >= 19041
            and bool(user32.SetWindowDisplayAffinity(hwnd, 0x11))
            and bool(user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity)))
            and affinity.value == 0x11
        )
        if not self.excluded:
            user32.SetWindowDisplayAffinity(hwnd, 0)
            log.info("Capture uses temporary overlay hiding (HWND %#x).", hwnd)
        else:
            log.info("Capture exclusion enabled (HWND %#x).", hwnd)

        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        user32.GetAsyncKeyState.restype = ctypes.c_short
        self.flush_compositor = ctypes.windll.dwmapi.DwmFlush

    def grab(self, screen):
        hide = not self.excluded and self.visible and bool(self.regions)
        if hide:
            self.user32.ShowWindow(self.hwnd, 0)
        try:
            if hide and self.flush_compositor() != 0:
                raise RuntimeError("Could not synchronize hidden overlay with the compositor")
            return screen.grab(self.region)
        finally:
            if hide:
                self.user32.ShowWindow(self.hwnd, 8)  # SW_SHOWNA

    def check_hotkey(self):
        if self.user32.GetAsyncKeyState(0x77) & 1:
            self.visible = not self.visible
            self.draw_translations(self.regions)
        self.root.after(90, self.check_hotkey)

    def accept(self, generation, regions):
        if generation <= self.generation:
            return
        self.generation = generation
        if regions != self.regions:
            self.regions = regions
            self.draw_translations(regions)

    def draw_translations(self, regions):
        self.canvas.delete("all")
        self.fonts = []
        if not self.visible:
            return
        occupied = []
        for region in regions:
            box = self.draw_text_block(region, occupied)
            if box:
                occupied.append(box)

    def draw_text_block(self, region, occupied):
        settings = self.settings
        capture_size = (self.region["width"], self.region["height"])
        canvas_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        anchor = capture_box(region["box"], capture_size, canvas_size)
        width, height = canvas_size
        # Tk's rectangle bounding box extends one pixel outside its coordinates.
        full = (1, 1, width - 1, height - 1)
        mode = settings["background_mode"]
        pad = max(2, settings["padding"])
        min_size = settings["min_font_size"]
        max_size = settings["max_font_size"]
        sizes = range(max_size, min_size - 1, -1) if settings["auto_fit_text"] else (max_size,)
        limits = [full]
        if mode == "speech_bubble" and region.get("bubble_box"):
            limits.insert(0, capture_box(region["bubble_box"], capture_size, canvas_size))
        # Only the placed block retains its font. Probe items are measured at the origin.
        for limit in limits:
            max_width = limit[2] - limit[0] - 2 * pad - 2
            if max_width < 1:
                continue
            for size in sizes:
                font = Font(root=self.root, family="Segoe UI", size=-size, weight="bold")
                for wrap in dict.fromkeys((min(max_width, max(150, anchor[2] - anchor[0])),
                                           min(max_width, max(240, anchor[2] - anchor[0])),
                                           min(max_width, max(360, anchor[2] - anchor[0])))):
                    item = self.canvas.create_text(0, 0, text=region["text"], width=wrap,
                                                   anchor="nw", font=font,
                                                   fill=self.config["overlay_foreground"])
                    x1, y1, x2, y2 = self.canvas.bbox(item)
                    block = place_block(anchor, (x2 - x1 + 2 * pad, y2 - y1 + 2 * pad),
                                        limit, occupied)
                    if block is None:
                        self.canvas.delete(item)
                        continue
                    x, y = block[0] + pad - x1, block[1] + pad - y1
                    if mode == "transparent":
                        shadow = self.canvas.create_text(x + 1, y + 1, text=region["text"],
                                                         width=wrap, anchor="nw", font=font,
                                                         fill=self.config["overlay_background"])
                        self.canvas.tag_lower(shadow, item)
                    else:
                        background = self.canvas.create_rectangle(
                            *block, fill=self.config["overlay_background"], outline="")
                        self.canvas.tag_lower(background, item)
                    self.canvas.coords(item, x, y)
                    self.fonts.append(font)
                    return block
        log.debug("No room for overlay region %s", region["box"])
        return None
