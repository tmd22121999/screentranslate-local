import tkinter as tk
import ctypes

from ctypes import wintypes

# Windows constants
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080

TRANSPARENT_COLOR = "#010203"

user32 = ctypes.windll.user32


class TestOverlay:
    def __init__(self):
        self.root = tk.Tk()

        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        # Fullscreen overlay
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()

        self.root.geometry(f"{width}x{height}+0+0")

        # Transparent background
        self.root.configure(bg=TRANSPARENT_COLOR)
        self.root.wm_attributes(
            "-transparentcolor",
            TRANSPARENT_COLOR,
        )

        self.canvas = tk.Canvas(
            self.root,
            bg=TRANSPARENT_COLOR,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Test text
        self.canvas.create_text(
            100,
            100,
            text="TEST OVERLAY - Xin chào",
            fill="#FF0000",
            font=("Arial", 36, "bold"),
            anchor="nw",
        )

        # Diagnostic rectangle
        self.canvas.create_rectangle(
            90,
            90,
            650,
            170,
            outline="#00FF00",
            width=3,
        )

        # Apply Win32 styles after window initialization
        self.root.after(200, self.enable_clickthrough)

        # Auto close after 15 seconds
        self.root.after(15000, self.root.destroy)

    def enable_clickthrough(self):
        hwnd = self.root.winfo_id()

        get_style = user32.GetWindowLongW
        get_style.argtypes = [wintypes.HWND, ctypes.c_int]
        get_style.restype = ctypes.c_long

        set_style = user32.SetWindowLongW
        set_style.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_long,
        ]
        set_style.restype = ctypes.c_long

        style = get_style(hwnd)

        style |= (
            WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_TOOLWINDOW
        )

        set_style(hwnd, GWL_EXSTYLE, style)

        print("HWND:", hwnd)
        print("Window style:", hex(style))
        print("Canvas items:", self.canvas.find_all())
        print("Visible:", self.root.winfo_viewable())

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    TestOverlay().run()