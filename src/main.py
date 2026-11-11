import ctypes
import json
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import mss
import numpy as np
from PIL import Image

from ocr import OCR
from translate import Translator

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
REGION = CFG['region']

# A complete capture-to-translation job; stale jobs are discarded.
class Pipeline:
    def __init__(self):
        self.events = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.ocr = None
        self.translator = None
        self.last_hash = None
        self.generation = 0

    def publish(self, value):
        try:
            self.events.put_nowait(value)
        except queue.Full:
            try: self.events.get_nowait()
            except queue.Empty: pass
            self.events.put_nowait(value)

    def run(self):
        try:
            self.publish(('status', 'Đang tải OCR và model dịch...'))
            self.ocr = OCR(CFG['ocr_min_confidence'], CFG['ocr_scale']) if CFG['source_language'] != 'ja' else None
            if CFG['source_language'] == 'ja':
                from manga_ocr import MangaOcr
                self.ja_ocr = MangaOcr()
            self.translator = Translator(CFG['translation_model'], CFG['translation_max_tokens'])
            self.publish(('status', 'Sẵn sàng. F8: ẩn/hiện overlay; Ctrl+C trong console để thoát.'))
            with mss.mss() as sct:
                while not self.stop.is_set():
                    t0 = time.monotonic()
                    shot = np.asarray(sct.grab(REGION))[:, :, :3][:, :, ::-1].copy()
                    # Tiny grayscale thumbnail avoids re-running OCR on identical frames.
                    small = np.asarray(Image.fromarray(shot).convert('L').resize((64, 24)))
                    if self.last_hash is not None:
                        difference = np.abs(small.astype(np.int16) - self.last_hash.astype(np.int16)).mean()
                        if difference < CFG['change_threshold']:
                            self.stop.wait(CFG['capture_interval_ms'] / 1000)
                            continue
                    self.last_hash = small
                    if CFG['source_language'] == 'ja':
                        source_ja = self.ja_ocr(Image.fromarray(shot))
                        blocks = []
                    else:
                        blocks = self.ocr.recognize(shot)
                    # Group all OCR lines in the selected dialogue region into one sentence.
                    # This preserves some sentence context, at the cost of coarse placement.
                    source = source_ja.strip() if CFG['source_language'] == 'ja' else ' '.join(b.text.strip() for b in blocks if b.text.strip())
                    if source:
                        translated = self.translator.translate(source, CFG['source_language'])
                        if CFG['source_language'] == 'ja':
                            self.publish(('translation', translated, (0, 0, REGION['width'], REGION['height'])))
                        elif blocks:
                            x1 = min(b.box[0] for b in blocks)
                            y1 = min(b.box[1] for b in blocks)
                            x2 = max(b.box[2] for b in blocks)
                            y2 = max(b.box[3] for b in blocks)
                            self.publish(('translation', translated, (x1, y1, x2, y2)))
                    else:
                        self.publish(('clear',))
                    elapsed = time.monotonic() - t0
                    self.stop.wait(max(0.0, CFG['capture_interval_ms']/1000 - elapsed))
        except Exception as exc:
            self.publish(('error', f'{type(exc).__name__}: {exc}'))

class Overlay:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.root = tk.Tk()
        self.root.title('ScreenTranslate Local')
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.geometry(f"{REGION['width']}x{REGION['height']}+{REGION['left']}+{REGION['top']}")
        self.transparent = '#ff00ff'
        self.root.configure(bg=self.transparent)
        self.root.wm_attributes('-transparentcolor', self.transparent)
        self.canvas = tk.Canvas(self.root, bg=self.transparent, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.visible = True
        self.root.update_idletasks()
        self.setup_win32()
        self.root.after(100, self.poll)
        self.root.after(100, self.check_hotkey)

    def setup_win32(self):
        hwnd = self.root.winfo_id()
        user32 = ctypes.windll.user32
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x80000
        WS_EX_TRANSPARENT = 0x20
        WS_EX_TOOLWINDOW = 0x80
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW)
        # Exclude the overlay from Windows capture to prevent OCR feedback loops.
        # WDA_EXCLUDEFROMCAPTURE requires Windows 10 2004+ and compatible capture path.
        ok = user32.SetWindowDisplayAffinity(hwnd, 0x11)
        if not ok:
            print('WARNING: Overlay exclusion unavailable. Capture may include translated text.')
        self.user32 = user32

    def check_hotkey(self):
        # F8 toggles rendering; polling avoids external hotkey packages.
        if self.user32.GetAsyncKeyState(0x77) & 1:
            self.visible = not self.visible
            if not self.visible: self.canvas.delete('all')
        self.root.after(90, self.check_hotkey)

    def draw(self, text, box):
        if not self.visible:
            return
        self.canvas.delete('all')
        x1, y1, x2, y2 = box
        margin = 8
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(REGION['width'], x2 + margin)
        # Expand the replacement background vertically for translated text wrapping.
        text_width = max(180, min(REGION['width'] - x1 - 8, max(300, x2 - x1)))
        font_size = CFG['font_size']
        text_id = self.canvas.create_text(x1 + 6, y1 + 5, anchor='nw', text=text,
                                          fill=CFG['overlay_foreground'],
                                          font=('Segoe UI', font_size, 'bold'),
                                          width=text_width - 12)
        bounds = self.canvas.bbox(text_id)
        if bounds:
            bottom = min(REGION['height'], max(y2, bounds[3] + 5))
            bg = self.canvas.create_rectangle(x1, y1, min(REGION['width'], x1 + text_width), bottom,
                                               fill=CFG['overlay_background'], outline='')
            self.canvas.tag_lower(bg, text_id)

    def poll(self):
        try:
            while True:
                msg = self.pipeline.events.get_nowait()
                if msg[0] == 'translation': self.draw(msg[1], msg[2])
                elif msg[0] == 'clear': self.canvas.delete('all')
                elif msg[0] == 'status': print(msg[1], flush=True)
                elif msg[0] == 'error':
                    print('ERROR:', msg[1], flush=True)
                    self.canvas.delete('all')
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def run(self):
        try: self.root.mainloop()
        finally: self.pipeline.stop.set()

if __name__ == '__main__':
    if sys.platform != 'win32':
        raise SystemExit('This app requires Windows 10/11.')
    pipeline = Pipeline()
    threading.Thread(target=pipeline.run, daemon=True).start()
    Overlay(pipeline).run()
