import json
import logging
import queue
import sys
import threading
from pathlib import Path

import mss
import numpy as np

from overlay import Overlay, configure_dpi
from pipeline import Pipeline, put_latest

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger(__name__)


def load_config(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["overlay"] = {
        "background_mode": "text_region",
        "padding": 6,
        "min_font_size": 12,
        "max_font_size": cfg.get("font_size", 24),
        "auto_fit_text": True,
        "hide_during_capture": False,
        **cfg.get("overlay", {}),
    }
    cfg["detection"] = {"padding": 4, "min_side": 8, "rescan_ms": 900, **cfg.get("detection", {})}
    debug = cfg.get("debug", False)
    cfg["debug"] = {
        "enabled": bool(debug) if isinstance(debug, bool) else False,
        "save_capture": False, "save_detected_regions": False,
        "log_ocr": False, "log_translation": False,
        **(debug if isinstance(debug, dict) else {}),
    }
    return cfg


def main():
    if sys.platform != "win32":
        raise SystemExit("This app requires Windows 10/11.")
    cfg = load_config(ROOT / "config.json")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if cfg["debug"]["enabled"]:
        logging.getLogger("overlay").setLevel(logging.DEBUG)
        logging.getLogger("translate").setLevel(logging.DEBUG)
    configure_dpi()
    overlay = Overlay(cfg)
    pipeline = Pipeline(cfg, ROOT / "debug")
    worker = threading.Thread(target=pipeline.run, daemon=True)
    worker.start()
    generation = 0
    last_frame = None
    with mss.mss() as screen:
        def capture():
            nonlocal generation, last_frame
            if pipeline.ready.is_set():
                try:
                    frame = np.asarray(overlay.grab(screen))[:, :, :3][:, :, ::-1].copy()
                    if last_frame is None or not np.array_equal(last_frame, frame):
                        generation += 1
                        put_latest(pipeline.frames, (generation, frame))
                        last_frame = frame
                except Exception:
                    log.exception("Capture failed")
            overlay.root.after(max(1, cfg["capture_interval_ms"]), capture)

        def poll():
            try:
                _, frame_id, regions = pipeline.events.get_nowait()
            except queue.Empty:
                pass
            else:
                overlay.accept(frame_id, regions)
            overlay.root.after(50, poll)

        overlay.root.after(0, capture)
        overlay.root.after(0, poll)
        try:
            overlay.root.mainloop()
        finally:
            pipeline.stop.set()


if __name__ == "__main__":
    main()
