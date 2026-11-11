import logging
import queue
import threading
import time
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

from detection import clamp
from ocr import OCR
from translate import Translator

log = logging.getLogger(__name__)


def put_latest(mailbox, value):
    try:
        mailbox.get_nowait()
    except queue.Empty:
        pass
    mailbox.put_nowait(value)


def same_regions(frame, previous, regions, padding):
    height, width = frame.shape[:2]
    for region in regions:
        x1, y1, x2, y2 = region["box"]
        x1, y1, x2, y2 = clamp((x1 - padding, y1 - padding, x2 + padding, y2 + padding),
                              width, height)
        if not np.array_equal(frame[y1:y2, x1:x2], previous[y1:y2, x1:x2]):
            return False
    return True


class Pipeline:
    def __init__(self, config, debug_dir):
        self.config = config
        self.debug_dir = debug_dir
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.ready = threading.Event()

    def run(self):
        cfg = self.config
        language = cfg["source_language"]
        try:
            print("Đang tải OCR và model dịch...", flush=True)
            ocr = OCR(cfg["ocr_min_confidence"], cfg["ocr_scale"], language=language,
                      padding=cfg["detection"]["padding"], min_side=cfg["detection"]["min_side"])
            translator = Translator(cfg["translation_model"], cfg["translation_max_tokens"])
        except Exception:
            log.exception("Could not load OCR or translation models")
            return
        self.ready.set()
        print("Sẵn sàng. F8: ẩn/hiện overlay.", flush=True)
        previous = None
        regions = []
        last_scan = 0.0
        rescan = cfg["detection"]["rescan_ms"] / 1000
        padding = max(0, cfg["detection"]["padding"])
        while not self.stop.is_set():
            try:
                generation, frame = self.frames.get(timeout=.2)
            except queue.Empty:
                continue
            remaining = last_scan + rescan - time.monotonic()
            # A deferred scan still runs if the background stops changing.
            while (regions and remaining > 0 and not self.stop.is_set()
                   and same_regions(frame, previous, regions, padding)):
                try:
                    generation, frame = self.frames.get(timeout=min(remaining, .1))
                except queue.Empty:
                    pass
                remaining = last_scan + rescan - time.monotonic()
            if self.stop.is_set():
                break
            try:
                last_scan = time.monotonic()
                regions = self.process(frame, ocr, translator, language)
                previous = frame
            except Exception:
                log.exception("Frame %d failed", generation)
                regions = []
            if not self.stop.is_set():
                put_latest(self.events, ("translations", generation, regions))

    def process(self, frame, ocr, translator, language):
        debug = self.config["debug"]
        start = time.perf_counter()
        regions = ocr.detector.detect(frame) if ocr.detector else ocr.recognize(frame)
        if debug["enabled"]:
            log.info("Detection + line processing: %d regions in %.1f ms",
                     len(regions), (time.perf_counter() - start) * 1000)
        image = Image.fromarray(frame) if ocr.detector else None
        if debug["enabled"] and (debug["save_capture"] or debug["save_detected_regions"]):
            try:
                self.save_debug(image if image is not None else Image.fromarray(frame), regions)
            except OSError:
                log.exception("Could not write debug images")
        translated = []
        for region in regions:
            try:
                if ocr.detector:
                    region.source_text, cached = ocr.recognize_crop(image, region)
                    if debug["enabled"] and cached:
                        log.info("OCR cache hit %s", region.box)
                if debug["enabled"] and debug["log_ocr"]:
                    log.info("OCR %s: %r", region.box, region.source_text)
                if not region.source_text.strip():
                    if debug["enabled"]:
                        log.info("Empty OCR result %s", region.box)
                    continue
                start = time.perf_counter()
                region.translated_text = translator.translate(region.source_text, language)
                if debug["enabled"]:
                    log.info("Translation %s: %.1f ms", region.box,
                             (time.perf_counter() - start) * 1000)
                    if debug["log_translation"]:
                        log.info("Translation %s: %r", region.box, region.translated_text)
                if region.translated_text.strip():
                    translated.append({"text": region.translated_text, "box": region.box,
                                       "bubble_box": region.bubble_box})
            except Exception:
                log.exception("Region %s failed", region.box)
        return translated

    def save_debug(self, image, regions):
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        debug = self.config["debug"]
        if debug["save_capture"]:
            image.save(self.debug_dir / f"{stamp}-capture.png")
        if debug["save_detected_regions"]:
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            for index, region in enumerate(regions, 1):
                x1, y1, x2, y2 = region.box
                draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline="red", width=2)
                draw.text((x1, y1), f"{index} {region.orientation}", fill="red")
            annotated.save(self.debug_dir / f"{stamp}-regions.png")
