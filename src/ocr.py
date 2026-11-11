from collections import OrderedDict
from hashlib import blake2b

import cv2
import numpy as np
from PIL import Image

from detection import JapaneseDetector, TextRegion, clamp, orientation, polygon_box


class OCR:
    def __init__(self, min_confidence=.55, scale=1.5, *, language="en", padding=4, min_side=8):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()
        self.min_confidence = min_confidence
        self.scale = scale
        self.padding = max(0, padding)
        self.crop_cache = OrderedDict()
        self.detector = None
        if language == "ja":
            from manga_ocr import MangaOcr

            self.detector = JapaneseDetector(self.engine, min_side)
            self.manga = MangaOcr(force_cpu=True)

    def recognize(self, rgb):
        image = Image.fromarray(rgb)
        width, height = image.size
        if self.scale != 1:
            image = image.resize((max(1, round(width * self.scale)),
                                  max(1, round(height * self.scale))), Image.Resampling.LANCZOS)
        result, _ = self.engine(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        regions = []
        for points, text, confidence in result or []:
            text = str(text).strip() if text else ""
            if not text or float(confidence) < self.min_confidence:
                continue
            box = polygon_box(points, width, height, image.width / width, image.height / height)
            if box[2] > box[0] and box[3] > box[1]:
                regions.append(TextRegion(box, text, confidence=float(confidence),
                                          orientation=orientation(box)))
        return sorted(regions, key=lambda r: (r.box[1], r.box[0]))

    def recognize_crop(self, image, region):
        x1, y1, x2, y2 = region.box
        pad = self.padding
        crop = image.crop(clamp((x1 - pad, y1 - pad, x2 + pad, y2 + pad), *image.size))
        key = (crop.size, blake2b(crop.tobytes(), digest_size=16).digest())
        if key in self.crop_cache:
            self.crop_cache.move_to_end(key)
            return self.crop_cache[key], True
        result = self.manga(crop)
        text = result.strip() if result else ""
        self.crop_cache[key] = text
        if len(self.crop_cache) > 512:
            self.crop_cache.popitem(last=False)
        return text, False
