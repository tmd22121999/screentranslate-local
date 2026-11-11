from dataclasses import dataclass
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

@dataclass
class TextBlock:
    text: str
    box: tuple[int, int, int, int]
    confidence: float

class OCR:
    def __init__(self, min_confidence: float = .55, scale: float = 1.5):
        self.engine = RapidOCR()
        self.min_confidence = min_confidence
        self.scale = scale

    def recognize(self, rgb: np.ndarray) -> list[TextBlock]:
        h, w = rgb.shape[:2]
        image = Image.fromarray(rgb)
        if self.scale != 1:
            image = image.resize((round(w * self.scale), round(h * self.scale)), Image.Resampling.LANCZOS)
        # RapidOCR accepts numpy RGB images.
        result, _ = self.engine(np.asarray(image))
        blocks = []
        for points, text, confidence in result or []:
            if float(confidence) < self.min_confidence:
                continue
            xs = [float(p[0]) / self.scale for p in points]
            ys = [float(p[1]) / self.scale for p in points]
            box = (max(0, int(min(xs))), max(0, int(min(ys))),
                   min(w, int(max(xs))), min(h, int(max(ys))))
            if box[2] > box[0] and box[3] > box[1]:
                blocks.append(TextBlock(str(text), box, float(confidence)))
        return sorted(blocks, key=lambda b: (b.box[1], b.box[0]))
