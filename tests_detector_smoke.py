"""Optional model-backed smoke test for Windows with Japanese system fonts."""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from detection import JapaneseDetector
from main import load_config
from ocr import OCR
from pipeline import Pipeline
from translate import Translator

def main():
    image = Image.new("RGB", (720, 420), "#494949")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(Path(os.environ["WINDIR"]) / "Fonts" / "msgothic.ttc"), 32)
    bubbles = [(20, 20, 230, 370), (265, 40, 465, 360), (490, 90, 700, 290)]
    for box in bubbles:
        draw.ellipse(box, fill="white", outline="black", width=3)
    for x, y, line in [(135, 70, "今日は良い天気"), (95, 90, "散歩しよう"),
                       (365, 85, "本当にそう思う"), (325, 110, "私も行きたい")]:
        for i, char in enumerate(line):
            draw.text((x, y + i * 32), char, font=font, fill="black")
    draw.text((510, 158), "ありがとう", font=font, fill="black")
    draw.text((515, 195), "また明日ね", font=font, fill="black")
    regions = JapaneseDetector(RapidOCR(), 8).detect(np.asarray(image))
    print([(r.box, r.orientation) for r in regions])
    assert len(regions) == 3
    assert [r.orientation for r in regions].count("vertical") == 2
    assert [r.orientation for r in regions].count("horizontal") == 1

    english = Image.new("RGB", (420, 120), "white")
    ImageDraw.Draw(english).text((25, 35), "Hello there!", font=font, fill="black")
    assert any("hello" in r.source_text.lower() for r in OCR().recognize(np.asarray(english)))

    if "--translate" in sys.argv:
        root = Path(__file__).resolve().parent
        config = load_config(root / "config.json")
        config["debug"]["enabled"] = False
        ocr = OCR(language="ja")
        translator = Translator(config["translation_model"], config["translation_max_tokens"])
        regions = Pipeline(config, root / "debug").process(np.asarray(image), ocr, translator, "ja")
        print(regions)
        assert len(regions) == 3 and all(region["text"].strip() for region in regions)


if __name__ == "__main__":
    main()
