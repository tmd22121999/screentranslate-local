import sys
import unittest
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from detection import TextRegion, group_regions, polygon_box
from main import load_config
from ocr import OCR
from overlay import Overlay, capture_box, place_block
from pipeline import Pipeline, same_regions
from translate import Translator


class PipelineTests(unittest.TestCase):
    def test_scaled_polygon_and_clamping(self):
        self.assertEqual(polygon_box([(-1, 3), (21, 30)], 10, 12, 2, 2), (0, 1, 10, 12))
        self.assertEqual(capture_box((20, 30, 101, 51), (100, 50), (200, 100)),
                         (40, 60, 200, 100))

    def test_grouping_keeps_bubbles_separate_and_suppresses_overlap(self):
        image = np.zeros((90, 150, 3), np.uint8)
        image[10:80, 10:65] = 255
        image[10:80, 80:140] = 255
        regions = group_regions(image, [(20, 18, 32, 64), (22, 20, 31, 61),
                                        (40, 20, 53, 70), (90, 30, 132, 47)])
        self.assertEqual([r.box for r in regions], [(20, 18, 53, 70), (90, 30, 132, 47)])
        self.assertEqual([r.orientation for r in regions], ["vertical", "horizontal"])
        self.assertEqual(group_regions(image, [(0, 0, 150, 90), (0, 0, 3, 4)]), [])

    def test_crop_padding_and_unchanged_ocr(self):
        ocr = OCR.__new__(OCR)
        ocr.padding = 4
        ocr.crop_cache = OrderedDict()
        ocr.manga = Mock(return_value=" 日本語 ")
        image = Image.new("RGB", (40, 40), "white")
        region = TextRegion((0, 1, 12, 20))
        self.assertEqual(ocr.recognize_crop(image, region), ("日本語", False))
        self.assertEqual(ocr.manga.call_args.args[0].size, (16, 24))
        image.putpixel((39, 39), (0, 0, 0))
        self.assertEqual(ocr.recognize_crop(image, region), ("日本語", True))
        ocr.manga.assert_called_once()

    def test_region_changes_ignore_background_motion_but_notice_disappearing_text(self):
        previous = np.zeros((40, 40, 3), np.uint8)
        previous[4:15, 4:15] = 255
        frame = previous.copy()
        frame[30:35, 30:35] = 255
        regions = [{"box": (4, 4, 15, 15)}]
        self.assertTrue(same_regions(frame, previous, regions, 2))
        frame[4:15, 4:15] = 0
        self.assertFalse(same_regions(frame, previous, regions, 2))

    def test_empty_and_failed_region_do_not_block_independent_translation(self):
        boxes = [TextRegion((i, i, i + 8, i + 8)) for i in (0, 10, 20, 30)]
        ocr = Mock()
        ocr.detector.detect.return_value = boxes
        ocr.recognize_crop.side_effect = [("", False), RuntimeError("bad crop"),
                                         ("一", False), ("二", False)]
        translator = Mock()
        translator.translate.side_effect = ["Một", "Hai"]
        pipeline = Pipeline({"debug": {"enabled": False}}, None)
        with self.assertLogs("pipeline", level="ERROR"):
            output = pipeline.process(np.zeros((40, 40, 3), np.uint8), ocr, translator, "ja")
        self.assertEqual([(r["box"], r["text"]) for r in output],
                         [(boxes[2].box, "Một"), (boxes[3].box, "Hai")])
        ocr.detector.detect.return_value = []
        self.assertEqual(pipeline.process(np.zeros((4, 4, 3), np.uint8), ocr, translator, "ja"), [])

    def test_translation_cache_is_normalized_and_language_specific(self):
        translator = Translator.__new__(Translator)
        translator.cache = OrderedDict()
        translator.lock = Lock()
        translator.tokenizer = Mock(return_value={})
        translator.tokenizer.batch_decode.return_value = ["Xin chào"]
        translator.model = Mock()
        translator.torch = SimpleNamespace(inference_mode=nullcontext)
        translator.max_tokens = 20
        self.assertEqual(translator.translate("  Hello\n there  ", "en"), "Xin chào")
        self.assertEqual(translator.translate("Hello there", "en"), "Xin chào")
        translator.model.generate.assert_called_once()
        translator.translate("Hello there", "ja")
        self.assertEqual(translator.model.generate.call_count, 2)

    def test_older_updates_cannot_restore_cleared_overlay(self):
        overlay = Overlay.__new__(Overlay)
        overlay.generation = -1
        overlay.regions = []
        overlay.draw_translations = Mock()
        overlay.accept(10, [{"box": (0, 0, 10, 10), "text": "Một"}])
        overlay.accept(11, [])
        overlay.accept(10, [{"box": (0, 0, 10, 10), "text": "Cũ"}])
        self.assertEqual(overlay.regions, [])
        self.assertEqual(overlay.draw_translations.call_count, 2)

    def test_placement_clamps_and_avoids_adjacent_block(self):
        self.assertEqual(place_block((90, 90, 100, 100), (30, 20), (0, 0, 100, 100), []),
                         (70, 80, 100, 100))
        self.assertEqual(place_block((5, 5, 30, 25), (40, 20), (0, 0, 100, 100),
                                     [(0, 0, 50, 25)]), (5, 25, 45, 45))


@unittest.skipUnless(sys.platform == "win32", "requires Windows desktop")
class WindowsOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from overlay import configure_dpi
        configure_dpi()

    def test_wrap_offsets_clamping_and_single_clear(self):
        cfg = load_config(Path(__file__).resolve().with_name("config.json"))
        cfg["region"] = {"left": 45, "top": 60, "width": 400, "height": 160}
        cfg["overlay"]["background_mode"] = "text_region"
        cfg["overlay"]["max_font_size"] = 21
        overlay = Overlay(cfg)
        self.addCleanup(overlay.root.destroy)
        canvas = overlay.canvas
        self.assertEqual((canvas.winfo_rootx(), canvas.winfo_rooty()), (45, 60))
        canvas.delete = Mock(wraps=canvas.delete)
        regions = [{"text": "Xin chào các bạn. Hẹn gặp lại!", "box": (20, 20, 115, 70)},
                   {"text": "Một đoạn khác", "box": (300, 90, 395, 155)}]
        overlay.accept(1, regions)
        items = canvas.find_all()
        self.assertEqual(len(items), 4)
        self.assertEqual([call.args for call in canvas.delete.call_args_list if call.args == ("all",)],
                         [("all",)])
        texts = [item for item in items if canvas.type(item) == "text"]
        x1, y1, x2, y2 = canvas.bbox(texts[0])
        self.assertGreater(y2 - y1, cfg["overlay"]["max_font_size"] * 2)
        for item in items:
            x1, y1, x2, y2 = canvas.bbox(item)
            self.assertTrue(0 <= x1 < x2 <= 400 and 0 <= y1 < y2 <= 160)
        hotkey = Mock(side_effect=[1, 1])
        overlay.user32.GetAsyncKeyState = hotkey
        overlay.check_hotkey()
        self.assertEqual(canvas.find_all(), ())
        overlay.check_hotkey()
        self.assertEqual(len(canvas.find_all()), 4)

    def test_capture_fallback_hides_and_restores_overlay(self):
        import mss
        import tkinter as tk

        cfg = load_config(Path(__file__).resolve().with_name("config.json"))
        cfg["region"] = {"left": 45, "top": 60, "width": 400, "height": 160}
        cfg["overlay"]["hide_during_capture"] = True
        cfg["overlay"]["background_mode"] = "text_region"
        cfg["overlay_background"] = "#101010"
        backdrop = tk.Tk()
        self.addCleanup(backdrop.destroy)
        backdrop.configure(bg="#35c063")
        backdrop.overrideredirect(True)
        backdrop.attributes("-topmost", True)
        backdrop.geometry("400x160+45+60")
        backdrop.update()
        overlay = Overlay(cfg)
        self.addCleanup(overlay.root.destroy)
        overlay.accept(1, [{"text": "Xin chào", "box": (20, 20, 100, 50)}])
        overlay.root.update()
        overlay.flush_compositor()
        style = overlay.user32.GetWindowLongW(overlay.hwnd, -20)
        self.assertTrue(style & 0x20 and style & 0x80000)
        with mss.mss() as screen:
            before = np.asarray(screen.grab(cfg["region"]))[22, 22, :3].tolist()
            during = np.asarray(overlay.grab(screen))[22, 22, :3].tolist()
            overlay.flush_compositor()
            after = np.asarray(screen.grab(cfg["region"]))[22, 22, :3].tolist()
        self.assertEqual(before, [16, 16, 16])
        self.assertEqual(during, [99, 192, 53])
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
