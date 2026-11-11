# ScreenTranslate Local

Capture a Windows desktop region, recognize Japanese or English text, and overlay Vietnamese translations. Detection and model inference run locally on CPU.

## Windows setup

1. Install 64-bit Python 3.11 with the Windows Python launcher. Check with `py -3.11 --version`.
2. Run `install.bat` to create `.venv` and install dependencies. If the environment already exists, update it with `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
3. In `config.json`, set `source_language` to `ja` or `en` and configure `region` in physical screen pixels. Use negative `left` or `top` for monitors left of or above the primary display.
4. Run `run.bat`. The first launch needs internet access for uncached models. Press **F8** to hide or restore the latest overlay. Close the console to quit.

| Model | Approximate download or installed size | Location |
| --- | --- | --- |
| RapidOCR ONNX detection, recognition and classifier | 16.2 MB bundled, including the 4.75 MB detector | `.venv\Lib\site-packages\rapidocr_onnxruntime\models` |
| MangaOCR, Japanese only | 444 MB for one weight file | Hugging Face Hub cache |
| NLLB-200 distilled 600M | 2.46 GB for one weight file, plus up to about 22 MB for tokenizer files | Hugging Face Hub cache |

The default Hub cache is `%USERPROFILE%\.cache\huggingface\hub`. An existing `HF_HUB_CACHE` or `HF_HOME` override is honored by Hugging Face. See the published [MangaOCR files](https://huggingface.co/kha-white/manga-ocr-base/tree/main) and [NLLB files](https://huggingface.co/facebook/nllb-200-distilled-600M/tree/main).

## Detection and placement

Japanese mode runs the PP-OCRv4 detector bundled with RapidOCR, groups lines when their orientation and shared enclosed light background support doing so, then runs MangaOCR on padded crops. English mode uses RapidOCR's recognition and confidence scores. Both paths translate each region with the existing NLLB model and render the result in capture-relative coordinates.

The bundled PaddleOCR detector uses the existing Windows CPU ONNX stack with no separate weight download. [CRAFT](https://github.com/clovaai/CRAFT-pytorch) provides character and affinity maps and would still need dialogue grouping. [Comic Text Detector](https://github.com/dmMaze/comic-text-detector) produces comic text blocks and masks, but would add separate weights and a comic-specific inference path for a project that also targets web text. The included model smoke test checks the selected detector on two vertical and one horizontal synthetic Japanese bubbles. Real manga may vary with artwork, stylized lettering, furigana, and dark or open bubbles. Lines without sufficient grouping evidence remain separate, including on web pages.

- `detection.padding` adds crop context and `detection.min_side` rejects tiny detections.
- Identical frames skip detection. When only pixels outside known text regions change, `detection.rescan_ms` limits how soon detection runs again. A deferred scan still looks for new text. Set this to `0` for immediate rescans. Unchanged Japanese crops reuse recognition and translations are cached by language and normalized source text. The old `change_threshold` setting is obsolete.
- `overlay.background_mode` accepts `text_region` for a compact text background, `transparent` for shadowed text, or `speech_bubble` to prefer fitting inside the detected dialogue span when grouping finds an enclosed light area. This is a text-span heuristic. It does not recover a bubble contour or replace artwork.
- `overlay.padding`, `min_font_size`, `max_font_size`, and `auto_fit_text` control placement. Font sizes are physical pixels. Text wraps and can shift slightly to stay in bounds and avoid another translation. If a translation still cannot fit, it is omitted. Enlarge the capture or lower `min_font_size` if this occurs. Existing configs use their `font_size` as the default maximum.
- `auto` uses RapidOCR and chooses the translation language from recognized text. Use `ja` for Japanese detection and recognition.

## Capture and diagnostics

The app resolves Tk's top-level Win32 window, positions it in physical screen coordinates, and checks capture affinity. When exclusion is unavailable, it briefly hides the overlay during each MSS capture and restores it without activating it. Set `overlay.hide_during_capture` to `true` if you see translated text in saved captures despite affinity being enabled. The fallback can flicker. Use a windowed or borderless source for capture.

Enable `debug.enabled` for region timing and cache diagnostics. Enable `save_capture` or `save_detected_regions` to write timestamped PNGs under `debug\`. The latter labels boxes and orientations. Set `log_ocr` and `log_translation` to print the corresponding text. Image saving and text logging are individually opt-in. Older boolean `debug` settings continue to control diagnostics.

## Verification

Run from the repository root with the virtual environment activated:

```bat
python -B -m unittest discover -v
python -B tests_detector_smoke.py
python -B tests_detector_smoke.py --translate
```

The Windows overlay tests require an interactive desktop. The smoke test uses the Windows Japanese Gothic font and the bundled RapidOCR models. The optional translation check also loads MangaOCR and NLLB. These checks cover synthetic dialogue and a Windows Canvas, with no representative manga accuracy or target-hardware performance benchmark.
