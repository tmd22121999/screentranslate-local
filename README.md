# ScreenTranslate Local — Windows MVP

Local Japanese/English OCR -> Vietnamese translation -> click-through overlay.

## Installation
1. Install Python 3.11 (64-bit) and check `py -3.11 --version`.
2. Extract this folder. Run `install.bat` (internet required for initial model/package download).
3. Set the `region` in `config.json` using **physical screen pixels**: left/top/width/height. For example a 1920x1080 screen, a dialogue box near bottom might be `{ "left": 180, "top": 650, "width": 1100, "height": 260 }`.
4. Run `run.bat`. First run downloads NLLB model from Hugging Face (several GB including weights/dependencies). Subsequent runs can be offline if cached.
5. F8 toggles overlay visibility. Ctrl+C in terminal or close terminal to quit.

## Important limitations
- **MVP / not benchmarked:** Dependencies and model compatibility have not been installed or run on the target Windows PC. Test OCR quality and translation latency before production use.
- Japanese OCR uses **MangaOCR**, while English OCR uses **RapidOCR**. Set `source_language` to `ja` or `en` in config.json. `auto` is not a validated bilingual OCR mode and may miss Japanese text. MangaOCR is optimized for Japanese manga and may struggle with game UI typography. It recognizes the entire selected region, so keep the region tightly cropped.
- Current overlay groups all recognized text into one block; Japanese mode covers the entire selected region. It does not replace each original word exactly. It covers the bounding rectangle and renders translated text at its top-left.
- OCR only runs when a 64x24 image thumbnail changes enough; subtle text changes can be missed. Adjust `change_threshold` downward or disable detection for testing.
- Overlay exclusion from screen capture is Windows/capture-backend dependent. Check the warning in console and confirm the overlay does not recursively OCR its own translated text.
- Overlay may not work over exclusive fullscreen, protected video, elevated apps, or some anti-cheat-protected games. Prefer borderless windowed mode; do not bypass anti-cheat.
- Model is CPU-based, even on Radeon RX 7800 XT; NLLB-600M latency may exceed 1 second for some sentences. The interval is a **minimum wait**, not a guaranteed processing rate.
- Set source_language explicitly to ja or en for best OCR selection. In auto mode, RapidOCR is used and Japanese is not reliable. Translation language detection uses kana/kanji heuristics. Japanese text written only in Latin letters is detected as English. Override `source_language` with `ja` or `en` when known.
- Screen scaling >100%, mixed DPI monitors, multiple monitors, and coordinate alignment may require adjustments. Test at 100% scaling on the primary monitor first.
- Overlay text can overflow the selected capture region for long Vietnamese translations.
- This app downloads a model at first launch; it does not send captured text to remote translation APIs.

## Architecture
`MSS capture -> thumbnail change detector -> MangaOCR (ja) or RapidOCR ONNX (en) -> normalize/group text -> NLLB translation (CPU) -> Tkinter Win32 click-through overlay`.

## Suggested production improvements
- Japanese-specific OCR model selection, and separate Japanese/English routing.
- Windows Graphics Capture with exclusion verification.
- Per-line grouping with geometric reading-order handling, temporal stabilization, and debouncing.
- Independent capture/OCR/translation workers with bounded latest-frame queues and cancellation of stale results.
- Dynamic text sizing and inpainting or background-aware masks.
- Configurable UI region picker, hotkeys, tray icon, debug screenshots and latency telemetry.
- Optional DirectML inference only after backend/model compatibility and benchmarks are verified.
