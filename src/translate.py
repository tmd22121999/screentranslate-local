import logging
import re
import threading
from collections import OrderedDict

LANG_CODES = {"ja": "jpn_Jpan", "en": "eng_Latn"}
log = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    if re.search(r'[\u3040-\u30ff\u3400-\u9fff]', text):
        return 'ja'
    return 'en'


class Translator:
    def __init__(self, model_name: str, max_tokens: int = 180):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval().to('cpu')
        self.max_tokens = max_tokens
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        torch.set_num_threads(max(1, min(6, ( __import__('os').cpu_count() or 6) - 2)))

    def translate(self, text: str, language: str = 'auto') -> str:
        text = ' '.join(text.split())
        if not text:
            return ''
        lang = detect_language(text) if language == 'auto' else language
        key = (lang, text)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                log.debug("Translation cache hit (%s, %d characters)", lang, len(text))
                return self.cache[key]
            self.tokenizer.src_lang = LANG_CODES[lang]
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=384)
            target_id = self.tokenizer.convert_tokens_to_ids('vie_Latn')
            with self.torch.inference_mode():
                ids = self.model.generate(**inputs, forced_bos_token_id=target_id,
                                          max_new_tokens=self.max_tokens, num_beams=1)
            output = self.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
            self.cache[key] = output
            if len(self.cache) > 1000:
                self.cache.popitem(last=False)
            return output
