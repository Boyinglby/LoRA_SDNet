# coding=utf-8
# Modernized tokenizer adapter for SDNet
# Works with Hugging Face transformers and keeps the old API surface.

from __future__ import absolute_import, division, print_function

import os
import logging
from typing import List, Union

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

try:
    from transformers import BertTokenizer as HFBertTokenizer
except Exception as e:
    raise RuntimeError(
        "transformers is required. Install with: pip install transformers"
    ) from e


class BertTokenizer(object):
    """
    Compatibility wrapper so legacy SDNet code can call:
        BertTokenizer.from_pretrained(<name_or_path>)
    or instantiate with a local vocab file path, while we actually use 🤗 transformers.
    """

    def __init__(self, vocab_or_name: str, do_lower_case: bool = True):
        """
        If `vocab_or_name` is a file path to a vocab.txt -> load tokenizer from that file.
        Otherwise treat it as a Hugging Face model name like 'bert-large-uncased'.
        """
        if os.path.isfile(vocab_or_name):
            logger.info(f"Loading BERT tokenizer from local vocab file: {vocab_or_name}")
            self._tok = HFBertTokenizer(vocab_file=vocab_or_name, do_lower_case=do_lower_case)
        else:
            logger.info(f"Loading BERT tokenizer from HF hub: {vocab_or_name}")
            # This will download/cache the vocab and merges as needed
            self._tok = HFBertTokenizer.from_pretrained(vocab_or_name, do_lower_case=do_lower_case)

    def tokenize(self, text: str) -> List[str]:
        return self._tok.tokenize(text)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return self._tok.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return self._tok.convert_ids_to_tokens(ids)

    @property
    def vocab(self):
        # Expose a dict token->id (legacy code sometimes accesses this)
        return self._tok.get_vocab()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, do_lower_case: bool = True):
        """
        Mirrors old API but delegates to transformers:
          - If you pass 'bert-large-uncased' (or any HF model id), it loads from the hub.
          - If you pass a local vocab.txt path, it loads from that file.
        """
        return cls(pretrained_model_name_or_path, do_lower_case=do_lower_case)

