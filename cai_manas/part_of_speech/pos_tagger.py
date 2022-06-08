import os
import json
import logging
from typing import Tuple, Dict, List

import torch
import numpy as np
from transformers import (
    AutoConfig,
    AlbertForTokenClassification)

from ..tokenizer import TibertTokenizer
from ..models.utils import get_local_ckpt

logger = logging.getLogger(__name__)

class PartOfSpeechTagger:

    def __init__(self, tokenizer_name, config_name, model_ckpt: str) -> None:
        logger.debug(f"Loading tokenizer {tokenizer_name}")
        self.tokenizer = TibertTokenizer.from_pretrained(TibertTokenizer.get_local_model_dir(tokenizer_name))
        self.tokenizer.stochastic_tokenization = False
        self.tokenizer.tsheg_pretokenization = True

        local_ckpt = get_local_ckpt(model_ckpt)
        logger.debug(f"Local model checkpoint {model_ckpt} resolved to {local_ckpt}")

        logger.debug(f"Loading model config.json")
        config_json_fn = os.path.join(os.path.dirname(local_ckpt), "config.json")
        with open(config_json_fn, 'r') as f:
            config_json = json.load(f)
        logger.debug(f"Extracting label2id maps")
        self.id_to_label_map = {
            int(id): label
            for id, label in config_json["id2label"].items()}

        logger.debug(f"Loading Huggingface model config")
        self.model_cfg = AutoConfig.from_pretrained(
            config_name,
            vocab_size=self.tokenizer.vocab_size,
            num_labels=len(self.id_to_label_map),
            id2label=self.id_to_label_map)

        logger.debug(f"Loading model")
        self.model = AlbertForTokenClassification.from_pretrained(local_ckpt, config=self.model_cfg)
        logger.debug(f"Configuring model")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def predict_tags(self, bo_text: str) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        tokens = self.tokenizer(bo_text, return_tensors='pt')
        mdl_res = self.model(**tokens)[0][0]
        return tokens, np.argmax(mdl_res.detach().numpy(), axis=1)

    def tag(self, bo_text: str) -> Dict[str, List[str]]:
        tokens, cur_preds = self.predict_tags(bo_text)
        logger.debug(f"Tokens:      {tokens['input_ids'][0].tolist()}")
        logger.debug(f"Predictions: {cur_preds}")

        labels = [self.id_to_label_map[pred] for pred in cur_preds]
        res = {
            "words": [],
            "tags": []
        }

        word_tokens, word_label = [], ""
        for token, label in zip(tokens['input_ids'][0], labels):
            if not label == '[MASK]':
                res["words"].append(self.tokenizer.decode(word_tokens))
                res["tags"].append(word_label)
                word_tokens, word_label = [], label
            word_tokens.append(token)
        if len(word_tokens) > 0:
            res["words"].append(self.tokenizer.decode(word_tokens))
            res["tags"].append(word_label)

        return res