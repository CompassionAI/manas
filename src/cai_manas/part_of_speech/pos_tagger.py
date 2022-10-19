import os
import json
import logging
from typing import Tuple, Dict, List

import torch
import numpy as np
from cai_common.models.utils import get_local_ckpt, get_cai_config
from transformers import (
    AutoConfig,
    AlbertForTokenClassification)

from ..tokenizer import CAITokenizer

logger = logging.getLogger(__name__)

class PartOfSpeechTagger:
    """A part-of-speech tagging utility class. It abstracts the PoS pipeline. See the cai_manas.part_of_speech.cli
    module for usage examples.

    Attributes:
        tokenizer: The loaded and configured tokenizer object.
        id_to_label_map: Dictionary mapping token tag ids to text labels, for example 34 -> [MASK]. Extracted from the
            config.json of the model checkpoint.
        model_cfg: Huggingface config for the fine-tuned model checkpoint.
        model: Huggingface fine-tuned model, set to eval mode.
    """

    def __init__(self, model_ckpt: str) -> None:
        """Loads all the relevant data and models for part-of-speech tagging.

        Args:
            model_ckpt: Name of the fine-tuned model checkpoint in the data registry to use for part-of-speech tagging.
                For example, part-of-speech-intrasyllabic-tags.
        """

        local_ckpt = get_local_ckpt(model_ckpt)
        logger.debug(f"Local model checkpoint {model_ckpt} resolved to {local_ckpt}")

        logger.debug("Loading CAI PoS model config")
        cai_pos_config = get_cai_config(model_ckpt)
        base_model = cai_pos_config['base_model']
        logger.debug(f"Base model resolved to {base_model}")

        logger.debug("Loading CAI base model config")
        cai_base_config = get_cai_config(base_model)
        tokenizer_name = cai_base_config['tokenizer_name']
        config_name = cai_base_config['hf_base_model_name']

        logger.debug(f"Loading tokenizer {tokenizer_name}")
        self.tokenizer = CAITokenizer.from_pretrained(CAITokenizer.get_local_model_dir(tokenizer_name))
        self.tokenizer.stochastic_tokenization = False
        self.tokenizer.tsheg_pretokenization = True

        logger.debug("Loading model config.json")
        config_json_fn = os.path.join(os.path.dirname(local_ckpt), "config.json")
        with open(config_json_fn, 'r') as f:
            config_json = json.load(f)
        logger.debug("Extracting label2id maps")
        self.id_to_label_map = {
            int(id): label
            for id, label in config_json["id2label"].items()}

        logger.debug("Loading Huggingface model config")
        self.model_cfg = AutoConfig.from_pretrained(
            config_name,
            vocab_size=self.tokenizer.vocab_size,
            num_labels=len(self.id_to_label_map),
            id2label=self.id_to_label_map)

        logger.debug("Loading model")
        self.model = AlbertForTokenClassification.from_pretrained(local_ckpt, config=self.model_cfg)
        logger.debug("Configuring model")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def predict_tokens(self, bo_tokens: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        # pylint: disable=no-member
        """Run the prediction of the part-of-speech tags on the list of tokens. Returns the tag IDs.

        Args:
            bo_tokens: List of Tibetan tokens to tag. *NOT* a PyTorch tensor.

        Returns:
            Predicted tag IDs as a numpy array.
        """

        if not bo_tokens[0] == self.tokenizer.bos_token_id:
            bo_tokens = [self.tokenizer.bos_token_id] + bo_tokens
        if not bo_tokens[-1] == self.tokenizer.eos_token_id:
            bo_tokens = bo_tokens + [self.tokenizer.eos_token_id]
        bo_tokens = torch.LongTensor([bo_tokens])
        tokens = {
            'input_ids': bo_tokens,
            'attention_mask': torch.ones_like(bo_tokens)
        }
        mdl_res = self.model(**tokens)[0][0]
        return np.argmax(mdl_res.detach().numpy(), axis=1)

    def predict_tags(self, bo_text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Run the core prediction of the part-of-speech tags. Returns the numerical tokens and tag IDs.

        Args:
            bo_text: The Tibetan text to tag, as a unicode string.

        Returns:
            A tuple of (tokens, predicted tag IDs) where both elements are numpy arrays.
        """

        tokens = self.tokenizer(bo_text, return_tensors='pt')
        mdl_res = self.model(**tokens)[0][0]
        return tokens['input_ids'][0].numpy(), np.argmax(mdl_res.detach().numpy(), axis=1)

    def tag(self, bo_text: str) -> Dict[str, List[str]]:
        """Segment and tag the passed in Tibetan text. Note that this function includes the word segmentation and does
            not return [MASK] tokens.

        Args:
            bo_text: The Tibetan text to tag, as a unicode string.

        Returns:
            A dictionary with two keys: words and tags. The words are a list of the segmented decoded Tibetan words, in
            unicode text. The tags are the predicted tags for each word.
        """

        tokens, cur_preds = self.predict_tags(bo_text)
        logger.debug(f"Tokens:      {tokens}")
        logger.debug(f"Predictions: {cur_preds}")

        labels = [self.id_to_label_map[pred] for pred in cur_preds]
        res = {
            "words": [],
            "tags": []
        }

        word_tokens, word_label = [], ""
        for token, label in zip(tokens, labels):
            if not label == '[MASK]':
                res["words"].append(self.tokenizer.decode(word_tokens))
                res["tags"].append(word_label)
                word_tokens, word_label = [], label
            word_tokens.append(token)
        if len(word_tokens) > 0:
            res["words"].append(self.tokenizer.decode(word_tokens))
            res["tags"].append(word_label)

        return res
