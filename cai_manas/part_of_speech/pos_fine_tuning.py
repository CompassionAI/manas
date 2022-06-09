# Sample run:
# ===
# python -m cai_manas.part_of_speech.pos_fine_tuning \
#     --tibert_pytorch_ckpt tibert-albert/base.bin \
#     --output_dir ~/workspace/temp/pos-tagger \
#     --train_dataset_name part-of-speech-intrasyllabic-words \
#     --do_train \
#     --use_mask_for_word_pieces \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 20 \
#     --logging_steps 10 \
#     --save_steps 50 \
#     --save_total_limit 50 \
#     --log_level debug \
#     --evaluation_strategy steps

import os
import sys
import pickle
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import random_split
from sklearn.metrics import precision_recall_fscore_support

from ..tokenizer import TibertTokenizer
from ..models.utils import get_local_ckpt
from cai_common.datasets import TokenTagDataset

import transformers
from transformers import (
    AutoConfig,
    AlbertForTokenClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    tibert_pytorch_ckpt: str = field(
        metadata={"help": "Path to pretrained model checkpoint"})
    config_name: Optional[str] = field(
        default="albert-base-v2", metadata={"help": "Pretrained config name"})
    tokenizer_name: Optional[str] = field(
        default="olive-large",
        metadata={"help": "Pretrained tokenizer name"})


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to the datasets."""

    train_dataset_name: str = field(
        default=None,
        metadata={
            "help": "Name of the processed training dataset."})
    test_dataset_name: str = field(
        default=None,
        metadata={
            "help": "Name of the processed test dataset."})
    use_mask_for_word_pieces: bool = field(
        default=False,
        metadata={
            "help": "When set to False, the first piece of a word in the dataset is marked with its label and the "
                    "rest of the pieces of that word are marked with the padding token, which is set to the ignored "
                    "index of the cross-entropy loss by default. When set to True the [MASK] token is used for this "
                    "instead, which is not ignored by the loss, so that the model has to learn word segmentation and "
                    "part-of-speech tagging end-to-end."})
    dupe_count: int = field(
        default=0,
        metadata={
            "help": "Number of duplicate walks through words in the SOAS dataset to make training examples. Meant to "
                    "be used in combination with dupe_offset. The actual number of walks will be dupe_count + 1."})
    dupe_offset: int = field(
        default=3,
        metadata={
            "help": "Offset when duplicating the walking through words in the SOAS dataset to make training "
                    "examples."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."})
    test_frac: float = field(
        default=0.1,
        metadata={
            "help": "What fraction of the data to put into the test dataset."})


def align_predictions_1d(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_label_list, preds_list = [], []

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list.append(label_ids[i][j])
                preds_list.append(preds[i][j])

    return preds_list, out_label_list


def compute_metrics(p):
    preds_list, out_label_list = align_predictions_1d(p.predictions, p.label_ids)
    precision, recall, fscore, support = precision_recall_fscore_support(preds_list, out_label_list, average='weighted')
    return {
        "precision": precision,
        "recall": recall,
        "f1": fscore}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.debug(f"Setting seed: {training_args.seed}")
    set_seed(training_args.seed)

    logger.info(f"Creating tokenizer: {model_args.tokenizer_name}")
    logger.debug(f"Tokenizer location: {TibertTokenizer.get_local_model_dir(model_args.tokenizer_name)}")
    tibert_tkn = TibertTokenizer.from_pretrained(TibertTokenizer.get_local_model_dir(model_args.tokenizer_name))
    tibert_tkn.stochastic_tokenization = False
    tibert_tkn.tsheg_pretokenization = True
    logger.debug(f"Tokenizer vocabulary size: {tibert_tkn.vocab_size}")

    logger.info("Loading datasets")
    TokenTagDataset.use_mask_for_word_pieces = data_args.use_mask_for_word_pieces
    TokenTagDataset.dupe_count, TokenTagDataset.dupe_offset = data_args.dupe_count, data_args.dupe_offset
    if data_args.train_dataset_name is None:
        raise ValueError("Must pass in a training dataset name in --train_dataset_name")
    dataset = TokenTagDataset(
        tokenizer=tibert_tkn,
        processed_dataset=data_args.train_dataset_name,
        verbose=True,
        tqdm=tqdm)
    logger.debug("Loaded training dataset from disk")
    if data_args.test_dataset_name is None:
        if data_args.test_frac <= 0:
            logger.debug("No test set")
            train_data = dataset
            test_data = []
        else:
            logger.debug("Random split test set")
            data_len = len(dataset)
            test_len = int(data_args.test_frac * data_len)
            train_len = data_len - test_len
            train_data, test_data = random_split(dataset, [train_len, test_len])
    else:
        logger.debug("Withheld test data")
        train_data = dataset
        test_data = TokenTagDataset(
            tibert_tkn,
            processed_dataset=data_args.test_dataset_name,
            verbose=True,
            tqdm=tqdm)
    logger.debug(f"Training data size: {len(train_data)}, test data size: {len(test_data)}")

    logger.info(
        f"Loading model from checkpoint {model_args.tibert_pytorch_ckpt} with config name {model_args.config_name}")
    logger.debug(f"num_labels={len(dataset.label_to_id_map)}")
    albert_cfg = AutoConfig.from_pretrained(
        model_args.config_name,
        num_labels=len(dataset.label_to_id_map),
        id2label={id_: label for label, id_ in dataset.label_to_id_map.items()},
        label2id=dataset.label_to_id_map)
    if model_args.tibert_pytorch_ckpt is None:
        raise ValueError("Must pass in checkpoint name in --tibert_pytorch_ckpt")
    local_ckpt = get_local_ckpt(model_args.tibert_pytorch_ckpt)
    logger.debug(f"Local checkpoint file parsed to {local_ckpt}")
    tibert_mdl = AlbertForTokenClassification.from_pretrained(
        local_ckpt,
        config=albert_cfg)
    tibert_mdl.resize_token_embeddings(len(tibert_tkn))

    logger.info("Kicking off training!")
    trainer = Trainer(
        model=tibert_mdl,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics)
    trainer.train()

    logger.info("Saving results")
    trainer.save_model()
    with open(os.path.join(training_args.output_dir, "train_dataset.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(training_args.output_dir, "test_dataset.pkl"), "wb") as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    main()
