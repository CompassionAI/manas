import os
import sys
import pickle
import logging
from cai_manas.tokenizer.tokenizer import CAITokenizer
from tqdm import tqdm

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from torch.utils.data import random_split
from sklearn.metrics import precision_recall_fscore_support

from cai_common.models.utils import get_local_ckpt
from cai_common.datasets import TokenTagDataset
from cai_common.utils.hydra_training_args import HydraTrainingArguments

import transformers
from transformers import (
    AutoConfig,
    AlbertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    set_seed
)
from ..tokenizer import CAITokenizerSlow

CAITokenizer = CAITokenizerSlow

logger = logging.getLogger(__name__)

cs = ConfigStore()
cs.store(group="training", name="huggingface_training_args", node=HydraTrainingArguments)


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
    precision, recall, fscore, _ = precision_recall_fscore_support(preds_list, out_label_list, average='weighted')
    return {
        "precision": precision,
        "recall": recall,
        "f1": fscore}


@hydra.main(version_base="1.2", config_path="./pos_fine_tuning.config", config_name="config")
def main(cfg):
    cfg.training.output_dir = HydraConfig.get().run.dir
    training_cfg = HydraTrainingArguments.as_hf_training_args(cfg.training)
    training_cfg.logging_dir = os.path.join(cfg.training.output_dir, "tb_logs")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_cfg.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.debug(f"Setting seed: {training_cfg.seed}")
    set_seed(training_cfg.seed)

    logger.info(f"Creating tokenizer: {cfg.model.tokenizer_name}")
    logger.debug(f"Tokenizer location: {CAITokenizer.get_local_model_dir(cfg.model.tokenizer_name)}")
    tokenizer = CAITokenizer.from_pretrained(CAITokenizer.get_local_model_dir(cfg.model.tokenizer_name))
    tokenizer.stochastic_tokenization = False
    tokenizer.tsheg_pretokenization = cfg.data.tsheg_pretokenization
    logger.debug(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    logger.info("Loading datasets")
    TokenTagDataset.concatenate_examples = cfg.data.concatenate_examples
    TokenTagDataset.use_mask_for_word_pieces = cfg.data.use_mask_for_word_pieces
    TokenTagDataset.dupe_count, TokenTagDataset.dupe_offset = cfg.data.dupe_count, cfg.data.dupe_offset
    if cfg.data.train_dataset_name is None:
        raise ValueError("Must pass in a training dataset name in --train_dataset_name")
    dataset = TokenTagDataset(
        tokenizer=tokenizer,
        processed_dataset=cfg.data.train_dataset_name,
        verbose=True,
        tqdm=tqdm)
    logger.debug("Loaded training dataset from disk")
    if cfg.data.test_dataset_name is None:
        if cfg.data.test_frac <= 0:
            logger.debug("No test set")
            train_data = dataset
            test_data = []
        else:
            logger.debug("Random split test set")
            data_len = len(dataset)
            test_len = int(cfg.data.test_frac * data_len)
            train_len = data_len - test_len
            train_data, test_data = random_split(dataset, [train_len, test_len])
    else:
        logger.debug("Withheld test data")
        train_data = dataset
        test_data = TokenTagDataset(
            tokenizer,
            processed_dataset=cfg.data.test_dataset_name,
            verbose=True,
            tqdm=tqdm)
    logger.debug(f"Training data size: {len(train_data)}, test data size: {len(test_data)}")

    logger.info(
        f"Loading model from checkpoint {cfg.model.tibert_pytorch_ckpt} with config name {cfg.model.config_name}")
    logger.debug(f"num_labels={len(dataset.label_to_id_map)}")
    albert_cfg = AutoConfig.from_pretrained(
        cfg.model.config_name,
        num_labels=len(dataset.label_to_id_map),
        id2label={id_: label for label, id_ in dataset.label_to_id_map.items()},
        label2id=dataset.label_to_id_map)
    if cfg.model.tibert_pytorch_ckpt is None:
        raise ValueError("Must pass in checkpoint name in --tibert_pytorch_ckpt")
    local_ckpt = get_local_ckpt(cfg.model.tibert_pytorch_ckpt)
    logger.debug(f"Local checkpoint file parsed to {local_ckpt}")
    tibert_mdl = AlbertForTokenClassification.from_pretrained(
        local_ckpt,
        config=albert_cfg)
    tibert_mdl.resize_token_embeddings(len(tokenizer))

    logger.info("Kicking off training!")
    trainer = Trainer(
        model=tibert_mdl,
        args=training_cfg,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    trainer.train()

    logger.info("Saving results")
    trainer.save_model()
    with open(os.path.join(training_cfg.output_dir, "train_dataset.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(training_cfg.output_dir, "test_dataset.pkl"), "wb") as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    main()      # pylint: disable=no-value-for-parameter
