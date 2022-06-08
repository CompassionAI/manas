import sys
import logging
import argparse

from .pos_tagger import PartOfSpeechTagger


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    logger = logging.getLogger(__package__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='tibert-bpe-large',
        help="Tokenizer name")
    parser.add_argument(
        "--model_cfg_name",
        type=str,
        default='albert-base-v2',
        help="Huggingface model config name")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        help="Fine-tuned model weights name in the data registry")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug logs")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    tagger = PartOfSpeechTagger(args.tokenizer_name, args.model_cfg_name, args.model_ckpt)

    print("Interactive Tibetan part-of-speech tagging...")
    while True:
        print("===")
        bo_text = input("Tibetan (or type exit): ")
        if bo_text == "exit":
            break
        res = tagger.tag(bo_text)
        for word, tag in zip(res["words"], res["tags"]):
            print(word, tag)
