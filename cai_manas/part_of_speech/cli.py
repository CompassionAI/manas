import sys
import logging
import hydra

from .pos_tagger import PartOfSpeechTagger


@hydra.main(version_base="1.2", config_path="./cli.config", config_name="config")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    tagger = PartOfSpeechTagger(
        cfg.model.tokenizer_name,
        cfg.model.model_cfg_name,
        cfg.model.model_ckpt)

    print("Interactive Tibetan part-of-speech tagging...")
    while True:
        print("===")
        bo_text = input("Tibetan (or type exit): ")
        if bo_text == "exit":
            break
        res = tagger.tag(bo_text)
        for word, tag in zip(res["words"], res["tags"]):
            print(word, tag)


if __name__ == "__main__":
    main()
