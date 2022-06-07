import os
import glob
import unicodedata
from transformers import AlbertTokenizer
import sentencepiece as spm

DATA_BASE_PATH = os.environ['CAI_DATA_BASE_PATH']

VOCAB_FILES_NAMES = {"vocab_file": "spm_model.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "tibert-bpe-tiny":
            os.path.join(DATA_BASE_PATH, "champion_models/spm-tokenizers/tibert_spm_bpe_tiny/spm_model.model"),
        "tibert-bpe-small":
            os.path.join(DATA_BASE_PATH, "champion_models/spm-tokenizers/tibert_spm_bpe_small/spm_model.model"),
        "tibert-bpe-large":
            os.path.join(DATA_BASE_PATH, "champion_models/spm-tokenizers/tibert_spm_bpe_big/spm_model.model"),
        "tibert-unigram-large":
            os.path.join(DATA_BASE_PATH, "champion_models/spm-tokenizers/tibert_spm_unigram_big/spm_model.model")
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "tibert-bpe-tiny": 512,
    "tibert-bpe-small": 512,
    "tibert-bpe-large": 512,
    "tibert-unigram-large": 512
}

SPIECE_UNDERLINE = "▁"


class TibertTokenizer(AlbertTokenizer):
    """Constructs the Tibert tokenizer. Very similar to the ALBERT tokenizer. Based on `SentencePiece
    <https://github.com/google/sentencepiece>`__

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to lowercase the input when tokenizing.
        remove_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to strip the text when tokenizing (removing excess spaces before and after the string).
        keep_accents (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to keep accents when tokenizing.
        bos_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.
            .. note::
                When building a sequence using special tokens, this is not the token that is used for the beginning
                of sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The end of sequence token.
            .. note::
                When building a sequence using special tokens, this is not the token that is used for the end
                of sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`string`, `optional`, defaults to "<unk>"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`string`, `optional`, defaults to "[MASK]"):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        stochastic_tokenization (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to perform stochastic tokenization in SentencePiece or use the Viterbi tokens.
        stochastic_encoding_nbest (:obj:`int`, `optional`, defaults to :obj:`True`):
            The n-best paramater for the stochastic encoding in SentencePiece.
        stochastic_encoding_alpha  (:obj:`float`, `optional`, defaults to :obj:`True`):
            The alpha paramater for the stochastic encoding in SentencePiece.
        tsheg_pretokenization (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Pre-tokenize along the intersyllabic marks, primarily the tsheg. Useful for token classification tasks like
            named-entity recognition and part-of-speech tagging.
        underline_between_sections (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to put the SentencePiece underline only between sections. Note that, if there is only one section,
            there will be no underline. If you're getting bad results for the unmasking of the first token in a section
            then try setting this to True.
    Attributes:
        sp_model (:obj:`SentencePieceProcessor`):
            The `SentencePiece` processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    intersyllabic_marks = [unicodedata.lookup(c) for c in [
        'TIBETAN MARK INITIAL YIG MGO MDUN MA',
        'TIBETAN MARK CLOSING YIG MGO SGAB MA',
        'TIBETAN MARK SBRUL SHAD',
        'TIBETAN MARK INTERSYLLABIC TSHEG',
        'TIBETAN MARK DELIMITER TSHEG BSTAR',
        'TIBETAN MARK SHAD',
        'TIBETAN MARK RIN CHEN SPUNGS SHAD',
        'TIBETAN MARK GTER TSHEG',
        'TIBETAN SIGN RDEL DKAR GSUM',
        'TIBETAN KU RU KHA']]

    _print_tokenizer = False
    stochastic_encoding_nbest = 64
    stochastic_encoding_alpha = 0.1
    stochastic_tokenization = False
    tsheg_pretokenization = False
    underline_between_sections = False

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 remove_space=True,
                 keep_accents=False,
                 bos_token='[CLS]',
                 eos_token='[SEP]',
                 unk_token='<unk>',
                 sep_token='[SEP]',
                 pad_token='<pad>',
                 cls_token='[CLS]',
                 mask_token='[MASK]',
                 **kwargs):
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs)
        self.keep_accents = True

    @staticmethod
    def get_local_model_dir(model_name):
        """Get the model local directory name in the CAI data registry from the Transformers model name.

        Args:
            model_name (:obj:`string`):
                The Transformers model name.
        
        Returns:
            The local directory name you can feed to TibertTokenizer.from_pretrained.
        """

        if model_name not in PRETRAINED_VOCAB_FILES_MAP['vocab_file']:
            valid_names = ', '.join(PRETRAINED_VOCAB_FILES_MAP['vocab_file'].keys())
            raise KeyError(f"Unknown TibertTokenizer model name {model_name}. Valid names are: {valid_names}")
        return os.path.dirname(PRETRAINED_VOCAB_FILES_MAP['vocab_file'][model_name])

    @staticmethod
    def train(training_data_glob,
              model_file,
              model_type='bpe',
              vocab_size=10000):
        """Train the SentencePiece tokenizer and save to a file.

        Args:
            training_data_glob (:obj:`string`):
                A glob for the training data to use to train SentencePiece. For example:
                '../../tibert_data/training/tibetan_sections/spm_train.*.txt'
            model_file (:obj:`string`):
                `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
                will contain the vocabulary necessary to instantiate a tokenizer.
            model_type (:obj:`string`, `optional`, defaults to :obj:`bpe`):
                What SentencePiece model type to train. Can be 'bpe' or 'unigram'.
            vocab_sze (:obj:`int`, `optional`, defaults to :obj:`10000`):
                SentencePiece target vocabulary size.
        """

        # Settings are from https://github.com/google-research/albert/blob/master/README.md
        spm.SentencePieceTrainer.train(
            input=glob.glob(training_data_glob),
            model_prefix=model_file,
            model_type=model_type,
            vocab_size=vocab_size,
            character_coverage=1,
            pad_id=0,
            unk_id=1,
            eos_id=-1,
            bos_id=-1,
            control_symbols="[CLS],[SEP],[MASK]",
            user_defined_symbols="(,),\",-,.,–,£,€")

    def _tokenize(self, text):
        # A copy of the _tokenize function from the original but with nbest and alpha exposed.
        if self._print_tokenizer:
            print("Tokenizing {} with sample={}".format(text, self.stochastic_tokenization))
        text = self.preprocess_text(text)

        if self.stochastic_tokenization:
            pieces = self.sp_model.SampleEncodeAsPieces(
                text,
                self.stochastic_encoding_nbest,
                self.stochastic_encoding_alpha)
        else:
            pieces = self.sp_model.EncodeAsPieces(text)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def tokenize(self, text, **kwargs):
        # The AlBERT tokenizer deliberately splits on the control tokens in the text to introduce white space between
        #   them. So [MASK]de dmar ro/ becomes ['[MASK]', '_de', 'dmar', 'ra', 'o', '/']. This doesn't work for
        #   Tibetan, we want it to be ['_', '[MASK]', 'de', 'dmar', 'ra', 'o', '/'] otherwise we have a huge mismatch
        #   between the training and test. So we replace the control tokens with non-Tibetan special text, tokenize as
        #   per AlBERT, then replace the special text with the control tokens back. Everything then flows correctly
        #   through the rest of the AlBERT tokenizer.
        #
        # We also introduce the option to only add the underline between sections. This can improve performance on
        #   unmasking short sentences. Likely this is happening because of a mismatch between the training and test
        #   data, which means the short sentence part of the BERT pre-training data generator is actually important.
        #   In Tibetan there is no obvious way to do this without a sentence segmentation model, so we leave this as is
        #   until we are able to make a reliable sentence segmenter as part of the part-of-speech tagging experiments.
        #   When we have a reliable segmenter we can augment the training set with short individual sentences that have
        #   the proper SentencePiece underscores in front of them and polish the transformer model.
        if ' ' in text and self.underline_between_sections:
            sections = [self.tokenize(section, **kwargs) for section in text.split(' ')]
            res = []
            for section in sections + [[]]:
                if len(section) > 0:
                    res.extend(section + [SPIECE_UNDERLINE])
            if len(res) > 0:
                res = res[:-1]  # Remove the final SentencePiece underline
            return res
        if self.tsheg_pretokenization:
            for mark in self.intersyllabic_marks:
                # Break up the intersyllabic marks with a non-Tibetan token. This will force SentencePiece's BPE to
                #   not join across the out-of-vocabulary token.
                text = text.replace(mark, mark + 'a')
        special_token_map = dict(
            [(str(idx), control_token) for idx, control_token in enumerate(list(self.unique_no_split_tokens))])
        for replacement, control_token in special_token_map.items():
            text = text.replace(control_token, replacement)
        res = [
            special_token_map[token] if token in special_token_map else token
            for token in super().tokenize(text, **kwargs)]
        if self.underline_between_sections:
            final_res = []
            for token in res:
                if token == SPIECE_UNDERLINE:
                    continue
                if token[0] == SPIECE_UNDERLINE:
                    token = token[1:]
                final_res.append(token)
        else:
            final_res = res
        if self.tsheg_pretokenization:
            # Now remove the non-Tibetan tokens to get a strictly intersyllabic tokenization.
            final_res = [token for token in final_res if not token == 'a']
        return final_res


if __name__ == "__main__":
    training_data = os.path.join(DATA_BASE_PATH, "processed_datasets/spm-tokenizer-training/spm_train.txt")
    spm_tokenizers = os.path.join(DATA_BASE_PATH, "champion_models/spm-tokenizers")

    TibertTokenizer.train(
        training_data,
        os.path.join(spm_tokenizers, 'tibert_spm_bpe_tiny'),
        model_type='bpe',
        vocab_size=1000)

    TibertTokenizer.train(
        training_data,
        os.path.join(spm_tokenizers, 'tibert_spm_bpe_small'),
        model_type='bpe',
        vocab_size=5000)

    TibertTokenizer.train(
        training_data,
        os.path.join(spm_tokenizers, 'tibert_spm_bpe_big'),
        model_type='bpe',
        vocab_size=10000)