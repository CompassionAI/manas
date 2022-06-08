import os

class CAITokenizerMixin:
    @classmethod
    def get_local_model_dir(cls, model_name):
        """Get the model local directory name in the CAI data registry from the Transformers model name.

        Args:
            model_name (:obj:`string`):
                The Transformers model name.
        
        Returns:
            The local directory name you can feed to TibertTokenizer.from_pretrained.
        """

        if model_name not in cls.pretrained_vocab_files_map['vocab_file']:
            valid_names = ', '.join(cls.pretrained_vocab_files_map['vocab_file'].keys())
            raise KeyError(f"Unknown tokenizer model name {model_name}. Valid names are: {valid_names}")
        return os.path.dirname(cls.pretrained_vocab_files_map['vocab_file'][model_name])
