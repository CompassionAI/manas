import os

from cai_common.models.utils import get_local_model_dir

class CAITokenizerMixin:
    @classmethod
    def get_local_model_dir(cls, model_name, download_if_missing=True):
        """Get the model local directory name in the CAI data registry from the Transformers model name.

        Args:
            model_name (:obj:`string`):
                The Transformers model name.
            download_if_missing (:obj:`bool`, `optional`):
                Download from the CompassionAI S3 repository if missing the local repository. This is expected in
                    inference installations. Defaults to True.

        Returns:
            The local directory name you can feed to AutoTokenizer.from_pretrained.
        """

        if model_name not in cls.pretrained_vocab_files_map['vocab_file']:
            valid_names = ', '.join(cls.pretrained_vocab_files_map['vocab_file'].keys())
            raise KeyError(f"Unknown tokenizer model name {model_name}. Valid names are: {valid_names}")
        dir_name = os.path.dirname(cls.pretrained_vocab_files_map['vocab_file'][model_name])
        dir_name = get_local_model_dir(dir_name)
        return dir_name
