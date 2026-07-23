from typing import Optional
from .base import ITokenizer, TokenizerConfig, TokenizerType
from .bpe import BPETokenizer
from .unigram import UnigramTokenizer
from .wordpiece import WordPieceTokenizer
from .sentencepiece import SentencePieceTokenizer
from .dynamic import DynamicVocabularyTokenizer


class TokenizerFactory:
    @staticmethod
    def create(config: Optional[TokenizerConfig] = None) -> ITokenizer:
        if config is None:
            config = TokenizerConfig()

        if config.type == TokenizerType.BPE:
            inner: ITokenizer = BPETokenizer(config)
        elif config.type == TokenizerType.Unigram:
            inner = UnigramTokenizer(config)
        elif config.type == TokenizerType.WordPiece:
            inner = WordPieceTokenizer(config)
        elif config.type in (TokenizerType.SentencePieceBPE, TokenizerType.SentencePieceUnigram):
            inner = SentencePieceTokenizer(config)
        elif config.type == TokenizerType.DynamicVocabulary:
            core_config = TokenizerConfig(
                type=TokenizerType.BPE,
                vocab_size=config.vocab_size // 2,
                add_special_tokens=config.add_special_tokens,
                pad_token=config.pad_token,
                unk_token=config.unk_token,
                bos_token=config.bos_token,
                eos_token=config.eos_token,
            )
            inner = DynamicVocabularyTokenizer(BPETokenizer(core_config), config)
        else:
            inner = BPETokenizer(config)

        return inner