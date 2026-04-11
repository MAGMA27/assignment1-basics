

class tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        '''
        vocab: dict[int, bytes]  
        merges: list[tuple[bytes, bytes]]  
        special_tokens: list[str] | None = None 
        '''


    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        '''
        vocab_filepath: str  
        merges_filepath: str  
        special_tokens: list[str] | None = None
        '''


    def encode(self, text: str) -> list[int]:
        '''
        '''

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        '''

    def decode(self, ids: list[int]) -> str:
        '''
        '''
        