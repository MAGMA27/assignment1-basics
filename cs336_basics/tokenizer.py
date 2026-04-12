from cs336_basics import train_bpe
import os
from collections.abc import Iterable, Iterator
import regex as re
import multiprocessing
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
import json


class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        '''
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        '''
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.vocab_reversed = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        '''
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        '''
        vocab = train_bpe.load_vocab_gpt2(vocab_filepath)
        merges = train_bpe.load_merges_gpt2(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        self.pre_tokenize_from_text(text)
        self.init_pair_2_index()
        self.merges_all()
        encode_seqs = []
        for tokens in self.seqs:
            for tk in tokens:
                encode_seqs.append(self.vocab_reversed[tk])

        return encode_seqs

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for line in iterable:
            text_to_process = buffer + line
            text_to_process = text_to_process.replace('\r\n', '\n')

            matches = list(re.finditer(self.PAT, text_to_process))

            if not matches:
                buffer = ""
                continue

            last_match = matches[-1]

            if last_match.end() == len(text_to_process):
                for m in matches[:-1]:
                    yield from self._encode_match(m)
                buffer = last_match.group()
            else:
                for m in matches:
                    yield from self._encode_match(m)
                buffer = ""

        if buffer:
            for m in re.finditer(self.PAT, buffer):
                yield from self._encode_match(m)

    def _encode_match(self, m):
        matched_text = m.group()

        if self.special_tokens and matched_text in self.special_tokens:
            yield self.vocab_reversed[matched_text.encode('utf-8')]
            return

        word = tuple(bytes([b]) for b in matched_text.encode("utf-8"))
        for pair in self.merges:
            i = 0
            new_tup = []
            while i < len(word):
                if i < len(word)-1 and pair[0]==word[i] and pair[1]==word[i+1]:
                    new_tup.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tup.append(word[i])
                    i += 1
            word = tuple(new_tup)

        for bt in word:
            yield self.vocab_reversed[bt]

    def decode(self, ids: list[int]) -> str:
        decode_seqs = []
        buffer = b''
        for i in range(len(ids)):
            buffer += self.vocab[ids[i]]
            try:
                decode_seqs.append(buffer.decode('utf_8'))
                buffer = b''
            except:
                continue

        return "".join(decode_seqs)

    def pre_tokenize_chunk(self, chunk):
        chunk = chunk.replace('\r\n', '\n')

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        seqs = []

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
            special_pattern = "|".join(escaped_tokens)

            # 先用 special pattern 分割文本，然后对非 special token 部分用 PAT 处理
            seqs = []
            last_end = 0
            for match in re.finditer(special_pattern, chunk):
                # 处理 special token 之前的普通文本
                if match.start() > last_end:
                    text_before = chunk[last_end:match.start()]
                    for m in re.finditer(PAT, text_before):
                        token_str = m.group()
                        token_bytes_tuple = tuple(bytes([b]) for b in token_str.encode("utf-8"))
                        seqs.append(token_bytes_tuple)

                # 添加 special token
                token_str = match.group()
                token_bytes = token_str.encode("utf-8")
                token_bytes_tuple = (token_bytes,)
                seqs.append(token_bytes_tuple)

                last_end = match.end()

            # 处理最后一个 special token 之后的普通文本
            if last_end < len(chunk):
                text_after = chunk[last_end:]
                for m in re.finditer(PAT, text_after):
                    token_str = m.group()
                    token_bytes_tuple = tuple(bytes([b]) for b in token_str.encode("utf-8"))
                    seqs.append(token_bytes_tuple)
                    
            return seqs
        else:
            for match in re.finditer(PAT, chunk):
                token_str = match.group()
                token_bytes_tuple = tuple(bytes([b]) for b in token_str.encode("utf-8"))
                seqs.append(token_bytes_tuple)

        return seqs

    def process_chunk(self, args):
        input_path, start, end = args
        with open(input_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            seqs_part = self.pre_tokenize_chunk(chunk)

        return seqs_part

    def pre_tokenize_from_file(self, input_path: str):
        file_size = os.path.getsize(input_path)

        if file_size < 1024 * 1024:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                chunk = f.read()
                seqs = self.pre_tokenize_chunk(chunk)
        else:
            with open(input_path, "rb") as f:
                num_processes = 4
                boundaries = train_bpe.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
                tasks = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
                with multiprocessing.Pool(processes=num_processes) as pool:
                    seqs_all = pool.map(self.process_chunk, tasks)

                seqs = [item for sublist in seqs_all for item in sublist]

        self.seqs = seqs

    def pre_tokenize_from_text(self, chunk: str):
        seqs = self.pre_tokenize_chunk(chunk)
        self.seqs = seqs

    def init_pair_2_index(self):
        pair_to_index = {}
        for i, tup in enumerate(self.seqs):
            for j in range(len(tup) - 1):
                pair = (tup[j], tup[j+1])
                if pair in pair_to_index:
                    pair_to_index[pair].append(i)
                else:
                    pair_to_index[pair] = [i]

        self.pair_to_index = pair_to_index

    def merge_pairs(self, pair: tuple, tup: tuple) -> tuple[tuple, list[tuple]]:
        i = 0
        new_tup = []
        new_pairs = []
        while i < len(tup):
            if i < len(tup)-1 and pair[0]==tup[i] and pair[1]==tup[i+1]:
                new_tup.append(pair[0] + pair[1])
                if i > 0:
                    new_pairs.append((tup[i-1], pair[0] + pair[1]))
                i += 2
                if i < len(tup):
                    new_pairs.append((pair[0] + pair[1], tup[i]))
            else:
                new_tup.append(tup[i])
                i += 1
        return tuple(new_tup), new_pairs

    def merges_all(self):
        for pair in self.merges:
            if pair not in self.pair_to_index:
                continue
            for i in self.pair_to_index[pair]:
                new_tup, new_pairs = self.merge_pairs(pair, self.seqs[i])
                self.seqs[i] = new_tup
                for new_pair in new_pairs:
                    if new_pair in self.pair_to_index:
                        self.pair_to_index[new_pair].append(i)
                    else:
                        self.pair_to_index[new_pair] = [i]

            del self.pair_to_index[pair]


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path, encoding="utf-8") as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return Tokenizer(vocab, merges, special_tokens)


if __name__ == '__main__':
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=r'D:\Dev\assignment1-basics\tests\fixtures\gpt2_vocab.json', 
        merges_path=r'D:\Dev\assignment1-basics\tests\fixtures\gpt2_merges.txt', 
        special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # print(tokenizer.seqs)
    assert tokenized_string.count("<|endoftext|>") == 3
