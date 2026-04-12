from cs336_basics import train_bpe
import os
from collections.abc import Iterable, Iterator
import regex as re
import multiprocessing


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
        for line in iterable:
            ids = self.encode(line)
            yield from ids

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
                    for m in re.finditer(self.PAT, text_before):
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
                for m in re.finditer(self.PAT, text_after):
                    token_str = m.group()
                    token_bytes_tuple = tuple(bytes([b]) for b in token_str.encode("utf-8"))
                    seqs.append(token_bytes_tuple)

            return seqs
        else:
            for match in re.finditer(self.PAT, chunk):
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
