from cs336_basics import train_bpe
import os
from collections.abc import Iterable, Iterator
import regex as re
import multiprocessing
from tqdm import tqdm
import multiprocessing as mp
import numpy as np



class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        '''
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]

        3
        special_tokens: list[str] | None = None
        '''
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.vocab_reversed = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None, format='gpt2'):
        '''
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        '''
        if format=='gpt2':
            vocab = train_bpe.load_vocab_gpt2(vocab_filepath)
            merges = train_bpe.load_merges_gpt2(merges_filepath)

        if format=='local':
            vocab = train_bpe.load_vocab_json(vocab_filepath)
            merges = train_bpe.load_merges_json(merges_filepath)

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
    
    # def encode_parallel(self, batch_id: int, text: str) -> tuple[int, list[int]]:
    #     self.pre_tokenize_from_text(text)
    #     self.init_pair_2_index()
    #     self.merges_all()
    #     encode_seqs = []
    #     for tokens in self.seqs:
    #         for tk in tokens:
    #             encode_seqs.append(self.vocab_reversed[tk])

    #     return batch_id, encode_seqs

    def encode_iterable(self, iterable: Iterable[str], batch_size=5000) -> Iterator[int]:
        current_batch = ""
        for line in iterable:
            current_batch += line
            if len(current_batch) >= batch_size:
                ids = self.encode(current_batch)
                for id in ids:
                    yield id
                current_batch = ""
        if current_batch:
            ids = self.encode(current_batch)
            for id in ids:
                yield id

    def decode(self, ids: list[int]) -> str:
        decode_seqs = []
        buffer = b''
        for i in range(len(ids)):
            token_id = ids[i]
            buffer += self.vocab[token_id]
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



_global_tokenizer = None

def init_worker(vocab_path, merges_path, special_tokens, fmt):
    """
    每个 Worker 进程启动时运行一次，初始化 Tokenizer
    """
    global _global_tokenizer
    # 在每个子进程中重新加载 Tokenizer
    _global_tokenizer = Tokenizer.from_files(vocab_path, merges_path, 
                                             special_tokens=special_tokens, format=fmt)
    
def get_batches_generator(filepath, batch_size):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        batch_id = 0
        current_batch = ""
        line_counter = 0
        
        for line in f:
            current_batch += line
            line_counter += 1
            # 当批次攒够 batch_size 行，就作为一个任务发出去
            if line_counter >= batch_size:
                yield (batch_id, current_batch)
                batch_id += 1
                current_batch = ""
                line_counter = 0
        
        # 处理最后剩余不足 batch_size 的行
        if current_batch:
            yield (batch_id, current_batch)

def worker_encode_batch(args):
    batch_id, current_batch = args
    tokens = _global_tokenizer.encode(current_batch)
    return batch_id, tokens
    


if __name__ == '__main__':
    vocab_filepath = r'data\vocab_TinyStoriesV2.json'
    merges_filepath = r'data\merges_TinyStoriesV2.json'
    special_tokens = ['<|endoftext|>']
    format='local'
    tker = Tokenizer.from_files(vocab_filepath, merges_filepath, 
                                special_tokens=special_tokens, format=format)
    
    # tk_path = r'D:\Dev\assignment1-basics\data\tokens_TinyStoriesV2_train.npy'
    # tks = np.fromfile(tk_path, dtype=np.uint16)
    # print(tker.decode(tks[:5000]))


    max_tokens = 10_000_000_000
    txt_filepath = r'D:\Dev\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt'
    tokens_outpath = r'data/tokens_TinyStoriesV2_valid.npy'
    fp = np.memmap(tokens_outpath, dtype=np.uint16, mode='w+', shape=(max_tokens,))
    current_idx = 0
    batch_size = 10000
    num_workers = 4

    with mp.Pool(processes=num_workers, initializer=init_worker, 
                 initargs=(vocab_filepath, merges_filepath, special_tokens, format)) as pool:
        
        tasks = pool.imap(worker_encode_batch, get_batches_generator(txt_filepath, batch_size), chunksize=1)

        for batch_id, token_list in tqdm(tasks, desc="Encoding", unit="batch"):
            if token_list: # 确保列表不为空
                # 转为 numpy 数组以便快速写入
                arr = np.array(token_list, dtype=np.uint16)
                
                # 写入 memmap 指定位置
                fp[current_idx : current_idx + len(arr)] = arr
                current_idx += len(arr)

    fp.flush()   # 确保数据落盘
    del fp       # 释放原对象
    
    # 使用 os.truncate 真正地在磁盘上截断文件
    # 计算最终文件应有的字节大小
    final_size_in_bytes = (current_idx+1) * np.uint16().itemsize
    with open(tokens_outpath, 'r+b') as f:
        f.truncate(final_size_in_bytes)
    
    print(f"保存完成，总 Token 数: {current_idx}")
    print(f"文件已截断至: {final_size_in_bytes / (1024**3):.2f} GB")