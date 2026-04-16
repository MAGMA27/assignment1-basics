import os
from typing import BinaryIO
import regex as re
import multiprocessing
import json

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_initial_stats(vocab):
    """统计所有相邻字节对的频率"""
    pairs = {}
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs

def update_pair_counts_incrementally(pair_counts, word_tuple, new_word_tuple, freq_delta):
    """增量更新 pairs"""
    # 扫描当前单词，找到所有涉及 old_pair 的位置，并计算变化
    word_pair_counts = {}
    # decrease
    i = 0
    while i < len(word_tuple) - 1:
        pair = (word_tuple[i], word_tuple[i+1])
        word_pair_counts[pair] = word_pair_counts.get(pair, 0) - freq_delta
        i += 1
    # increase
    i = 0
    while i < len(new_word_tuple) - 1:
        pair = (new_word_tuple[i], new_word_tuple[i+1])
        word_pair_counts[pair] = word_pair_counts.get(pair, 0) + freq_delta
        i += 1
    # update
    for tup, freq in word_pair_counts.items():
        pair_counts[tup] = pair_counts.get(tup, 0) + freq
        if pair_counts[tup] <= 0:
            pair_counts.pop(tup, None)

def build_vocab_from_merges(merges, special_tokens=[]):
    '''从merges中恢复出BPE词表'''
    vocab = {}

    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode('utf-8')

    next_id = len(special_tokens)

    for i in range(256):
        vocab[i + next_id] = bytes([i])

    next_id = i + next_id + 1
    
    for pair in merges:
        p0, p1 = pair

        new_token_bytes = p0 + p1
        vocab[next_id] = new_token_bytes
        
        next_id += 1
    
    return vocab

def save_tokenizer_json(vocab, merges, vocab_path="bpe_vocab.json", merges_path="bpe_merges.json"):
    '''将结果保存下来'''
    vocab_str = {str(k): v.decode('latin-1') for k, v in vocab.items()}
    merges_str = []
    for p1, p2 in merges:
        merges_str.append([p1.decode('latin-1'), p2.decode('latin-1')])
    
    vocab_data = {
        "vocab": vocab_str,
    }
    merges_data = {
        "merges": merges_str,
    }
    
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_data, f, indent=2, ensure_ascii=False)
        
def load_vocab_json(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 还原 Vocab
    vocab = {int(k): v.encode('latin-1') for k, v in data['vocab'].items()}
    
    return vocab

def load_vocab_gpt2(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
        # 还原 Vocab
        vocab = {int(v): k.encode('utf-8') for k, v in data.items()}
    return vocab

def load_merges_json(merges_path):
    with open(merges_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # print(data)
    merges = []
    for pair in data['merges']:
        p1_str, p2_str = pair[0], pair[1]
        merges.append((p1_str.encode('latin-1'), p2_str.encode('latin-1')))
    
    return merges

def load_merges_gpt2(merges_path):
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split(' ')
            byte_tuple = tuple(part.encode('utf-8') for part in parts)
            merges.append(byte_tuple)

    return merges

def pretokenize_and_count(chunk, special_tokens):
    """对单个文本片段进行预分词并统计"""
    counts = {}

    chunk = chunk.replace('\r\n', '\n') # 处理windows与linux文本换行符的差异

    if special_tokens:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        split_pattern = "|".join(escaped_tokens)
        if split_pattern:
             segments = re.split(split_pattern, chunk)
        else:
             segments = [chunk]
    else:
        segments = [chunk]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for seq in segments:
        if not seq:
            continue

        for match in re.finditer(PAT, seq):
            token_str = match.group()
            token_bytes_tuple = tuple(bytes([b]) for b in token_str.encode("utf-8"))
            counts[token_bytes_tuple] = counts.get(token_bytes_tuple, 0) + 1

    return counts

def process_chunk(args):
    '''包装对chunk的处理，方便并行'''
    input_path, start, end, special_tokens = args
    counts = {}
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        counts = pretokenize_and_count(chunk, special_tokens)
    return counts

def to_run_train_bpe(
        input_path: str, 
        vocab_size: int, 
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges = []
    '''主训练函数'''

    # 获取文件大小来决定是否使用多进程
    file_size = os.path.getsize(input_path)

    if file_size < 1024 * 1024:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            chunk = f.read()
            total_counts = pretokenize_and_count(chunk, special_tokens)
    else:
        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(process_chunk, tasks)
                
            total_counts = {}
        
            for partial_counts in results:
                for token, count in partial_counts.items():
                    if token in total_counts:
                        total_counts[token] += count
                    else:
                        total_counts[token] = count

    # 统计所有 pair 的频率
    pairs = get_initial_stats(total_counts)

    # BPE 合并循环
    while len(special_tokens) + 256 + len(merges) < vocab_size:
        if not pairs:
            break

        # 找到频率最高的 pair，频率相同时选择字典序最大的
        max_freq = max(pairs.values())
        best_pair = max(pair for pair, freq in pairs.items() if freq == max_freq)

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        new_total_counts = {} 
        # 增量更新pairs字典，统计相邻token出现的次数，同时更新total_counts
        # 想要更快的话，还要维护pairs: word_tuple 字典，用best_pair查找要更新的word_tuple
        for word_tuple, freq in total_counts.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                # 检查当前位置是否是待合并的 pair
                if i < len(word_tuple) - 1 and word_tuple[i] == best_pair[0] and word_tuple[i+1] == best_pair[1]:
                    # 合并这两个字节
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_word_tuple = tuple(new_word)
            new_total_counts[new_word_tuple] = freq
            if new_word_tuple != word_tuple:
                update_pair_counts_incrementally(pairs, word_tuple, new_word_tuple, freq)
        total_counts = new_total_counts

    vocab = build_vocab_from_merges(merges, special_tokens)
    return vocab, merges


if __name__ == '__main__':
    input_path = r'D:\Dev\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_tokens = [r'<|endoftext|>']

    vocab, merges = to_run_train_bpe(input_path, vocab_size, special_tokens)
    save_tokenizer_json(vocab, merges, 
                        vocab_path="data/vocab_TinyStoriesV2.json", 
                        merges_path="data/merges_TinyStoriesV2.json")
