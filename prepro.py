import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open

'''
这是个对数据预处理的文件，并把数据保存为tfrecord文件和json文件
'''
# 创建一个空的模型
nlp = spacy.blank("en")

# 分词
def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    """返回分词后每个单词的字母起始和结束位置
    :param text: 一个样本
    :param tokens: 分词后的样本
    :return: shape = [len(tokens), 1, 1]
    """
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

#
def process_file(filename, data_type, word_counter, char_counter):
    """
    :param filename: 文件名
    :param data_type: 数据类型
    :param word_counter: 单词计数器
    :param char_counter: 字母计数器
    :return:
    """
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                # 统一context中的引号
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                # context分词获得单词列表
                context_tokens = word_tokenize(context)
                # 将单词列表转成context的字母矩阵
                context_chars = [list(token) for token in context_tokens]
                # 返回每个单词的首字母位置和末字母位置
                spans = convert_idx(context, context_tokens)

                for token in context_tokens:
                    # 统计每个单词与几个问题相关
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        # 统计每个字母与几个问题相关
                        char_counter[char] += len(para["qas"])
                # 对于每个问题
                for qa in para["qas"]:
                    # 记录问题数
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    # question分词获得单词列表
                    ques_tokens = word_tokenize(ques)
                    # 将单词列表转成question的字母矩阵
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    # 记录答案字符串的列表
                    answer_texts = []
                    # 一个question的多个answer拼接成一个向量
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        # 答案首字母位置
                        answer_start = answer['answer_start']
                        # 答案末字母位置
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        # 把span中落在区间[answer_start, answer_end]的单词索引取出来
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        # 对应某个问题所有答案的单词起始位置和单词结束位置
                        y1s.append(y1)
                        y2s.append(y2)
                    # 包含分词后的context单词列表和字母矩阵，分词后的question单词列表和字母矩阵，答案单词的首末位置
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    # 保存格式{"context":context, "span":每个单词的首末位置, "answer":答案列表, "uuid":qa["id"]}
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    """从词嵌入或字符嵌入文件读取出嵌入向量，如果没有指定文件名，则随机初始化一个嵌入矩阵
    :param counter: 计数器
    :param data_type: 数据类型
    :param limit: int，忽略低于这个阈值的key
    :param emb_file:
    :param size: 读取嵌入矩阵的前size行
    :param vec_size: 嵌入维度
    :return: 嵌入矩阵emb_mat，行为索引，列为嵌入特征
            字典token2idx_dict，key为单词或字母，value为索引
    """
    print("Generating {} embedding...".format(data_type))
    # 用于存放词嵌入或者字符嵌入
    embedding_dict = {}
    # 取出计数值大于limit的key(单词或者字母)
    filtered_elements = [k for k, v in counter.items() if v > limit]
    # 如果指定了文件，直接从文件读取所要的词向量或字符向量
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    # 如果没有指定，则随机初始化词向量或字符向量
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    # word or char-->idx映射，idx从2开始编号
    token2idx_dict = {token: idx for idx,
                      token in enumerate(embedding_dict.keys(), 2)}
    # 添加两个特殊字符
    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    # idx-->embedding vector映射
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict

def convert_to_features(config, data, word2idx_dict, char2idx_dict):

    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = config.test_para_limit
    ques_limit = config.test_ques_limit
    ans_limit = 100
    char_limit = config.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
    y1 = np.zeros([para_limit], dtype=np.float32)
    y2 = np.zeros([para_limit], dtype=np.float32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs

def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):

    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    ans_limit = 100 if is_test else config.ans_limit
    char_limit = config.char_limit

    # 用于判断样本是否出现context过长，question过长，answer起始值小于结束值
    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit or \
               (example["y2s"][0] - example["y1s"][0]) > ans_limit

    print("Processing {} examples...".format(data_type))
    # 写入到TFRecords文件
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        # 过滤有问题数据
        if filter_func(example, is_test):
            continue

        total += 1
        # 初始化参数矩阵
        # 存放一个样本的context word_index，shape=[para_limit]
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        # 存放一个样本的context char_index，shape=[para_limit, char_limit]
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        # 存放一个样本的question word_index，shape=[ques_limit]
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        # 存放一个样本的question char_index，shape=[ques_limit, char_limit]
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        # 返回单词的index
        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1
        # 返回字母的index
        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0

        record = tf.train.Example(features=tf.train.Features(feature={
                                  "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                                  "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                                  "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                                  "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                                  "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                                  "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                                  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
                                  }))
        # 写入数据序列化后的字符串
        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    # 用于过滤低频词和低频字母
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(
        config.train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(
        config.dev_file, "dev", word_counter, char_counter)
    test_examples, test_eval = process_file(
        config.test_file, "test", word_counter, char_counter)

    # 决定词嵌入使用glove还是wiki，词嵌入维度都是300维
    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    # 决定是否使用预训练的字符嵌入
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim


    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, size=config.glove_word_size, vec_size=config.glove_dim)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim)

    # 保存训练数据
    build_features(config, train_examples, "train",
                   config.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev",
                              config.dev_record_file, word2idx_dict, char2idx_dict)
    test_meta = build_features(config, test_examples, "test",
                               config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    # 保存词嵌入和字符嵌入
    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")

    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    # 记录dev，test数据量
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")
    # 保存word-->index, char-->index映射
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")
