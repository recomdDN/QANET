import tensorflow as tf
import re
from collections import Counter
import string

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


def get_record_parser(config, is_test=False):
    """
    解析序列化后的tf.train.Example数据
    :param config:
    :param is_test:
    :return: 函数解析parse
    """
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit
        # 解析tfrecord中的每条记录，即序列化后的tf.train.Example
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.string),
                                               "y2": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })
        # 将数据进行解析
        context_idxs = tf.reshape(tf.decode_raw(
            features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
            features["ques_idxs"], tf.int32), [ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(
            features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(
            features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        y1 = tf.reshape(tf.decode_raw(
            features["y1"], tf.float32), [para_limit])
        y2 = tf.reshape(tf.decode_raw(
            features["y2"], tf.float32), [para_limit])
        qa_id = features["id"]
        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    # 从tfrecord文件创建TFRecordDataset
    # shuffle是打乱数据，并取出buffer_size个，repeat表示无限期重复数据
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(buffer_size=config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


def convert_tokens(eval_file, qa_id, pp1, pp2):
    """
    返回答案字符串
    :param eval_file: 验证文件
    :param qa_id: qa对的id
    :param pp1: y1
    :param pp2: y2
    :return: 两个dict, qa_id-->answer_str, uuid-->answer_str
    """
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        # 首字母位置
        start_idx = spans[p1][0]
        # 末字母位置
        end_idx = spans[p2][1]
        # qaid-->answer字符串
        answer_dict[str(qid)] = context[start_idx: end_idx]
        # uuid-->answer字符串
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    """
    根据ground_truth_str和answer_str
    :param eval_file: 验证文件
    :param answer_dict: dict, {qa_id: answer_str}
    :return: 匹配度和f1值的dict
    """
    f1 = exact_match = total = 0
    # 逐个预测值计算匹配度和f1值
    for key, value in answer_dict.items():
        total += 1
        # 多个ground_truth_str
        ground_truths = eval_file[key]["answers"]
        # answer_str
        prediction = value
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    # 百分比
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    # 把单词a,an,the换成空格
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    # 多个空格换成一个空格
    def white_space_fix(text):
        return ' '.join(text.split())
    # 去除特殊符号
    def remove_punc(text):
        # 特殊符号
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    # 字母转小写
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    返回f1值
    :param prediction: 预测值
    :param ground_truth: 真实值
    :return:
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    # 并集
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    判断预测值字符串和ground_truth字符串是否完全一样
    :param prediction: 预测字符串
    :param ground_truth: 真实值字符串
    :return:
    """
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    :param metric_fn: 评价函数
    :param prediction: 预测值
    :param ground_truths: 多个真实值
    :return: 返回评估最大值
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
