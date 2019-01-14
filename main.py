import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


from model import Model
from demo import Demo
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)
    # 验证集总数据量
    dev_total = meta["total"]
    print("Building model...")
    parser = get_record_parser(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        # tf.data.TFRecordDataset类型
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)

        # 单次迭代器
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        # 迭代器句柄占位符
        handle = tf.placeholder(tf.string, shape=[])
        # 这个迭代器会根据handle不同而使用handle对应的迭代器
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
        model = Model(config, iterator, word_mat, char_mat, graph=g)

        # 可动态申请显存
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        loss_save = 100.0
        patience = 0
        best_f1 = 0.
        best_em = 0.

        with tf.Session(config=sess_config) as sess:
            # 指定一个文件来保存图，可以调用add_summary()将训练过成数据保存在FileWriter指定的文件中
            writer = tf.summary.FileWriter(config.log_dir)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # 各个数据集创建对应的handle
            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())

            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            global_step = max(sess.run(model.global_step), 1)

            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                                          handle: train_handle, model.dropout: config.dropout})
                # 每config.period轮记录一次损失batch_loss
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="model/loss", simple_value=loss), ])
                    # 向logs写入模型误差
                    writer.add_summary(loss_sum, global_step)
                # 每config.checkpoint轮评估和保存一次模型
                if global_step % config.checkpoint == 0:
                    # 训练集效果
                    _, summ = evaluate_batch(
                        model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                    for s in summ:
                        # 写入训练效果
                        writer.add_summary(s, global_step)

                    # 验证集效果
                    metrics, summ = evaluate_batch(
                        model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                    for s in summ:
                        # 写入验证效果
                        writer.add_summary(s, global_step)

                    dev_f1 = metrics["f1"]
                    dev_em = metrics["exact_match"]
                    # 如果大于early_stop轮没提升就停止训练
                    if dev_f1 < best_f1 and dev_em < best_em:
                        patience += 1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_em = max(best_em, dev_em)
                        best_f1 = max(best_f1, dev_f1)

                    # 保存logs
                    writer.flush()
                    # 保存模型
                    filename = os.path.join(
                        config.save_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)

def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    """
    模型评估
    :param model: Model
    :param num_batches: int, batches的数量
    :param eval_file: 验证文件
    :param sess: tf.Session
    :param data_type: 别名
    :param handle: tf.placeholder
    :param str_handle: iterator.handler
    :return:
    """
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        # 将y1,y2转换成对应的answer字符串{qa_id: answer_str}
        answer_dict_, _ = convert_tokens(
            eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    # 平均损失
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]


def demo(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    model = Model(config, None, word_mat, char_mat, trainable=False, demo = True)
    demo = Demo(model, config)


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]

    graph = tf.Graph()
    print("Loading model...")
    with graph.as_default() as g:
        test_batch = get_dataset(config.test_record_file, get_record_parser(
            config, is_test=True), config).make_one_shot_iterator()

        model = Model(config, test_batch, word_mat, char_mat, trainable=False, graph = g)
        # 可动态申请显存
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            # 加载模型
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            if config.decay < 1.0:
                sess.run(model.assign_vars)
            losses = []
            answer_dict = {}
            remapped_dict = {}
            for step in tqdm(range(total // config.batch_size + 1)):
                qa_id, loss, yp1, yp2 = sess.run(
                    [model.qa_id, model.loss, model.yp1, model.yp2])
                answer_dict_, remapped_dict_ = convert_tokens(
                    eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
                answer_dict.update(answer_dict_)
                remapped_dict.update(remapped_dict_)
                losses.append(loss)
            loss = np.mean(losses)
            metrics = evaluate(eval_file, answer_dict)
            with open(config.answer_file, "w") as fh:
                json.dump(remapped_dict, fh)
            print("Exact Match: {}, F1: {}".format(
                metrics['exact_match'], metrics['f1']))
