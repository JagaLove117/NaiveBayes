# coding: UTF-8
# import pandas as pd
# import texttable
import os
import pickle
import sys
import csv
import copy
import numpy as np

# args = sys.argv
# args[1] = "-train"
mode = "-test"

"""
train_file_name =args[1]+'.stem.txt'
"""

file_excist = 0  # ファイルがあるかないかのフラグ
train_file_num = 0  # ファイルナンバー 0から
test_file_num = 0

zf = 4  # ファイル名が数値の場合、何文字詰か
stem_dict_list = []  # ファイル毎の単語辞書　stem_dict_list[train_file_num] = list
stem_count_list = []  # ファイル毎の単語出現回数リスト stem_dict_list[train_file_num] = list
# stem_vocablary = []
word_dict_list = []  # 単語辞書
word_count_list = []  # 単語ごとの出現回数リスト
vocabulary = 0  # 語彙数
which_class_list = []  # ファイル毎の該当クラス
which_file_list = []


def regist_dict(word_dict_list, x, word_count_list_list, num):  # 全ファイルと個別の辞書登録＆カウント
    print("ファイルナンバーは" + str(num))

    global vocabulary
    if word_dict_list.count(x) == 0:  # 辞書に載っていない単語の場合
        word_dict_list.append(x)  # 辞書に登録
        print(x + "を登録しました。")  # メッセージ : 登録
        word_count_list.append(1)  # 出現回数1を追加
        vocabulary += 1  # 語彙数を1増やす

    else:  # 辞書に載っている単語の場合
        word_count_list[word_dict_list.index(x)] += 1  # 出現回数リストの対応する数値を+1
        print(x + "は登録済みです。")  # メッセージ : 登録済み

    print("ファイルナンバーは" + str(num))

    if list_dict.count(x) == 0:  # 個別ファイル辞書に載っていない単語の場合
        list_dict.append(x)  #
        print(x + "を登録しました")
        list_count.append(1)
    else:
        list_count[list_dict.index(x)] += 1
        print(x + "は" + str(num) + "に登録済みです")
    return list_dict, list_count


def which_file(cn):  # そのクラスにどのファイルが属しているか
    wf_result = []
    for i in range(0, cn + 1):
        tmp = []
        for var in range(0, train_file_num):
            if which_class_list[var] == i:
                tmp.append(var)
        wf_result.append(tmp)
    return wf_result


def class_probability_list(cn, f_num):  # クラス毎の出現確率のlogをリストにして返してくれる
    tmp = np.empty(cn + 1)
    cp_result = np.empty(cn + 1)
    a = np.empty(cn + 1)
    for var in range(0, cn + 1):
        tmp[var] = which_class_list.count(var)
    for var in range(0, len(tmp)):
        a[var] = tmp[var] / f_num
        print(a)
        if (a[var] == 0):
            cp_result[var] = 0
        else:
            cp_result[var] = np.log2(a[var])
            # pc_result.append(tmp[var]/f_num)
    return cp_result


def word_class_probability_list(dic_list, cnt_list, f_num, v, cn, dt):  # {P(w|c)}を返す{ファイル数,語彙数,クラス数,スムージング}
    # P(w1|c1) =n(w1;c1)/n(c1)
    # そのclassのなかの単語総数
    class_words_num = []
    w_c_p_result = np.empty((cn + 1, v))

    result = []
    np_result = np.empty((cn, v))
    tmp = []
    for i in range(0, cn + 1):
        s = 0
        for var in range(0, len(which_file_list[i])):
            s = s + sum(cnt_list[var])
        class_words_num.append(s)
    s = 0
    for c in range(0, cn + 1):
        __word = []
        for var in range(0, v):
            for i in range(0, f_num):
                if which_class_list[i] == c:
                    for j in range(0, len(dic_list[i])):
                        if word_dict_list[var] == dic_list[i][j]:
                            s = s + cnt_list[i][j]
                            break
            __word.append(s)
            s = 0
        tmp = copy.deepcopy(__word)
        result.append(tmp)
    np_result = result
    print("-----クラス毎の単語分布-----")
    print(np_result)
    f = open('../data/training_count.list', 'wb')
    pickle.dump(np_result, f)
    f.close()
    print(word_dict_list)
    print("---クラス毎の語数---")
    for c in range(0, cn + 1):
        a = (np.sum(np_result, axis=1))[c]
        print("クラス" + str(c) + ":" + str(a) + "語")
        for var in range(0, v):
            w_c_p_result[c][var] = np.log2((np_result[c][var] + dt) / (a + v * dt))
            # w_c_p_result = np.log(w_c_p_result)
    return w_c_p_result


def tester(w_freq, v, pc, pwc, c):
    p_score = np.zeros(c)
    print(p_score)
    for var in range(0, c):
        for i in range(0, v):
            print(str(i)+"　の文字頻度は　"+str(w_freq[i]))
            p_score[var] = p_score[var]+(pc[var] + (w_freq[i] * pwc[var][i]))
            print(str(var)+"のスコアは"+str(p_score[var]))
    return np.argmax(p_score), np.amax(p_score)


def name_path_maker(name, ext, p, zf):
    f_name = ""
    f_path = ""
    if isinstance(name, int):
        f_name = str(name).zfill(zf) + ext
        f_path = p + f_name
    return f_name, f_path


# table = texttable()

if (mode == "-train"):
    # train_file_name = str(train_file_num).zfill(4) + '.stem.txt'  # ファイルの名前取得
    # file_path = '../doc/' + train_file_name  # ファイルパス取得
    train_file_name, file_path = name_path_maker(train_file_num, '.stem', '../doc/', zf)  # ファイル名,ファイルパス取得
    # if args[1] == "-train" :
    print("訓練を行います。")
    while os.path.isfile(file_path):  # ファイルが存在している間実行
        list_dict = []
        list_count = []
        print('ファイルの場所は' + file_path)  # メッセージ : ファイルのパス
        f = open(file_path, 'r')
        word = (f.readline()).replace('\n', '')

        # クラス読み込み
        which_class_list.append(int(word))
        word = (f.readline()).replace('\n', '')
        # ファイル一行づつ読み込み
        while word:
            print(word + "を読み込みました。")
            dict, count = regist_dict(word_dict_list, word, word_count_list, train_file_num)
            word = (f.readline()).replace('\n', '')
        f.close()
        stem_dict_list.append(dict)
        stem_count_list.append(count)
        print(train_file_name + 'までの作業を終了しました。\n')
        train_file_num += 1
        train_file_name, file_path = name_path_maker(train_file_num, '.stem', '../doc/', zf)
        print(word_dict_list)

    print('\n最終結果')
    print("クラスは")
    print(which_class_list)
    c_num = max(which_class_list)
    print("クラスの最大値は" + str(c_num))
    print(word_dict_list)
    print(word_count_list)
    print('辞書作成終了しました。')
    print(stem_dict_list)
    print(stem_count_list)
    print("train_file_numは" + str(train_file_num))

    which_file_list = which_file(c_num)
    print("which_fileは")
    print(which_file_list)

    # P{c}を計算
    print("計算開始")
    prob_class = class_probability_list(c_num, train_file_num)
    print("---log{P(c)}の結果は---")
    print(prob_class)

    prob_word_class = word_class_probability_list(stem_dict_list, stem_count_list, train_file_num, vocabulary, c_num, 1)
    print("---{P(w|c)}の結果は---")
    print(prob_word_class)

    print(np.sum(prob_word_class, axis=1))

    # データの保存
    np.save('../data/pc_train', prob_class)
    np.save('../data/pwc_train', prob_word_class)
    f = open('../data/vocablary.list', 'wb')
    pickle.dump(word_dict_list, f)
    f.close()

    for i in range(0, train_file_num):
        f = open('../bin/' + str(i).zfill(4) + '.csv', 'wt')  # ファイルナンバーに対応したcsvファイルを作る
        csv.writer(f).writerow(stem_dict_list[i])
        csv.writer(f).writerow(stem_count_list[i])
        f.close()
    f = open('../data/training_sample.list', 'wb')
    stem_all = []
    stem_all.append(stem_dict_list)
    stem_all.append(stem_count_list)
    pickle.dump(stem_all, f)
    a = np.load('../data/pc_train.npy')
    b = np.load('../data/pwc_train.npy')
    print("ロードした配列(pc)を出力")
    print(a)
    print("ロードした配列(pwc)を出力")
    print(b)

elif mode == "-test":
    print("testを行います。")
    prob_class = np.load('../data/pc_train.npy')
    prob_word_class = np.load('../data/pwc_train.npy')
    f = open('../data/vocablary.list', 'rb')
    train_dict = []
    train_dict = pickle.load(f)
    f.close()
    f = open('../data/training_count.list', 'rb')
    train_count = []
    train_count = pickle.load(f)
    f.close()
    # print(train_count)
    f = open('../data/training_sample.list', 'rb')
    train_sample = []
    train_sample = pickle.load(f)
    f.close()
    vocabulary = len(train_dict)
    # print(train_dict)
    # print(prob_class)
    # print(prob_word_class)
    # print(len(prob_class))
    # print(train_sample)

    test_file_name, test_file_path = name_path_maker(test_file_num, '.test', '../test/', zf)
    with open('../data/test_result.csv', 'wt') as score_file:
        while os.path.isfile(test_file_path):  # ファイルが存在している間実行
            f = open(test_file_path, 'r')
            test_words = []
            word_frequency = np.empty(vocabulary)
            string = (f.readline()).replace('\n', '')
            while string:
                test_words.append(string)
                string = (f.readline()).replace('\n', '')
            f.close()
            for i in range(0, vocabulary):
                word_frequency[i] = test_words.count(train_dict[i])
                #print(str(i)+"　の文字頻度は　"+str(word_frequency[i]))
            # print(word_frequency)
            estimate_class, score = tester(word_frequency, vocabulary, prob_class, prob_word_class, prob_class.size)
            writer = csv.writer(score_file, lineterminator='\n')
            writer.writerow([test_file_num, estimate_class, score])
            print(str(test_file_name) + " の予測されるクラスは")
            print(estimate_class)
            print("scoreは " + str(score))
            test_file_num += 1
            test_file_name, test_file_path = name_path_maker(test_file_num, '.test', '../test/', zf)




else:
    print("python main.py [mode]")