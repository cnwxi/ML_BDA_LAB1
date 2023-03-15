from res import treeplotter as tpt

from res.myread import read_data
from math import log
import operator
import numpy as np

train_dataset, test_features, test_labels = read_data('dt')

step = 50
for i in range(len(train_dataset)):
    train_dataset[i][1] = train_dataset[i][1] // step * step
for i in range(len(test_features)):
    test_features[i][1] = test_features[i][1] // step * step


def calc_shannon_ent(dataset):
    num_entries = len(dataset)
    label_counts = {}
    for featVec in dataset:
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def majority_cnt(class_list):
    """
    输入：分类类别列表
    输出：子节点的分类
    描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
          采用多数判决的方法决定该子节点的分类
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def choose_best_feature_to_split(dataset):
    """
    输入：数据集
    输出：最好的划分维度
    描述：选择最好的数据集划分维度
    """
    num_features = len(dataset[0]) - 1  # feature个数
    base_entropy = calc_shannon_ent(dataset)  # 整个dataset的熵
    best_info_gain_ratio = 0.0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]  # 每个feature的list
        unique_vals = set(feat_list)  # 每个list的唯一值集合
        new_entropy = 0.0
        split_info = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)  # 每个唯一值对应的剩余feature的组成子集
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannon_ent(sub_dataset)
            split_info += -prob * log(prob, 2)
        info_gain = base_entropy - new_entropy  # 这个feature的infoGain
        if split_info == 0:  # fix the overflow bug
            continue
        info_gain_ratio = info_gain / split_info  # 这个feature的infoGainRatio
        if info_gain_ratio > best_info_gain_ratio:  # 选择最大的gain ratio
            best_info_gain_ratio = info_gain_ratio
            best_feature = i  # 选择最大的gain ratio对应的feature
    return best_feature


def split_dataset(dataset, axis, value):
    """
    输入：数据集，选择维度，选择值
    输出：划分数据集
    描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
    """
    ret_dataset = []
    for featVec in dataset:
        if featVec[axis] == value:  # 只看当第i列的值＝value时的item
            reduce_feat_vec = featVec[:axis]  # featVec的第i列给除去
            reduce_feat_vec.extend(featVec[axis + 1:])
            ret_dataset.append(reduce_feat_vec)
    return ret_dataset


def create_tree(dataset, label):
    """
    输入：数据集，特征标签
    输出：决策树
    描述：递归构建决策树，利用上述的函数
    """
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        # classList所有元素都相等，即类别完全相同，停止划分
        return class_list[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(dataset)
    # 选择最大的gain ratio对应的feature
    best_feat_label = label[best_feat]
    my_tree = {best_feat_label: {}}
    del (label[best_feat])
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = label[:]
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_labels)
        # 划分数据，为下一层计算准备
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """

    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    class_label = None
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            # test向量的当前feature是哪个值，就走哪个树杈
            if type(second_dict[key]).__name__ == 'dict':
                # 如果secondDict[key]仍然是字典，则继续向下层走
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                # 如果secondDict[key]已经只是分类标签了，则返回这个类别标签
                class_label = second_dict[key]
    return class_label


def classify_all(input_tree, feat_labels, test_dataset):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    class_label_all = []
    for testVec in test_dataset:  # 逐个item进行分类判断
        class_label_all.append(classify(input_tree, feat_labels, testVec))
    return class_label_all


# 生成决策树
labels = ['pclass', 'age', 'embarked', 'sex']
labels_tmp = labels[:]
desicion_tree = create_tree(train_dataset, labels_tmp)
# tpt.createPlot(desicion_tree)
# 分类
# test_pred = classifyAll(desicionTree, labels, test_dataset)
# print(test_dataset[:10])
# test = [1.0, 0.0, 2.0, 22.0, 0.0]
# print(classify(desicionTree, labels, test))


test_pred = np.array(classify_all(desicion_tree, labels, test_features))
correct = ((test_pred == test_labels).sum())  # 计算正确预测的样本个数
acc = correct.item() / len(test_labels)  # 计算分类准确率
print('acc:', acc)
