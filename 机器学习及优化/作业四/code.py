import operator
import random
from collections import Counter
from math import log
from random import randrange

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

DATA_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

FEATURE_COLUMNS = [
    'age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

NUMERIC_COLUMNS = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']


def load_raw_data():
    """读取 Adult 数据集。"""
    train_df = pd.read_csv('adult/adult.data', header=None, names=DATA_COLUMNS)
    test_df = pd.read_csv('adult/adult.test', header=None, names=DATA_COLUMNS, skiprows=1)
    return train_df, test_df


def clean_raw_data(train_df, test_df):
    """清理空白、缺失值标记和数值类型。"""
    for frame in (train_df, test_df):
        for column in frame.select_dtypes(include='object').columns:
            frame[column] = frame[column].str.strip()
        frame.replace('?', np.nan, inplace=True)

    test_df['income'] = test_df['income'].str.replace('.', '', regex=False)

    for frame in (train_df, test_df):
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
        frame.dropna(subset=NUMERIC_COLUMNS, inplace=True)
        frame.dropna(inplace=True)
        frame.reset_index(drop=True, inplace=True)

    return train_df, test_df


def bin_age(series):
    return pd.cut(series, bins=[-np.inf, 24, 49, 74, np.inf], labels=[0, 1, 2, 3]).astype(int)


def bin_binary_numeric(series):
    return (series > 0).astype(int)


def bin_hours(series):
    return pd.cut(series, bins=[-np.inf, 39, 40, np.inf], labels=[0, 1, 2]).astype(int)


def bin_country(series):
    return np.where(series == 'United-States', 0, 1)


def bin_workclass(series):
    mapping = {
        'Federal-gov': 'Government',
        'Local-gov': 'Government',
        'State-gov': 'Government',
        'Self-emp-not-inc': 'Proprietor',
        'Self-emp-inc': 'Proprietor',
    }
    return series.replace(mapping)


def bin_education_num(series):
    return pd.cut(series, bins=[-np.inf, 4, 9, np.inf], labels=[0, 1, 2]).astype(int)


def bin_marital_status(series):
    return np.where(series.isin(['Divorced', 'Never-married', 'Separated', 'Widowed']), 'not-married', 'married')


def bin_occupation(series):
    high = {'Prof-specialty', 'Exec-managerial'}
    med = {'Tech-support', 'Transport-moving', 'Protective-serv', 'Sales', 'Craft-repair', 'Armed-Forces'}
    return np.select([series.isin(high), series.isin(med)], ['High', 'Med'], default='Low')


def bin_relationship(series):
    return np.where(series.isin(['Husband', 'Wife']), series, 'Other')


def bin_race(series):
    return np.where(series == 'White', 'White', 'Other')


def preprocess_data(train_df, test_df):
    """完成特征离散化、标签编码和最终列裁剪。"""
    transformations = {
        'age': bin_age,
        'capital-gain': bin_binary_numeric,
        'capital-loss': bin_binary_numeric,
        'hours-per-week': bin_hours,
        'native-country': bin_country,
        'workclass': bin_workclass,
        'education-num': bin_education_num,
        'marital-status': bin_marital_status,
        'occupation': bin_occupation,
        'relationship': bin_relationship,
        'race': bin_race,
    }

    for column, transform_func in transformations.items():
        train_df[column] = transform_func(train_df[column].copy())
        test_df[column] = transform_func(test_df[column].copy())

    label_encoder = LabelEncoder()
    train_df['income'] = label_encoder.fit_transform(train_df['income'])
    test_df['income'] = label_encoder.transform(test_df['income'])

    train_df = train_df.drop(columns=['fnlwgt', 'education'])
    test_df = test_df.drop(columns=['fnlwgt', 'education'])

    return train_df, test_df, label_encoder


def createDataSet(train_df):
    dataSet = train_df.values.tolist()
    featureName = FEATURE_COLUMNS + ['income']
    return dataSet, featureName


def evaluate_predictions(y_true, y_pred, title):
    correct_num = sum(int(pred == true) for pred, true in zip(y_pred, y_true))
    rate = correct_num / len(y_true)
    print(f'{title}:', rate)
    print(f'{title}AUC值:', roc_auc_score(y_true, y_pred))


def run_decision_tree(train_df, test_df):
    data_set, feature_name = createDataSet(train_df)
    tree = createTree(data_set, feature_name)
    feature_labels = FEATURE_COLUMNS + ['income']

    train_records = train_df.values.tolist()
    test_records = test_df.values.tolist()
    y_train = train_df['income'].tolist()
    y_test = test_df['income'].tolist()

    train_predictions = [classify(tree, feature_labels, record) for record in train_records]
    test_predictions = [classify(tree, feature_labels, record) for record in test_records]

    evaluate_predictions(y_test, test_predictions, '决策树准确率')
    evaluate_predictions(y_train, train_predictions, '决策树训练集上准确率')
    return data_set, y_test, test_records


def get_subsample(dataSet):
    subdataSet = []
    lenSubdata = len(dataSet)
    while len(subdataSet) < lenSubdata:
        index = randrange(len(dataSet))
        subdataSet.append(dataSet[index])
    return subdataSet


def get_subfeature(featureLabels, n_features):
    subFeatIndex = random.sample(range(0, len(featureLabels)), n_features)
    subFeature = [featureLabels[index] for index in subFeatIndex]
    subFeature.append('income')
    subFeatIndex.append(len(featureLabels))
    return subFeature, subFeatIndex


def generateDataSet(dataSet, featLabels, n_features):
    subDataSet = get_subsample(dataSet)
    subFeature, subFeatIndex = get_subfeature(featLabels, n_features)
    final_subData = []
    for row in subDataSet:
        row_list = [row[index] for index in subFeatIndex]
        final_subData.append(row_list)
    return final_subData, subFeature


def RandomForest(dataSet, featLabels, n_trees, n_features):
    tree_list = []
    for _ in range(n_trees):
        final_subData, subFeature = generateDataSet(dataSet, featLabels, n_features)
        tree_feature_labels = subFeature[:]
        myTree = createTree(final_subData, tree_feature_labels)
        tree_list.append((myTree, subFeature[:]))
    return tree_list


def predict_forest(tree_list, test_records):
    pred_list = []
    for record in test_records:
        class_list = [classify(tree, feat_labels, record) for tree, feat_labels in tree_list]
        pred_list.append(class_list)
    return [Counter(votes).most_common(1)[0][0] for votes in pred_list]


def run_random_forest(data_set, feat_labels, test_records, y_test, n_features=8, n_trees=10):
    tree_list = RandomForest(data_set, feat_labels, n_trees, n_features)
    predictions = predict_forest(tree_list, test_records)
    evaluate_predictions(y_test, predictions, '随机森林准确率')


def main():
    train_df, test_df = load_raw_data()
    train_df, test_df = clean_raw_data(train_df, test_df)
    train_df, test_df, _ = preprocess_data(train_df, test_df)

    train_df.info()
    print(train_df.describe())
    test_df.info()

    data_set, y_test, test_records = run_decision_tree(train_df, test_df)
    run_random_forest(data_set, FEATURE_COLUMNS, test_records, y_test)

# 分割数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

# 计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 计算条件熵
def calcConditionalEntropy(dataSet, i, featList, uniqueVals):
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))
        ce += prob * calcShannonEnt(subDataSet)
    return ce

# 计算信息增益
def calcInformationGain(dataSet, baseEntropy, i):
    featList = [example[i] for example in dataSet]
    uniqueVals = set(featList)
    newEntropy = calcConditionalEntropy(dataSet, i, featList, uniqueVals)
    infoGain = baseEntropy - newEntropy
    return infoGain

# 选择最佳特征
def chooseBestFeatureToSplitByID3(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -2
    for i in range(numFeatures):
        infoGain = calcInformationGain(dataSet, baseEntropy, i)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 多数表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建决策树
def createTree(dataSet, featureName, chooseBestFeatureToSplitFunc=chooseBestFeatureToSplitByID3):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplitFunc(dataSet)
    bestFeatLabel = featureName[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(featureName[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = featureName[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    if key not in secondDict:
        key = list(secondDict.keys())[0]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


if __name__ == '__main__':
    main()

