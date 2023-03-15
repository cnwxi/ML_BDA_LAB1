import pandas as pd
import numpy as np
import torch


def read_data(algorithm):
    dataset = pd.read_csv("./res/dataset.txt")
    print(dataset.head())
    print(dataset.columns)

    dataset = dataset.drop(columns='row.names', axis=1)
    dataset = dataset.drop(columns='name', axis=1)
    # x = dataset[['pclass', 'age', 'embarked', 'sex']]
    # y = dataset[['survived']]

    for columname in dataset.columns:
        if dataset[columname].count() != len(dataset):
            loc = dataset[columname][dataset[columname].isnull().values ==
                                     True].index.tolist()
            print('列名："{}", 第{}行位置有缺失值'.format(columname, loc))
    # 填充年龄缺省值
    median = dataset['age'].median()
    dataset['age'] = dataset['age'].fillna(median)
    # 填充登船港口缺省值
    embarked = dataset['embarked'].value_counts()
    item = embarked.idxmax()
    dataset['embarked'] = dataset['embarked'].fillna(item)

    # 字符映射到数值
    print(dataset['pclass'].unique())
    print(dataset['embarked'].unique())
    print(dataset['sex'].unique())

    dataset.loc[dataset['sex'] == 'female', 'sex'] = 0
    dataset.loc[dataset['sex'] == 'male', 'sex'] = 1
    dataset.loc[dataset['pclass'] == '1st', 'pclass'] = 0
    dataset.loc[dataset['pclass'] == '2nd', 'pclass'] = 1
    dataset.loc[dataset['pclass'] == '3rd', 'pclass'] = 2

    dataset['pclass'] = pd.to_numeric(dataset['pclass'])
    # dataset['embarked'] = pd.to_numeric(dataset['embarked'])
    dataset['sex'] = pd.to_numeric(dataset['sex'])
    if algorithm == 'dt':
        dataset.loc[dataset['embarked'] == 'Southampton', 'embarked'] = 0
        dataset.loc[dataset['embarked'] == 'Cherbourg', 'embarked'] = 1
        dataset.loc[dataset['embarked'] == 'Queenstown', 'embarked'] = 2
        dataset['embarked'] = pd.to_numeric(dataset['embarked'])
    else:
        # 逻辑回归
        # onehot编码
        dd = pd.get_dummies(dataset['embarked'],
                            drop_first=True,
                            prefix='embarked')
        dataset = pd.concat([dd, dataset.drop(columns='embarked')], axis=1)

    np.random.seed(0)
    shuffled_index = np.random.permutation(len(dataset))
    split_index = int(len(dataset) * 0.75)
    train_index = shuffled_index[:split_index]
    test_index = shuffled_index[split_index:]
    train_df = dataset.iloc[train_index]
    test_df = dataset.iloc[test_index]
    x_train = train_df.drop(columns='survived', axis=1)
    y_train = train_df['survived']
    train_features = torch.tensor(x_train.values).to(torch.float64)
    if algorithm == 'dt':
        train_df = pd.concat([x_train, y_train], axis=1)
        train_dataset = np.array(
            torch.tensor(train_df.values).to(torch.float64)).tolist()
        x_test = test_df.drop(columns='survived', axis=1)
        y_test = test_df['survived']
        test_features = torch.tensor(x_test.values).to(torch.float64)
        test_features = np.array(test_features).tolist()
        test_labels = torch.tensor(y_test.values.reshape(-1)).to(torch.float64)
        test_labels = np.array(test_labels)
        return train_dataset,  test_features, test_labels
    else:
        train_labels = torch.tensor(y_train.values.reshape(-1)).to(
            torch.float64)
        x_test = test_df.drop(columns='survived', axis=1)
        y_test = test_df['survived']
        test_features = torch.tensor(x_test.values).to(torch.float64)
        test_labels = torch.tensor(y_test.values.reshape(-1)).to(torch.float64)
        return train_features, train_labels, test_features, test_labels


if __name__ == '__main__':
    print('lr', read_data(algorithm='lr'))
    print('dt', read_data(algorithm='dt'))
