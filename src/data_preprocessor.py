import random


class DataPreprocessor:
    """
    数据预处理类，负责数据标准化、划分训练集和测试集
    """

    def __init__(self):
        self.scaler_params = {}

    def standardize(self, X):
        """
        Z-Score标准化：x' = (x - μ) / σ
        :param X: 特征数组（二维列表）
        :return: 标准化后的特征数组
        """
        if not X or not X[0]:
            return X

        n_samples = len(X)
        n_features = len(X[0])

        means = []
        stds = []

        for j in range(n_features):
            col = [X[i][j] for i in range(n_samples)]
            mean_val = sum(col) / n_samples
            variance = sum((x - mean_val) ** 2 for x in col) / n_samples
            std_val = variance ** 0.5 if variance > 0 else 1.0

            means.append(mean_val)
            stds.append(std_val)

        self.scaler_params = {'means': means, 'stds': stds}

        X_std = []
        for i in range(n_samples):
            row = []
            for j in range(n_features):
                row.append((X[i][j] - means[j]) / stds[j])
            X_std.append(row)

        return X_std

    def split(self, X, y, test_ratio=0.2, random_seed=None):
        """
        划分训练集和测试集
        :param X: 特征数组
        :param y: 目标数组
        :param test_ratio: 测试集比例
        :param random_seed: 随机种子
        :return: (X_train, X_test, y_train, y_test)
        """
        if len(X) != len(y):
            raise ValueError("特征和目标数据长度不一致")

        n_samples = len(X)
        n_test = int(n_samples * test_ratio)

        if random_seed is not None:
            random.seed(random_seed)

        indices = list(range(n_samples))
        random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]

        return X_train, X_test, y_train, y_test

    def inverse_standardize(self, X_std):
        """
        逆标准化，将标准化数据还原为原始数据
        :param X_std: 标准化后的特征数组
        :return: 原始特征数组
        """
        if not self.scaler_params:
            return X_std

        means = self.scaler_params['means']
        stds = self.scaler_params['stds']

        X = []
        for row in X_std:
            original_row = []
            for j, val in enumerate(row):
                original_row.append(val * stds[j] + means[j])
            X.append(original_row)

        return X

    def get_scaler_params(self):
        """获取标准化参数"""
        return self.scaler_params