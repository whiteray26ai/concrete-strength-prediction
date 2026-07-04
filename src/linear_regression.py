import math


class LinearRegression:
    """
    多元线性回归类，基于梯度下降实现，支持L2正则化
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, alpha=0.0):
        """
        初始化线性回归模型
        :param learning_rate: 学习率
        :param max_iter: 最大迭代次数
        :param alpha: L2正则化系数，0表示不使用正则化
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.theta = None

    def _add_bias(self, X):
        """
        为特征矩阵添加偏置列（全1列）
        :param X: 特征数组（二维列表）
        :return: 添加偏置后的特征数组
        """
        return [[1.0] + row for row in X]

    def _predict(self, X, theta):
        """
        计算预测值 h(x) = Xθ
        :param X: 添加偏置后的特征数组
        :param theta: 参数向量
        :return: 预测值数组
        """
        predictions = []
        for row in X:
            pred = sum(theta[j] * row[j] for j in range(len(theta)))
            predictions.append(pred)
        return predictions

    def _compute_mse(self, y_true, y_pred):
        """
        计算均方误差
        :param y_true: 真实值数组
        :param y_pred: 预测值数组
        :return: MSE值
        """
        n = len(y_true)
        return sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n

    def _gradient_descent(self, X_bias, y):
        """
        梯度下降算法
        :param X_bias: 添加偏置后的特征数组
        :param y: 目标数组
        """
        n_samples = len(X_bias)
        n_features = len(X_bias[0])

        self.theta = [0.0] * n_features

        for _ in range(self.max_iter):
            predictions = self._predict(X_bias, self.theta)

            gradients = [0.0] * n_features
            for j in range(n_features):
                gradient = sum(
                    (predictions[i] - y[i]) * X_bias[i][j]
                    for i in range(n_samples)
                ) / n_samples

                if j > 0:
                    gradient += (self.alpha / n_samples) * self.theta[j]

                gradients[j] = gradient

            for j in range(n_features):
                self.theta[j] -= self.learning_rate * gradients[j]

    def train(self, X, y):
        """
        训练线性回归模型
        :param X: 特征数组（二维列表）
        :param y: 目标数组
        """
        if not X or not y:
            raise ValueError("训练数据不能为空")

        X_bias = self._add_bias(X)
        self._gradient_descent(X_bias, y)

    def predict(self, X):
        """
        预测目标值
        :param X: 特征数组（二维列表）
        :return: 预测值数组
        """
        if self.theta is None:
            raise ValueError("模型未训练，请先调用train方法")

        X_bias = self._add_bias(X)
        return self._predict(X_bias, self.theta)

    def get_params(self):
        """获取模型参数"""
        return {'theta': self.theta, 'learning_rate': self.learning_rate,
                'max_iter': self.max_iter, 'alpha': self.alpha}

    def set_params(self, params):
        """设置模型参数"""
        if 'theta' in params:
            self.theta = params['theta']
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        if 'max_iter' in params:
            self.max_iter = params['max_iter']
        if 'alpha' in params:
            self.alpha = params['alpha']