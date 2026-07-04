class DecisionTreeRegressor:
    """
    决策树回归类，基于CART算法，使用MSE作为分裂准则
    """

    def __init__(self, max_depth=5, min_samples_split=2, min_mse_decrease=0.0):
        """
        初始化决策树回归模型
        :param max_depth: 树的最大深度
        :param min_samples_split: 节点分裂所需的最小样本数
        :param min_mse_decrease: 分裂所需的最小MSE下降值
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_mse_decrease = min_mse_decrease
        self.tree = None

    def _compute_mse(self, y):
        """
        计算MSE
        :param y: 目标值数组
        :return: MSE值
        """
        if len(y) == 0:
            return 0.0
        mean_val = sum(y) / len(y)
        return sum((val - mean_val) ** 2 for val in y) / len(y)

    def _find_best_split(self, X, y):
        """
        寻找最佳分裂特征和分裂点
        :param X: 特征数组
        :param y: 目标数组
        :return: (best_feature_idx, best_split_val, best_mse)
        """
        n_samples = len(X)
        n_features = len(X[0]) if X else 0

        best_mse = float('inf')
        best_feature_idx = None
        best_split_val = None

        for feature_idx in range(n_features):
            feature_values = sorted(set([X[i][feature_idx] for i in range(n_samples)]))

            for i in range(len(feature_values) - 1):
                split_val = (feature_values[i] + feature_values[i + 1]) / 2

                left_indices = []
                right_indices = []
                for j in range(n_samples):
                    if X[j][feature_idx] <= split_val:
                        left_indices.append(j)
                    else:
                        right_indices.append(j)

                if len(left_indices) < self.min_samples_split or \
                        len(right_indices) < self.min_samples_split:
                    continue

                left_y = [y[j] for j in left_indices]
                right_y = [y[j] for j in right_indices]

                mse_left = self._compute_mse(left_y)
                mse_right = self._compute_mse(right_y)

                mse_split = (len(left_y) * mse_left + len(right_y) * mse_right) / n_samples

                if mse_split < best_mse:
                    best_mse = mse_split
                    best_feature_idx = feature_idx
                    best_split_val = split_val

        return best_feature_idx, best_split_val, best_mse

    def _build_tree(self, X, y, depth=0):
        """
        递归构建决策树
        :param X: 特征数组
        :param y: 目标数组
        :param depth: 当前深度
        :return: 树节点（字典或叶子节点值）
        """
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'type': 'leaf', 'value': sum(y) / len(y)}

        current_mse = self._compute_mse(y)

        feature_idx, split_val, split_mse = self._find_best_split(X, y)

        if feature_idx is None or current_mse - split_mse < self.min_mse_decrease:
            return {'type': 'leaf', 'value': sum(y) / len(y)}

        left_indices = []
        right_indices = []
        for i in range(len(X)):
            if X[i][feature_idx] <= split_val:
                left_indices.append(i)
            else:
                right_indices.append(i)

        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]

        left_tree = self._build_tree(left_X, left_y, depth + 1)
        right_tree = self._build_tree(right_X, right_y, depth + 1)

        return {
            'type': 'split',
            'feature_idx': feature_idx,
            'split_val': split_val,
            'left': left_tree,
            'right': right_tree
        }

    def train(self, X, y):
        """
        训练决策树模型
        :param X: 特征数组（二维列表）
        :param y: 目标数组
        """
        if not X or not y:
            raise ValueError("训练数据不能为空")

        self.tree = self._build_tree(X, y)

    def _predict_single(self, x, tree):
        """
        对单个样本进行预测
        :param x: 单个样本特征
        :param tree: 决策树
        :return: 预测值
        """
        if tree['type'] == 'leaf':
            return tree['value']

        if x[tree['feature_idx']] <= tree['split_val']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

    def predict(self, X):
        """
        预测目标值
        :param X: 特征数组（二维列表）
        :return: 预测值数组
        """
        if self.tree is None:
            raise ValueError("模型未训练，请先调用train方法")

        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.tree))
        return predictions

    def get_params(self):
        """获取模型参数"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_mse_decrease': self.min_mse_decrease
        }

    def set_params(self, params):
        """设置模型参数"""
        if 'max_depth' in params:
            self.max_depth = params['max_depth']
        if 'min_samples_split' in params:
            self.min_samples_split = params['min_samples_split']
        if 'min_mse_decrease' in params:
            self.min_mse_decrease = params['min_mse_decrease']