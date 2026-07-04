from .linear_regression import LinearRegression
from .decision_tree import DecisionTreeRegressor


class ModelTrainer:
    """
    模型训练器类，负责创建和训练指定的回归模型
    """

    def __init__(self, model_type='linear_regression', parameters=None):
        """
        初始化模型训练器
        :param model_type: 模型类型，可选 'linear_regression' 或 'decision_tree'
        :param parameters: 模型参数字典
        """
        self.model_type = model_type
        self.parameters = parameters or {}
        self.model = None

    def create_model(self):
        """
        创建指定类型的模型
        :return: 创建的模型实例
        """
        if self.model_type == 'linear_regression':
            learning_rate = self.parameters.get('learning_rate', 0.01)
            max_iter = self.parameters.get('max_iter', 1000)
            alpha = self.parameters.get('alpha', 0.0)
            self.model = LinearRegression(learning_rate, max_iter, alpha)
        elif self.model_type == 'decision_tree':
            max_depth = self.parameters.get('max_depth', 5)
            min_samples_split = self.parameters.get('min_samples_split', 2)
            min_mse_decrease = self.parameters.get('min_mse_decrease', 0.0)
            self.model = DecisionTreeRegressor(max_depth, min_samples_split, min_mse_decrease)
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

        return self.model

    def train(self, X_train, y_train):
        """
        训练模型
        :param X_train: 训练特征数组
        :param y_train: 训练目标数组
        """
        if self.model is None:
            self.create_model()

        self.model.train(X_train, y_train)

    def predict(self, X_test):
        """
        使用训练好的模型进行预测
        :param X_test: 测试特征数组
        :return: 预测值数组
        """
        if self.model is None:
            raise ValueError("模型未创建，请先调用create_model方法")

        return self.model.predict(X_test)

    def get_model(self):
        """获取当前模型实例"""
        return self.model

    def get_model_type(self):
        """获取模型类型"""
        return self.model_type

    def get_parameters(self):
        """获取模型参数"""
        return self.parameters

    def set_parameters(self, parameters):
        """设置模型参数"""
        self.parameters = parameters
        self.model = None