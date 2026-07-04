import math


class Evaluator:
    """
    评估器类，负责计算模型评估指标
    """

    def __init__(self):
        self.metrics = {}

    def _compute_mse(self, y_true, y_pred):
        """
        计算均方误差
        MSE = (1/n) * Σ(y_true - y_pred)^2
        :param y_true: 真实值数组
        :param y_pred: 预测值数组
        :return: MSE值
        """
        n = len(y_true)
        return sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n

    def _compute_rmse(self, y_true, y_pred):
        """
        计算均方根误差
        RMSE = sqrt(MSE)
        :param y_true: 真实值数组
        :param y_pred: 预测值数组
        :return: RMSE值
        """
        return math.sqrt(self._compute_mse(y_true, y_pred))

    def _compute_mae(self, y_true, y_pred):
        """
        计算平均绝对误差
        MAE = (1/n) * Σ|y_true - y_pred|
        :param y_true: 真实值数组
        :param y_pred: 预测值数组
        :return: MAE值
        """
        n = len(y_true)
        return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

    def _compute_r2(self, y_true, y_pred):
        """
        计算决定系数
        R² = 1 - SS_res / SS_tot
        SS_res = Σ(y_true - y_pred)^2
        SS_tot = Σ(y_true - mean(y_true))^2
        :param y_true: 真实值数组
        :param y_pred: 预测值数组
        :return: R²值
        """
        n = len(y_true)
        mean_y = sum(y_true) / n

        ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y_true[i] - mean_y) ** 2 for i in range(n))

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def evaluate(self, y_true, y_pred):
        """
        计算所有评估指标
        :param y_true: 真实值数组
        :param y_pred: 预测值数组
        :return: 包含所有指标的字典
        """
        if len(y_true) != len(y_pred):
            raise ValueError("真实值和预测值长度不一致")

        if len(y_true) == 0:
            raise ValueError("输入数据不能为空")

        mse = self._compute_mse(y_true, y_pred)
        rmse = self._compute_rmse(y_true, y_pred)
        mae = self._compute_mae(y_true, y_pred)
        r2 = self._compute_r2(y_true, y_pred)

        self.metrics = {
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R2': round(r2, 4)
        }

        return self.metrics

    def get_metrics(self):
        """获取评估指标"""
        return self.metrics

    def get_report(self):
        """获取评估报告文本"""
        if not self.metrics:
            return "暂无评估数据"

        report = "模型评估报告\n"
        report += "=" * 30 + "\n"
        for metric_name, value in self.metrics.items():
            report += f"{metric_name}: {value}\n"

        return report