import csv
import os
from datetime import datetime


class ResultExporter:
    """
    结果导出器类，负责将评估报告和预测结果保存到文件
    """

    def __init__(self):
        self.output_dir = 'output'

    def _ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def export_report(self, metrics, model_type, parameters, file_path=None):
        """
        导出评估报告到TXT文件
        :param metrics: 评估指标字典
        :param model_type: 模型类型
        :param parameters: 模型参数字典
        :param file_path: 输出文件路径，为None时自动生成
        :return: 导出的文件路径
        """
        self._ensure_output_dir()

        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.output_dir, f'report_{model_type}_{timestamp}.txt')

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("混凝土抗压强度预测系统 - 评估报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型类型: {model_type}\n")
                f.write("\n模型参数:\n")
                for key, value in parameters.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n评估指标:\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value}\n")
                f.write("=" * 50 + "\n")

            return file_path
        except Exception as e:
            raise ValueError(f"导出报告失败: {str(e)}")

    def export_predictions(self, y_true, y_pred, file_path=None):
        """
        导出预测结果到CSV文件
        :param y_true: 真实值数组
        :param y_pred: 预测值数组
        :param file_path: 输出文件路径，为None时自动生成
        :return: 导出的文件路径
        """
        self._ensure_output_dir()

        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.output_dir, f'predictions_{timestamp}.csv')

        try:
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['真实值', '预测值', '残差'])
                for i in range(len(y_true)):
                    residual = y_true[i] - y_pred[i]
                    writer.writerow([y_true[i], y_pred[i], round(residual, 4)])

            return file_path
        except Exception as e:
            raise ValueError(f"导出预测结果失败: {str(e)}")

    def export_all(self, metrics, model_type, parameters, y_true, y_pred):
        """
        导出评估报告和预测结果
        :param metrics: 评估指标字典
        :param model_type: 模型类型
        :param parameters: 模型参数字典
        :param y_true: 真实值数组
        :param y_pred: 预测值数组
        :return: (报告路径, 预测结果路径)
        """
        report_path = self.export_report(metrics, model_type, parameters)
        predictions_path = self.export_predictions(y_true, y_pred)
        return report_path, predictions_path