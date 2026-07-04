import csv
import os


class DataReader:
    """
    数据读取器类，负责从CSV文件读取数据集
    """

    def __init__(self, file_path=None):
        """
        初始化数据读取器
        :param file_path: CSV文件路径
        """
        self.file_path = file_path
        self.feature_names = []
        self.target_name = ''

    def load_data(self, file_path=None):
        """
        加载CSV数据文件
        :param file_path: CSV文件路径，如果为None则使用初始化时的路径
        :return: (X, y, feature_names) 特征数组、目标数组、特征名称列表
        :raises FileNotFoundError: 文件不存在
        :raises ValueError: 数据格式错误
        """
        if file_path is not None:
            self.file_path = file_path

        if self.file_path is None:
            raise ValueError("文件路径未指定")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        X = []
        y = []

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)

                if len(header) < 2:
                    raise ValueError("CSV文件至少需要2列（特征+目标）")

                self.feature_names = header[:-1]
                self.target_name = header[-1]

                for row in reader:
                    if len(row) != len(header):
                        continue

                    try:
                        features = [float(val) for val in row[:-1]]
                        target = float(row[-1])
                        X.append(features)
                        y.append(target)
                    except ValueError:
                        continue

            if len(X) == 0:
                raise ValueError("没有有效数据")

            return X, y, self.feature_names

        except Exception as e:
            raise ValueError(f"数据读取错误: {str(e)}")

    def get_file_path(self):
        """获取当前文件路径"""
        return self.file_path

    def get_feature_names(self):
        """获取特征名称列表"""
        return self.feature_names

    def get_target_name(self):
        """获取目标列名称"""
        return self.target_name