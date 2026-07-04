import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

from .data_reader import DataReader
from .data_preprocessor import DataPreprocessor
from .model_trainer import ModelTrainer
from .evaluator import Evaluator
from .result_exporter import ResultExporter


class MainApplication(tk.Tk):
    """
    混凝土抗压强度预测系统主界面类
    """

    def __init__(self):
        super().__init__()
        self.title("混凝土抗压强度预测系统")
        self.geometry("1000x700")
        self.resizable(True, True)

        self.reader = DataReader()
        self.preprocessor = DataPreprocessor()
        self.trainer = None
        self.evaluator = Evaluator()
        self.exporter = ResultExporter()

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.metrics = None
        self.feature_names = None

        self._create_widgets()

    def _create_widgets(self):
        """创建界面控件"""
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        self._create_top_panel()
        self._create_left_panel()
        self._create_right_panel()
        self._create_bottom_panel()

    def _create_top_panel(self):
        """创建顶部文件选择区域"""
        top_frame = tk.Frame(self, padx=10, pady=5, bg="#f0f0f0")
        top_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        tk.Label(top_frame, text="数据文件:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)

        self.file_path_var = tk.StringVar(value="未选择文件")
        self.file_label = tk.Label(top_frame, textvariable=self.file_path_var,
                                   bg="white", relief=tk.SUNKEN, width=60)
        self.file_label.pack(side=tk.LEFT, padx=5)

        self.load_btn = tk.Button(top_frame, text="打开数据文件", command=self.load_file)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.data_info_var = tk.StringVar(value="")
        tk.Label(top_frame, textvariable=self.data_info_var, bg="#f0f0f0").pack(side=tk.RIGHT)

    def _create_left_panel(self):
        """创建左侧参数设置区域"""
        left_frame = tk.Frame(self, padx=10, pady=5, bg="#f5f5f5")
        left_frame.grid(row=1, column=0, sticky="nsew")

        tk.Label(left_frame, text="参数设置", font=("Arial", 12, "bold"),
                 bg="#f5f5f5").pack(pady=(0, 10))

        model_frame = tk.LabelFrame(left_frame, text="模型选择", padx=5, pady=5)
        model_frame.pack(fill="x", pady=5)

        self.model_var = tk.StringVar(value="linear_regression")
        tk.Radiobutton(model_frame, text="多元线性回归", variable=self.model_var,
                       value="linear_regression", command=self._update_params_panel).pack(anchor=tk.W)
        tk.Radiobutton(model_frame, text="决策树回归", variable=self.model_var,
                       value="decision_tree", command=self._update_params_panel).pack(anchor=tk.W)

        self.params_frame = tk.LabelFrame(left_frame, text="模型参数", padx=5, pady=5)
        self.params_frame.pack(fill="x", pady=5)
        self._update_params_panel()

        split_frame = tk.LabelFrame(left_frame, text="数据划分", padx=5, pady=5)
        split_frame.pack(fill="x", pady=5)

        tk.Label(split_frame, text="训练集比例:").grid(row=0, column=0, sticky="w")
        self.split_ratio_var = tk.DoubleVar(value=0.8)
        self.split_slider = tk.Scale(split_frame, from_=0.5, to=0.95, resolution=0.05,
                                     orient=tk.HORIZONTAL, variable=self.split_ratio_var)
        self.split_slider.grid(row=0, column=1, sticky="ew", padx=5)
        self.split_label = tk.Label(split_frame, text="80%")
        self.split_label.grid(row=0, column=2)
        self.split_slider.config(command=lambda val: self.split_label.config(text=f"{float(val)*100:.0f}%"))

        std_frame = tk.LabelFrame(left_frame, text="数据预处理", padx=5, pady=5)
        std_frame.pack(fill="x", pady=5)
        self.standardize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(std_frame, text="标准化特征", variable=self.standardize_var).pack(anchor=tk.W)

    def _update_params_panel(self):
        """更新参数面板"""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        model_type = self.model_var.get()

        if model_type == "linear_regression":
            tk.Label(self.params_frame, text="学习率:").grid(row=0, column=0, sticky="w")
            self.lr_var = tk.DoubleVar(value=0.01)
            tk.Entry(self.params_frame, textvariable=self.lr_var, width=10).grid(row=0, column=1)

            tk.Label(self.params_frame, text="迭代次数:").grid(row=1, column=0, sticky="w")
            self.max_iter_var = tk.IntVar(value=1000)
            tk.Entry(self.params_frame, textvariable=self.max_iter_var, width=10).grid(row=1, column=1)

            tk.Label(self.params_frame, text="正则化系数:").grid(row=2, column=0, sticky="w")
            self.alpha_var = tk.DoubleVar(value=0.0)
            tk.Entry(self.params_frame, textvariable=self.alpha_var, width=10).grid(row=2, column=1)
        else:
            tk.Label(self.params_frame, text="最大深度:").grid(row=0, column=0, sticky="w")
            self.max_depth_var = tk.IntVar(value=5)
            tk.Entry(self.params_frame, textvariable=self.max_depth_var, width=10).grid(row=0, column=1)

            tk.Label(self.params_frame, text="最小样本数:").grid(row=1, column=0, sticky="w")
            self.min_samples_var = tk.IntVar(value=2)
            tk.Entry(self.params_frame, textvariable=self.min_samples_var, width=10).grid(row=1, column=1)

    def _create_right_panel(self):
        """创建右侧结果展示区域"""
        right_frame = tk.Frame(self, padx=10, pady=5)
        right_frame.grid(row=1, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        metrics_frame = tk.LabelFrame(right_frame, text="评估指标", padx=5, pady=5)
        metrics_frame.grid(row=1, column=0, sticky="ew")

        self.metrics_text = tk.Text(metrics_frame, height=6, width=60)
        self.metrics_text.pack(fill="both", expand=True)
        self.metrics_text.insert(tk.END, "评估指标将在此显示\n\n请先加载数据并训练模型")
        self.metrics_text.config(state=tk.DISABLED)

    def _create_bottom_panel(self):
        """创建底部操作按钮区域"""
        bottom_frame = tk.Frame(self, padx=10, pady=5, bg="#f0f0f0")
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        self.train_btn = tk.Button(bottom_frame, text="开始训练", command=self.train_model,
                                   state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        self.export_btn = tk.Button(bottom_frame, text="导出结果", command=self.export_results,
                                    state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(bottom_frame, text="清除", command=self.clear_results)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="就绪")
        self.status_bar = tk.Label(bottom_frame, textvariable=self.status_var,
                                   bg="#f0f0f0", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.RIGHT, fill="x", expand=True, padx=5)

    def load_file(self):
        """加载CSV数据文件"""
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        try:
            self.status_var.set("正在加载数据...")
            self.update()

            self.X, self.y, self.feature_names = self.reader.load_data(file_path)

            self.file_path_var.set(os.path.basename(file_path))
            self.data_info_var.set(f"样本数: {len(self.X)}, 特征数: {len(self.feature_names)}")
            self.train_btn.config(state=tk.NORMAL)

            self.status_var.set("数据加载成功")
            messagebox.showinfo("成功", f"数据加载成功！\n样本数: {len(self.X)}\n特征: {', '.join(self.feature_names)}")

        except Exception as e:
            self.status_var.set("数据加载失败")
            messagebox.showerror("错误", str(e))

    def train_model(self):
        """训练模型"""
        if self.X is None or self.y is None:
            messagebox.showwarning("警告", "请先加载数据")
            return

        try:
            self.status_var.set("正在预处理数据...")
            self.update()

            X_processed = self.X
            if self.standardize_var.get():
                X_processed = self.preprocessor.standardize(self.X)

            test_ratio = 1 - self.split_ratio_var.get()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.preprocessor.split(X_processed, self.y, test_ratio, random_seed=42)

            self.status_var.set("正在训练模型...")
            self.update()

            model_type = self.model_var.get()
            parameters = {}

            if model_type == "linear_regression":
                parameters = {
                    'learning_rate': self.lr_var.get(),
                    'max_iter': self.max_iter_var.get(),
                    'alpha': self.alpha_var.get()
                }
            else:
                parameters = {
                    'max_depth': self.max_depth_var.get(),
                    'min_samples_split': self.min_samples_var.get()
                }

            self.trainer = ModelTrainer(model_type, parameters)
            self.trainer.train(self.X_train, self.y_train)

            self.status_var.set("正在预测...")
            self.update()

            self.y_pred = self.trainer.predict(self.X_test)

            self.status_var.set("正在评估...")
            self.update()

            self.metrics = self.evaluator.evaluate(self.y_test, self.y_pred)

            self._show_results()
            self.export_btn.config(state=tk.NORMAL)
            self.status_var.set("训练完成")

        except Exception as e:
            self.status_var.set("训练失败")
            messagebox.showerror("错误", str(e))

    def _show_results(self):
        """展示结果"""
        self.figure.clear()

        ax1 = self.figure.add_subplot(121)
        ax1.scatter(self.y_test, self.y_pred, alpha=0.6, s=20)
        ax1.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)],
                 'r--', label='y=x')
        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')
        ax1.set_title('真实值 vs 预测值')
        ax1.legend()
        ax1.grid(True)

        residuals = [self.y_test[i] - self.y_pred[i] for i in range(len(self.y_test))]
        ax2 = self.figure.add_subplot(122)
        ax2.hist(residuals, bins=20, edgecolor='black')
        ax2.set_xlabel('残差')
        ax2.set_ylabel('频率')
        ax2.set_title('残差分布图')
        ax2.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        report = "模型评估报告\n"
        report += "=" * 30 + "\n"
        report += f"模型类型: {self.trainer.get_model_type()}\n\n"
        for metric_name, value in self.metrics.items():
            report += f"{metric_name}: {value}\n"
        self.metrics_text.insert(tk.END, report)
        self.metrics_text.config(state=tk.DISABLED)

    def export_results(self):
        """导出结果"""
        if self.metrics is None or self.y_pred is None:
            messagebox.showwarning("警告", "请先训练模型")
            return

        try:
            model_type = self.trainer.get_model_type()
            parameters = self.trainer.get_parameters()

            report_path, predictions_path = self.exporter.export_all(
                self.metrics, model_type, parameters, self.y_test, self.y_pred
            )

            self.status_var.set("导出完成")
            messagebox.showinfo("成功", f"结果导出成功！\n\n报告文件: {report_path}\n预测结果: {predictions_path}")

        except Exception as e:
            self.status_var.set("导出失败")
            messagebox.showerror("错误", str(e))

    def clear_results(self):
        """清除结果"""
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.metrics = None
        self.feature_names = None
        self.trainer = None

        self.file_path_var.set("未选择文件")
        self.data_info_var.set("")
        self.train_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.status_var.set("就绪")

        self.figure.clear()
        self.canvas.draw()

        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "评估指标将在此显示\n\n请先加载数据并训练模型")
        self.metrics_text.config(state=tk.DISABLED)


def main():
    """主函数"""
    app = MainApplication()
    app.mainloop()


if __name__ == "__main__":
    main()