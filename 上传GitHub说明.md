# 上传到GitHub说明

## 当前状态

本地Git仓库已初始化并完成首次提交：
- **提交哈希**: 1fa19c1
- **提交文件**: 19个文件，2863行代码
- **分支**: master

## 上传步骤

### 步骤1: 在GitHub上创建新仓库

1. 登录GitHub: https://github.com
2. 点击右上角 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `concrete-strength-prediction` (建议名称，可自定义)
   - **Description**: `基于机器学习的混凝土抗压强度预测系统`
   - **Visibility**: 选择 Public 或 Private
   - **不要勾选** "Add a README file"（我们已有README）
   - **不要勾选** "Add .gitignore"（我们已有.gitignore）
4. 点击 "Create repository"

### 步骤2: 复制仓库URL

创建完成后，GitHub会显示仓库URL，格式类似：
```
https://github.com/你的用户名/concrete-strength-prediction.git
```

### 步骤3: 关联远程仓库并推送

在项目目录下执行以下命令（请将`YOUR_USERNAME`替换为你的GitHub用户名）：

```bash
# 关联远程仓库
git remote add origin https://github.com/YOUR_USERNAME/concrete-strength-prediction.git

# 重命名分支为main（GitHub默认分支名）
git branch -M main

# 推送到GitHub
git push -u origin main
```

### 步骤4: 验证上传

1. 刷新GitHub仓库页面
2. 应该能看到所有项目文件
3. 仓库URL就是你的项目地址

## 一键上传脚本

如果你已经创建好GitHub仓库，可以直接修改下面的URL后执行：

```powershell
# 请将下面的URL替换为你的实际仓库URL
$repoUrl = "https://github.com/YOUR_USERNAME/concrete-strength-prediction.git"

git remote add origin $repoUrl
git branch -M main
git push -u origin main
```

## 认证说明

如果推送时提示需要认证，有以下几种方式：

### 方式1: 使用Personal Access Token (推荐)

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选 `repo` 权限
4. 生成并复制token
5. 推送时，用户名输入GitHub用户名，密码输入token

### 方式2: 使用GitHub Desktop

1. 下载安装 GitHub Desktop: https://desktop.github.com
2. 登录GitHub账号
3. 添加本地仓库并推送

### 方式3: 使用SSH

1. 生成SSH密钥: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. 将公钥添加到GitHub: https://github.com/settings/keys
3. 修改远程URL为SSH格式:
   ```bash
   git remote set-url origin git@github.com:YOUR_USERNAME/concrete-strength-prediction.git
   ```

## 后续更新

如果后续修改了代码，可以使用以下命令更新GitHub仓库：

```bash
# 添加修改的文件
git add .

# 提交修改
git commit -m "更新说明"

# 推送到GitHub
git push
```

## 项目文件结构

```
concrete-strength-prediction/
├── .gitignore                 # Git忽略文件配置
├── README.md                  # 项目说明文档
├── main.py                    # 启动脚本
├── 代码设计思路.md             # 设计思路文档
├── src/                       # 源代码目录
│   ├── __init__.py
│   ├── data_reader.py         # 数据读取模块
│   ├── data_preprocessor.py   # 数据预处理模块
│   ├── linear_regression.py   # 多元线性回归算法
│   ├── decision_tree.py       # 决策树回归算法
│   ├── model_trainer.py       # 模型训练器
│   ├── evaluator.py           # 评估器
│   ├── result_exporter.py     # 结果导出器
│   └── main_app.py            # GUI主界面
├── data/                      # 数据目录
│   └── concrete_data.csv      # 示例数据集
└── docs/                      # 文档目录
    ├── 系统需求分析.md
    ├── 总体设计文档.md
    ├── 详细设计文档.md
    ├── 测试分析报告.md
    └── 用户手册.md
```

## 常见问题

### Q: 推送时提示"failed to push some refs"
**A**: 可能是远程仓库已有内容。执行 `git push -u origin main --force` 强制推送（仅适用于首次推送）。

### Q: 提示"Permission denied"
**A**: 检查GitHub认证是否正确配置，参考上面的认证说明。

### Q: 中文文件名显示乱码
**A**: 执行 `git config --global core.quotepath false` 解决中文文件名显示问题。
