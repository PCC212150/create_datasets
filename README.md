# create_datasets

甘蔗**根系分割标注**的半自动工具：模型先预测 → 人在图上用蓝/绿笔刷修掩码 →
画折线记根长 → 导出 labelme 标注 + 像素级训练掩码。

代码、文档在 [`root/`](root/)，**详细说明见 [root/readme.md](root/readme.md)**。

## 快速开始

```
git clone <本仓库>
cd root
setup_runtime.bat      # 一次性：装便携运行环境（约 400MB，需联网一次）
run.bat                # 启动
```

克隆下来是**没有数据和权重**的（都在 .gitignore 里），按 `root/readme.md` 补齐：

| 缺什么 | 放哪 | 说明 |
| --- | --- | --- |
| 原图 | `root/pictures/` | 5472×3648 的 jpg |
| 模型权重 | `root/model/<版本名>/<版本名>.pth` | 单文件 124MB，超出 GitHub 上限，只能单独拷 |
| 便携运行环境 | `root/runtime/` | 跑一次 `setup_runtime.bat` 生成 |
| 预测缓存 | `root/cache/<模型名>/` | 跑一次 `run.bat --prefetch` 生成（32 张约 26 秒） |

## 这个工具干什么

- **阶段 1 修掩码**：红色是模型预测的根，蓝笔涂掉错的、绿笔补上漏的；
  `[` `]` 调预测松紧（比一笔笔抠快得多）。最终掩码 = `(红 ∪ 绿) \ 蓝`。
- **阶段 2 画折线**：`G` 照着掩码自动起草（交叉处可以 `B` 断开），拖控制点微调。
- **保存**：`datasets\<名>.json`（labelme）+ `datasets\masks\<名>.png`（黑底白条，
  逐像素的 root 真值，训练优先读它）+ 一张验收用的 overlay。

产物可以直接喂给 `root_model` 训练（布局和它要求的扁平数据集一致）。
