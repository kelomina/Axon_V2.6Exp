# GA 特征子集搜索实验报告

更新时间：2026-06-20

## 一句话结论

本项目可以加入“遗传算法搜索精简特征子集”的流程，而且目前已经有可运行、可复现、可评估的实现。最稳妥的做法不是重新训练模型，也不是改字节序列分支，而是先把 PE/stat 结构特征当成一排开关：某个特征开着，就按原样输入模型；某个特征关掉，就在输入端把它置零。遗传算法负责反复尝试不同开关组合，目标是在准确率尽量不掉的前提下，关掉更多疑似冗余或噪声特征。

当前推荐的可复用特征掩码是：

- 文件：`config/feature_masks/ga_recall_guard_2000.json`
- 有效搜索特征总数：192
- 保留特征：125
- 保留 PE 特征：95
- 保留 stat 特征：30

也就是说，它从 192 个真正参与搜索的结构特征里去掉了 67 个。默认 fixed_v2 PE 特征配置里还有 113 个已知补零维度，GA 默认不搜索这些维度，因为它们本来就不是有效信号。

## 为什么这件事有价值

可以把模型的结构特征想象成一张很大的控制台。每个按钮都可能提供一点判断线索，但有些按钮可能只是重复别人已经提供的信息，有些按钮甚至会制造干扰。GA 本身不“理解病毒”，它只是像反复试配方一样，不断尝试不同按钮组合：如果某个组合判断效果好，又比原来少看很多按钮，就把它留下继续繁殖和微调。

这件事的业务价值是：我们可以在不推倒重来模型的情况下，找出一组更短、更干净的结构特征输入，减少噪声和冗余，让后续模型评估、线上推理、特征解释都更容易控制。

## 关键证据

目前最强的验证是 20,000 样本评估，每类 10,000 个样本。

复现实验命令：

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\evaluate_feature_mask.py --checkpoint models\best_model.pt --data-dir data --feature-mask config\feature_masks\ga_recall_guard_2000.json --samples-per-class 10000 --batch-size 256 --device cuda --thresholds "0.45,0.475,0.50,0.525,0.55,0.575,0.60,0.625,0.65" --baseline-threshold 0.50 --output-json reports\feature_mask_eval_all20000_thresholds.json
```

完整特征基线，阈值 0.50：

- F1：0.931013
- 总错误数：1340
- 误报 / 漏报：382 / 958

完整特征在本轮阈值扫描里的最佳结果，阈值 0.45：

- F1：0.937077
- 总错误数：1237
- 误报 / 漏报：448 / 789
- 相比完整特征 @ 0.50：F1 增加 0.006064，总错误少 103 个，漏报少 169 个，误报多 66 个

GA 掩码的最佳 F1，阈值 0.50：

- F1：0.939163
- 总错误数：1216
- 误报 / 漏报：602 / 614
- 相比完整特征 @ 0.50：F1 增加 0.008150，总错误少 124 个，漏报少 344 个，误报多 220 个

GA 掩码的最低总错误数，阈值 0.525：

- F1：0.939104
- 总错误数：1210
- 误报 / 漏报：540 / 670
- 相比完整特征 @ 0.50：F1 增加 0.008091，总错误少 130 个，漏报少 288 个，误报多 158 个

## 按来源目录的补充检查

我还用 `manifest_7792a33b.json` 里的原始路径做了一轮来源目录分组。这个分组不是严格的病毒家族标签，但它能回答一个很实际的问题：GA 掩码的收益是不是只集中在某一个目录。

对比完整特征 @ `0.50` 和 GA 掩码 @ `0.525`：

| 来源目录 | 样本数 | 完整特征误报/漏报 | GA 掩码误报/漏报 | 总错误变化 |
| --- | ---: | ---: | ---: | ---: |
| `benign/待加入白名单` | 10000 | 382 / 0 | 540 / 0 | +158 |
| `malicious/2020-02` | 309 | 0 / 5 | 0 / 3 | -2 |
| `malicious/2020-03` | 1949 | 0 / 307 | 0 / 177 | -130 |
| `malicious/2020-06` | 7742 | 0 / 646 | 0 / 490 | -156 |

这个结果说明：GA 掩码不是只在一个恶意目录上“碰巧变好”，三个恶意来源目录的漏报都减少了；代价也很明确，白名单误报增加了 158 个。因此这不是一个“无成本提升”，而是一个更偏向减少漏报的特征精简方案。

## 当前推荐

推荐怎么用，取决于业务更怕哪种错误：

- 如果最重视 F1 和少漏报，优先看 GA 掩码 @ `0.50`。
- 如果最重视总错误数最低，优先看 GA 掩码 @ `0.525`。
- 如果误报成本很高，希望保守一点，可以看 GA 掩码 @ `0.55`。

当前我更推荐 GA 掩码 @ `0.525` 作为候选方案，因为它在 20,000 样本阈值扫描里总错误数最低，同时 F1 仍然明显高于完整特征基线。

## 已落地的代码

当前已经提交到仓库的内容：

- `scripts/search_feature_subset_ga.py`：用 GA 搜索 PE/stat 特征子集。
- `scripts/evaluate_feature_mask.py`：评估已导出的特征掩码，并自动生成阈值摘要。
- `src/feature_mask.py`：加载和应用可复用特征掩码。
- `config/feature_masks/ga_recall_guard_2000.json`：当前推荐的可复用掩码。
- `tests/test_feature_subset_ga.py`：GA 相关单元测试。
- `tests/test_feature_mask.py`：特征掩码和评估摘要单元测试。

## 复现步骤

运行测试：

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" -m pytest tests\test_feature_mask.py tests\test_feature_subset_ga.py -q
```

运行一个短时间 GA 冒烟搜索：

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\search_feature_subset_ga.py --checkpoint models\best_model.pt --data-dir data --split all --samples-per-class 80 --batch-size 64 --max-batches 3 --device cuda --holdout-ratio 0.25 --population-size 8 --generations 3 --elite-size 2 --target-keep-ratio 0.55 --min-pe-features 20 --min-stat-features 5 --objective f1 --feature-penalty 0.01 --max-objective-drop 0.02 --max-recall-drop 0.03 --max-fn-increase-rate 0.02 --thresholds "0.45,0.50,0.55" --seed 20260620 --top-k 5 --output-json reports\ga_debug_after_push.json
```

运行 20,000 样本特征掩码评估：

```powershell
& "E:\Project\python\Axon_v2.6Exp\vnev\Scripts\python.exe" scripts\evaluate_feature_mask.py --checkpoint models\best_model.pt --data-dir data --feature-mask config\feature_masks\ga_recall_guard_2000.json --samples-per-class 10000 --batch-size 256 --device cuda --thresholds "0.45,0.475,0.50,0.525,0.55,0.575,0.60,0.625,0.65" --baseline-threshold 0.50 --output-json reports\feature_mask_eval_all20000_thresholds.json
```

## 仍需谨慎的地方

这组结果已经足够支持我们把 GA 特征筛选作为正式实验工具保留下来，也足够支持把当前掩码作为候选方案继续验证。但它还不应该不经确认就变成所有线上推理的默认开关。

正式默认启用前，建议继续确认三件事：

- 在更严格的时间切分或真实家族标签切分上，当前掩码是否仍然稳定。
- 误报增加是否符合产品流程的承受范围。
- 下游推理脚本是否需要正式暴露 `--feature-mask` 参数，让产品化调用更顺手。
