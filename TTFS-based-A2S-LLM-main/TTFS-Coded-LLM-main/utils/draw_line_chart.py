# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np

raw_errors = [1.5903867733868537e-06,
              1.3166751159587875e-05,
              2.1844620277988724e-05,
              4.069610440637916e-05,
              7.295945397345349e-05,
              0.00011530343181220815,
              0.00017609228962101042,
              0.0002504639851395041,
              0.0003360786067787558,
              0.00042454851791262627,
              0.0005180228035897017,
              0.0006062056636437774,
              0.0007107889396138489,
              0.000822074303869158,
              0.000957188312895596,
              0.0011305001098662615,
              0.001436082529835403,
              0.001720400876365602,
              0.0021566192153841257,
              0.002674343064427376,
              0.003350543323904276,
              0.004093581810593605,
              0.00503150001168251,
              0.00598085718229413,
              0.007087983191013336,
              0.008316446095705032,
              0.009872755967080593,
              0.011587845161557198,
              0.013777323998510838,
              0.016542110592126846,
              0.0217074453830719,
              0.051659177988767624
              ]
numLayers = len(raw_errors)
# —— 随机数据（可复现）——
rng = np.random.default_rng(42)

layers = list(range(1, numLayers + 1))
if layers[-1] != numLayers:
    layers.append(numLayers)
x = np.arange(len(layers))

# 两组单调上升的误差（做柱状图）
base = np.linspace(0.01, numLayers / 100, len(layers))
# ann_vs_snn = (base + rng.normal(0, 0.01, len(layers))).clip(min=0.005)
ann_vs_snn = np.array(raw_errors)
# ann_vs_qann = (0.6 * base + rng.normal(0, 0.008, len(layers))).clip(min=0.003)

# “对数曲线”（用 log10 仅用于做样式演示）
log_ann_vs_snn = np.log10(ann_vs_snn)
# log_ann_vs_qann = np.log10(ann_vs_qann)

# —— 配色方案（ColorBrewer Dark2 系）——
palette = {
    "bars1": "#7FA7F7",  # 绿青
    "bars2": "#1e73f2",  # 橙
    "line1": "#EB7FB9",  # 紫
    "line2": "#e7298a",  # 洋红
    "grid": "#bdbdbd",  # 网格浅灰
}

# —— 画图 —— 
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("#fcfcfc")

# 柱状图：叠加到一列（stacked）
width = 0.55
ax.bar(
    x, ann_vs_snn, width=width, alpha=0.95,
    label="Conversion Error", color=palette["bars1"], linewidth=0.8
)
# ax.bar(
#     x, ann_vs_qann, width=width, bottom=ann_vs_snn, alpha=0.85,
#     label="ANN vs. QANN", color=palette["bars2"], linewidth=0.8
# )

# 右侧轴：对数曲线
ax2 = ax.twinx()
ax2.plot(
    x, log_ann_vs_snn, linestyle="--", marker="o", linewidth=2.0,
    label="Log Scaled Conversion Error", color=palette["line1"], zorder=3, markerfacecolor="white"
)
# ax2.plot(
#     x, log_ann_vs_qann, linestyle=":", marker="x", linewidth=2.0,
#     label="ANN vs. QANN (log)", color=palette["line2"], zorder=3
# )

# 轴标签与刻度
ax.set_xlabel("Layer", labelpad=8)
ax.set_ylabel("Conversion Error Curve", labelpad=8)
ax2.set_ylabel("Conversion Error Curve (Log)", labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(layers)

# 网格（仅主轴上的纵向网格）
ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

# 细化外观：更淡的边框
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_color("#d9d9d9")
    ax2.spines[spine].set_color("#d9d9d9")

# 合并图例（主轴+副轴）
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
legend = ax.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.95, facecolor="white")
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_edgecolor("#e0e0e0")

fig.tight_layout()

# 保存图片（可改成本地路径或文件名）
outfile = "stacked_conversion_error_colored.png"
fig.savefig(os.path.join(f"../log/figures/", outfile), dpi=300, bbox_inches="tight")
plt.show()

print("Saved to:", outfile)
