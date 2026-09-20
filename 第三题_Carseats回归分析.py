# -*- coding: utf-8 -*-
"""第二课作业第三题：Carseats 多元线性回归。

安装依赖：python -m pip install numpy pandas scipy statsmodels
直接运行：python 第三题_Carseats回归分析.py
离线运行：python 第三题_Carseats回归分析.py --data Carseats.csv
默认优先读取脚本旁的 Carseats.csv，否则从公开数据源加载。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv"
FORMULA = 'Sales ~ Price + Income + Advertising + C(ShelveLoc, Treatment(reference="Bad"))'


def load_data(path=None):
    """只使用题目指定变量；发现缺失或异常类别时明确报错。"""
    local = Path(__file__).resolve().with_name("Carseats.csv")
    source = path if path is not None else (local if local.exists() else DATA_URL)
    try:
        data = pd.read_csv(source)
    except Exception as exc:
        raise RuntimeError(
            "无法读取数据。请检查网络，或下载 Carseats.csv 后使用 --data 指定文件。"
        ) from exc
    required = ["Sales", "Price", "Income", "Advertising", "ShelveLoc"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"数据缺少字段：{sorted(missing)}")
    data = data[required].copy()
    if data.isna().any().any():
        raise ValueError("题目所需字段存在缺失值，请核对原始数据。")
    for name in required[:-1]:
        data[name] = pd.to_numeric(data[name], errors="raise")
        if not np.isfinite(data[name]).all():
            raise ValueError(f"{name} 含有无穷值。")
    if set(data["ShelveLoc"].unique()) != {"Bad", "Good", "Medium"}:
        raise ValueError("ShelveLoc 应包含 Bad、Good、Medium 三个类别。")
    data["ShelveLoc"] = pd.Categorical(
        data["ShelveLoc"], categories=["Bad", "Good", "Medium"]
    )
    return data


def fit_and_vif(data):
    model = smf.ols(FORMULA, data=data, missing="raise").fit()
    design = model.model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("设计矩阵不满秩，无法进行常规系数解释和 VIF 分析。")
    # 辅助回归保留截距，但截距不是解释变量，不列入 VIF 评价。
    vif = pd.DataFrame([
        {"variable": name, "VIF": variance_inflation_factor(design, i)}
        for i, name in enumerate(model.model.exog_names)
        if name != "Intercept"
    ])
    return model, vif


def main():
    parser = argparse.ArgumentParser(description="Carseats 多元线性回归与 VIF 分析")
    parser.add_argument("--data", type=Path, help="可选：本地 Carseats.csv 路径")
    args = parser.parse_args()
    data = load_data(args.data)
    model, vif = fit_and_vif(data)
    print("样本量：", len(data))
    print("ShelveLoc 类别计数：")
    print(data["ShelveLoc"].value_counts(sort=False).to_string())
    print("\n一、OLS 模型拟合报告")
    print(model.summary())
    print("\n二、基准组与系数解释")
    print("ShelveLoc 基准组：Bad（差）；模型包含截距和 Good、Medium 两个虚拟变量。")
    good_name = next(name for name in model.params.index if name.endswith("[T.Good]"))
    good_coef = model.params[good_name]
    print(f"控制 Price、Income、Advertising 不变，Good 相比 Bad 的预测销量平均高 "
          f"{good_coef:.6f} 千件，约 {good_coef * 1000:.0f} 件。")
    print("Sales 的单位是千件，不是销售金额；该系数表示条件关联，不直接证明因果关系。")
    print("\n三、VIF（在含截距的完整设计矩阵上计算）")
    print(vif.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    largest = vif["VIF"].max()
    if largest < 5:
        print("全部 VIF 小于 5，未发现明显多重共线性风险。")
    elif largest < 10:
        print("存在 VIF 不低于 5 的变量，应进一步检查多重共线性。")
    else:
        print("存在 VIF 不低于 10 的变量，提示较强多重共线性风险。")
    print("5 和 10 是经验阈值。ShelveLoc 以两个虚拟变量分别评价；不对 Sales 计算 VIF。")


if __name__ == "__main__":
    main()
