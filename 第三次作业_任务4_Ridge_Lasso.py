# -*- coding: utf-8 -*-
"""第三次作业任务4：Hitters 的 Ridge、Lasso、10折CV、系数路径与1-SE。
安装：python -m pip install numpy pandas scipy scikit-learn matplotlib threadpoolctl
运行：python 第三次作业_任务4_Ridge_Lasso.py
优先读取脚本旁的 Hitters.csv，不存在时自动联网下载；--data 可指定路径。
输出到 第三次作业_results/，并在终端报告RMSE、非零变量数和1-SE讨论。
采用10折CV；截距不惩罚；路径图使用训练集；随机种子固定为42。
"""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

SEED = 42
TOL_ZERO = 1e-8


def evaluate(y, pred):
    return dict(RMSE=float(np.sqrt(mean_squared_error(y, pred))),
                MAE=float(mean_absolute_error(y, pred)), R2=float(r2_score(y, pred)))


def preprocessor(X):
    cat = ['League', 'Division', 'NewLeague']
    num = [c for c in X.columns if c not in cat]
    encode = ColumnTransformer([
        ('numeric', 'passthrough', num),
        ('category', OneHotEncoder(drop='first', handle_unknown='ignore',
                                  sparse_output=False), cat)], verbose_feature_names_out=False)
    return Pipeline([('encode', encode), ('scale', StandardScaler())])



def ensure_data(path):
    """本地文件优先；缺失时从ISLR公开镜像下载，不覆盖已有文件。"""
    if path.exists():
        return
    import urllib.request
    import io
    url = 'https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Hitters.csv'
    try:
        payload = urllib.request.urlopen(url, timeout=90).read()
        check = pd.read_csv(io.BytesIO(payload))
        if not {'Salary', 'AtBat', 'League'}.issubset(check.columns):
            raise ValueError('下载内容不符合Hitters格式')
    except Exception as exc:
        raise RuntimeError('数据下载失败。请手动下载Hitters.csv并用 --data 指定路径：' + url) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('Hitters.csv'))
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / '第三次作业_results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('error', category=ConvergenceWarning)
    ensure_data(args.data)
    raw = pd.read_csv(args.data)
    data = raw.dropna(subset=['Salary']).copy()
    X = data.drop(columns=['Salary', 'rownames'], errors='ignore')
    y = data['Salary']
    if X.isna().any().any():
        raise ValueError('解释变量有缺失，请核对数据。')
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    splits = list(cv.split(Xtr))
    # 两种实现的alpha目标函数归一化不同，分别使用足够宽的网格。
    grids = {'Ridge': np.logspace(-3, 6, 73), 'Lasso': np.logspace(-3, 3, 73)}
    estimators = {'Ridge': Ridge(), 'Lasso': Lasso(max_iter=200000, tol=1e-7)}
    prep = preprocessor(Xtr)
    Ztr = prep.fit_transform(Xtr)
    Zte = prep.transform(Xte)
    names = prep.named_steps['encode'].get_feature_names_out()
    result = {'raw_n': len(raw), 'missing_salary': int(raw.Salary.isna().sum()),
              'n': len(data), 'train_n': len(Xtr), 'test_n': len(Xte),
              'p': len(names), 'seed': SEED, 'cv_folds': 10,
              'baseline': evaluate(yte, np.repeat(ytr.mean(), len(yte))),
              'cv_classes': {}, 'strict_cv': {}}
    coeff = pd.DataFrame({'feature': names})
    preds = pd.DataFrame({'row_index': yte.index, 'actual': yte.values})

    # 按题目指定类直接建模。缩放只使用外层训练集，测试集始终隔离。
    # 但CV类内部无法逐折重新拟合外部缩放器，下面另做严格折内预处理核验。
    for name in estimators:
        if name == 'Ridge':
            fitted = RidgeCV(alphas=grids[name], cv=splits,
                             scoring='neg_mean_squared_error').fit(Ztr, ytr)
        else:
            fitted = LassoCV(alphas=grids[name], cv=splits,
                             max_iter=200000, tol=1e-7).fit(Ztr, ytr)
        result['cv_classes'][name] = {
            'alpha': float(fitted.alpha_),
            'nonzero': int(np.sum(np.abs(fitted.coef_) > TOL_ZERO)),
            **evaluate(yte, fitted.predict(Zte))}
        coeff[name + 'CV'] = fitted.coef_
        preds[name + 'CV'] = fitted.predict(Zte)

    # 严格版本：每个CV训练折独立拟合编码和缩放，避免验证折分布进入预处理。
    for name, estimator in estimators.items():
        print('Fitting fold-wise pipeline:', name, flush=True)
        pipe = Pipeline([('prep', preprocessor(Xtr)), ('model', estimator)])
        search = GridSearchCV(pipe, {'model__alpha': grids[name]}, cv=splits,
                              scoring='neg_mean_squared_error', n_jobs=1,
                              error_score='raise').fit(Xtr, ytr)
        scores = -np.array([search.cv_results_[f'split{k}_test_score'] for k in range(10)]).T
        means = scores.mean(axis=1)
        ses = scores.std(axis=1, ddof=1) / np.sqrt(10)
        alphas = np.array([p['model__alpha'] for p in search.cv_results_['params']])
        best = int(np.argmin(means))
        threshold = means[best] + ses[best]
        eligible = np.flatnonzero(means <= threshold)
        one_se = int(eligible[np.argmax(alphas[eligible])])
        cv_table = pd.DataFrame({'alpha': alphas, 'mean_MSE': means, 'SE_MSE': ses})
        for k in range(10):
            cv_table[f'fold{k+1}_MSE'] = scores[:, k]
        cv_table.to_csv(args.out / f'{name}_cv.csv', index=False)
        models = {}
        for rule, i in [('min', best), ('1se', one_se)]:
            model = clone(pipe).set_params(model__alpha=float(alphas[i])).fit(Xtr, ytr)
            models[rule] = model
            beta = model.named_steps['model'].coef_
            result['strict_cv'][name + '_' + rule] = {
                'alpha': float(alphas[i]), 'cv_MSE': float(means[i]),
                'cv_SE': float(ses[i]), 'threshold': float(threshold),
                'nonzero': int(np.sum(np.abs(beta) > TOL_ZERO)),
                'selected': names[np.abs(beta) > TOL_ZERO].tolist(),
                **evaluate(yte, model.predict(Xte))}
            coeff[name + '_' + rule] = beta
            preds[name + '_' + rule] = model.predict(Xte)
        assert alphas[one_se] >= alphas[best]
        if best in (0, len(alphas)-1):
            raise RuntimeError(f'{name}最优参数位于网格边界，请扩展网格。')

        # 全训练集标准化后的系数路径，不用测试集拟合。
        path = np.array([clone(estimator).set_params(alpha=float(a)).fit(Ztr, ytr).coef_
                         for a in alphas])
        pd.DataFrame(path, columns=names).assign(alpha=alphas).to_csv(
            args.out / f'{name}_path.csv', index=False)
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
        for j, label in enumerate(names):
            ax.plot(alphas, path[:, j], label=label, linewidth=1.2)
        ax.set_xscale('log')
        ax.axvline(alphas[best], color='black', linestyle='--', label='CV minimum')
        ax.axvline(alphas[one_se], color='black', linestyle=':', label='1-SE')
        ax.set(xlabel='alpha (scikit-learn penalty parameter)',
               ylabel='Coefficient (thousand USD per standardized unit)',
               title=f'{name}: coefficient paths on training data')
        ax.legend(loc='center left', bbox_to_anchor=(1, .5), fontsize=8)
        ax.grid(alpha=.2)
        fig.savefig(args.out / f'{name}_path.png', dpi=170)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        ax.plot(alphas, means, color='#147d92')
        ax.fill_between(alphas, means-ses, means+ses, alpha=.15, color='#147d92')
        ax.axhline(threshold, color='#e58935', linestyle='--', label='min MSE + SE at min')
        ax.axvline(alphas[best], color='black', linestyle='--', label='CV minimum')
        ax.axvline(alphas[one_se], color='black', linestyle=':', label='1-SE')
        ax.set(xscale='log', xlabel='alpha', ylabel='10-fold validation MSE',
               title=f'{name}: fold-wise preprocessing and 1-SE rule')
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
        fig.savefig(args.out / f'{name}_cv.png', dpi=170)
        plt.close(fig)

    coeff.to_csv(args.out / 'coefficients.csv', index=False)
    preds.to_csv(args.out / 'test_predictions.csv', index=False)
    (args.out / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n任务4：测试结果与1-SE讨论（RMSE单位：千美元）")
    for name in ['Ridge', 'Lasso']:
        direct = result['cv_classes'][name]
        best = result['strict_cv'][name+'_min']
        simple = result['strict_cv'][name+'_1se']
        print(f"{name}CV：alpha={direct['alpha']:.6f}，测试RMSE={direct['RMSE']:.4f}，"
              f"非零变量数={direct['nonzero']}")
        print(f"严格折内预处理：min模型 alpha={best['alpha']:.6f}，"
              f"RMSE={best['RMSE']:.4f}，非零数={best['nonzero']}；"
              f"1-SE模型 alpha={simple['alpha']:.6f}，RMSE={simple['RMSE']:.4f}，"
              f"非零数={simple['nonzero']}")
        if simple['nonzero'] < best['nonzero']:
            print('1-SE得到更稀疏模型；若重视简洁，可按预先确定的1-SE规则选用。')
        else:
            print('本次1-SE没有减少非零变量数，不能称其更稀疏。')
    print('Ridge通常只收缩而不筛除变量；Lasso可产生零系数。')
    print('1-SE选择仅依据训练集交叉验证；测试结果用于评价，不能反向挑选参数。')
    print('1-SE是经验规则，不是两模型无显著差异的正式统计检验。')
    print('Ridge目标为RSS+alpha*L2平方；Lasso目标为RSS/(2n)+alpha*L1，alpha不可直接横比。')
    print('系数路径图、逐折误差、完整系数和测试预测已写入：', args.out.resolve())



if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
