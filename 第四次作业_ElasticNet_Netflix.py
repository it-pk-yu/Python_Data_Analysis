# -*- coding: utf-8 -*-
"""使用 Kaggle Netflix Movies and TV Shows 预测电影时长。
安装：python -m pip install numpy pandas scipy scikit-learn matplotlib threadpoolctl
运行：python 第四次作业_ElasticNet_Netflix.py
优先读取同目录原始CSV；没有时自动联网下载Kaggle第5版。
--data / --out 可覆盖路径。结果写入 第四次作业_results/。
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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

SEED = 42


def split_tags(text):
    """按逗号切分，保留完整标签，例如 International Movies。"""
    return [s.strip() for s in str(text).split(',') if s.strip()]


def count_names(value):
    return 0 if pd.isna(value) else len(split_tags(value))


def prepare(raw):
    required = {'show_id', 'type', 'duration', 'release_year', 'cast',
                'director', 'country', 'listed_in', 'rating'}
    if not required.issubset(raw.columns):
        raise ValueError(f'缺少字段：{required-set(raw.columns)}')
    unique = raw.drop_duplicates(subset='show_id').copy()
    movies = unique.loc[unique.type.eq('Movie')].copy()
    # 只接受明确标为分钟的目标，不能把TV Show的季数混进来。
    duration = pd.to_numeric(movies.duration.astype('string').str.extract(
        r'^\s*(\d+)\s+min\s*$', expand=False), errors='coerce')
    valid = duration.notna() & duration.gt(0)
    info = {'raw_n': len(raw), 'duplicates_removed': len(raw)-len(unique),
            'movie_n': len(movies), 'invalid_duration': int((~valid).sum()),
            'tv_n': int(unique.type.eq('TV Show').sum())}
    movies = movies.loc[valid].copy()
    y = duration.loc[valid].astype(float)
    X = pd.DataFrame(index=movies.index)
    X['release_year'] = pd.to_numeric(movies.release_year, errors='coerce')
    for column in ['cast', 'director']:
        X[column+'_count'] = movies[column].map(count_names).astype(float)
        X[column+'_missing'] = movies[column].isna().astype(float)
    X['country_missing'] = movies.country.isna().astype(float)
    for column in ['country', 'listed_in', 'rating']:
        X[column] = movies[column].fillna('Unknown').astype(str)
    # 没有使用原始duration、标题、ID、简介，避免目标直接进入解释变量。
    info['usable_n'] = len(X)
    info['duration_summary'] = {k: float(v) for k, v in y.describe().items()}
    info['missing_in_valid_movies'] = {c: int(movies[c].isna().sum())
                                       for c in ['cast', 'director', 'country', 'rating']}
    return X, y, movies, info


def build_pipeline():
    numeric = ['release_year', 'cast_count', 'cast_missing', 'director_count',
               'director_missing', 'country_missing']
    # 词表与低频过滤只能从当前训练折学习；多国、多类型均保留多标签信息。
    prep = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numeric),
        ('rating', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                 min_frequency=30, sparse_output=False), ['rating']),
        ('country', CountVectorizer(tokenizer=split_tags, token_pattern=None,
                                   lowercase=False, binary=True, min_df=30), 'country'),
        ('genre', CountVectorizer(tokenizer=split_tags, token_pattern=None,
                                 lowercase=False, binary=True, min_df=10), 'listed_in')
    ], sparse_threshold=0)
    return Pipeline([
        ('features', prep), ('scale', StandardScaler()),
        ('model', ElasticNet(max_iter=100000, tol=1e-6, random_state=SEED))])


def evaluate(y, predicted):
    return {'RMSE': float(np.sqrt(mean_squared_error(y, predicted))),
            'MAE': float(mean_absolute_error(y, predicted)),
            'R2': float(r2_score(y, predicted))}



def ensure_data(path):
    """优先本地CSV；否则获取Kaggle第5版ZIP，仅读取指定CSV成员。"""
    if path.exists():
        return
    import urllib.request
    import io
    import zipfile
    url = 'https://www.kaggle.com/api/v1/datasets/download/shivamb/netflix-shows?datasetVersionNumber=5'
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        payload = urllib.request.urlopen(request, timeout=90).read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_data = archive.read('netflix_titles.csv')
        check = pd.read_csv(io.BytesIO(csv_data))
        if not {'show_id', 'type', 'duration', 'listed_in'}.issubset(check.columns):
            raise ValueError('下载内容不符合Netflix数据格式')
    except Exception as exc:
        raise RuntimeError('自动下载失败。请从 https://www.kaggle.com/datasets/shivamb/netflix-shows '
                           '下载netflix_titles.csv，并用 --data 指定路径。') from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('netflix_titles.csv'))
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / '第四次作业_results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('error', category=ConvergenceWarning)
    ensure_data(args.data)
    raw = pd.read_csv(args.data)
    X, y, movies, info = prepare(raw)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=SEED)
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    grid = {'model__alpha': np.logspace(-2, 1.5, 22),
            'model__l1_ratio': [.1, .5, .9]}
    print('Fitting 66 candidate models with 10-fold CV...', flush=True)
    search = GridSearchCV(build_pipeline(), grid, cv=cv, n_jobs=1,
                          scoring='neg_mean_squared_error', error_score='raise').fit(Xtr, ytr)
    model = search.best_estimator_
    alpha = float(search.best_params_['model__alpha'])
    rho = float(search.best_params_['model__l1_ratio'])
    if alpha in [grid['model__alpha'][0], grid['model__alpha'][-1]]:
        raise RuntimeError('alpha最优值位于边界，请扩展参数网格。')
    predicted = model.predict(Xte)
    base = np.repeat(ytr.mean(), len(yte))
    beta = model.named_steps['model'].coef_
    names = model.named_steps['features'].get_feature_names_out()
    scale = model.named_steps['scale']
    coefficients = pd.DataFrame({'feature': names, 'standardized_coef': beta,
                                 'original_unit_coef': beta / scale.scale_})
    coefficients['abs_coef'] = coefficients.standardized_coef.abs()
    coefficients = coefficients.sort_values('abs_coef', ascending=False)
    coefficients.to_csv(args.out / 'coefficients.csv', index=False)
    original_intercept = float(model.named_steps['model'].intercept_ -
                               np.dot(scale.mean_, beta / scale.scale_))
    info.update({'seed': SEED, 'train_n': len(Xtr), 'test_n': len(Xte),
                 'cv_folds': 10, 'alpha': alpha, 'l1_ratio': rho,
                 'cv_MSE': float(-search.best_score_),
                 'cv_RMSE_from_mean_MSE': float(np.sqrt(-search.best_score_)),
                 'features': len(beta), 'nonzero': int(np.sum(np.abs(beta) > 1e-8)),
                 'baseline': evaluate(yte, base), 'elastic_net': evaluate(yte, predicted),
                 'intercept_standardized': float(model.named_steps['model'].intercept_),
                 'intercept_original': original_intercept,
                 'max_iterations_used': int(model.named_steps['model'].n_iter_)})
    scores = -np.array([search.cv_results_[f'split{k}_test_score'] for k in range(10)]).T
    cv_table = pd.DataFrame({
        'alpha': [p['model__alpha'] for p in search.cv_results_['params']],
        'l1_ratio': [p['model__l1_ratio'] for p in search.cv_results_['params']],
        'mean_MSE': scores.mean(axis=1), 'SE_MSE': scores.std(axis=1, ddof=1)/np.sqrt(10)})
    for k in range(10):
        cv_table[f'fold{k+1}_MSE'] = scores[:, k]
    cv_table.to_csv(args.out / 'cv_results.csv', index=False)
    pd.DataFrame({'show_id': movies.loc[Xte.index, 'show_id'],
                  'actual_minutes': yte, 'predicted_minutes': predicted,
                  'residual': yte-predicted, 'baseline_minutes': base}).to_csv(
        args.out / 'test_predictions.csv', index=False)
    (args.out / 'metrics.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for ratio, group in cv_table.groupby('l1_ratio'):
        ax.plot(group.alpha, np.sqrt(group.mean_MSE), marker='.', label=f'l1_ratio = {ratio}')
    ax.scatter([alpha], [np.sqrt(-search.best_score_)], s=70, c='black', label='Selected')
    ax.set(xscale='log', xlabel='alpha', ylabel='sqrt(mean 10-fold MSE), minutes',
           title='Elastic Net: joint selection of penalty strength and mixture')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(args.out / 'elasticnet_cv.png', dpi=170)
    plt.close(fig)

    top = coefficients.head(18).sort_values('standardized_coef')
    fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
    ax.barh(top.feature, top.standardized_coef,
            color=['#147d92' if v > 0 else '#cc6941' for v in top.standardized_coef])
    ax.axvline(0, color='black', linewidth=.7)
    ax.set(xlabel='Coefficient, minutes per 1 SD', title='Elastic Net: largest standardized coefficients')
    fig.savefig(args.out / 'elasticnet_coefficients.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    axes[0].scatter(yte, predicted, alpha=.25, s=14, color='#147d92')
    limits = [min(yte.min(), predicted.min()), max(yte.max(), predicted.max())]
    axes[0].plot(limits, limits, '--', color='black')
    axes[0].set(xlabel='Actual duration (minutes)', ylabel='Predicted duration (minutes)',
                title='Held-out test predictions')
    axes[1].scatter(predicted, yte-predicted, alpha=.25, s=14, color='#147d92')
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set(xlabel='Predicted duration (minutes)', ylabel='Actual - predicted (minutes)',
                title='Test residuals')
    fig.savefig(args.out / 'elasticnet_diagnostics.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    ax.hist(ytr, bins=40, color='#147d92', alpha=.85)
    ax.set(xlabel='Movie duration (minutes)', ylabel='Training movies',
           title='Duration distribution in the training set')
    fig.savefig(args.out / 'duration_distribution.png', dpi=170)
    plt.close(fig)
    # 验证原始尺度方程与Pipeline预测完全一致。
    raw_features = model.named_steps['features'].transform(Xte)
    np.testing.assert_allclose(raw_features @ (beta/scale.scale_) + original_intercept,
                               predicted, rtol=1e-8, atol=1e-8)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
