"""
Resampling, Model Selection and Regularization with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_data
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# ----------------- Step 1: 数据加载与概览 -----------------
def load_data():
    """从 sklearn 加载糖尿病数据集 (DataFrame 格式)。"""
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    return X, y

def describe_data(X, y):
    """返回样本量、特征数、特征列表与目标均值。"""
    return {
        'n': len(X),
        'p': X.shape[1],
        'features': list(X.columns),
        'y_mean': round(float(y.mean()), 2)
    }

# ----------------- Step 2: 划分训练集与测试集 -----------------
def train_test(X, y, test_size=0.25, random_state=0):
    """一次性保留测试集，后续模型选择与调优只在训练集上进行。"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ----------------- Step 3: 单次验证集切分曲线与波动评估 -----------------
def poly_model(degree):
    """多项式回归 Pipeline。"""
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

def validation_set_curve(X, y, feature, degrees, random_state):
    """使用 50/50 单次切分评估不同多项式阶数的验证集 MSE。"""
    X_sub = X[[feature]]
    X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y, test_size=0.5, random_state=random_state)
    curve = {}
    for d in degrees:
        m = poly_model(d).fit(X_tr, y_tr)
        pred = m.predict(X_val)
        curve[d] = round(float(mean_squared_error(y_val, pred)), 1)
    return curve

def curve_spread(X, y, feature, degrees, seeds):
    """统计不同随机切分下验证集 MSE 的极差（最大值减最小值）。"""
    curves = [validation_set_curve(X, y, feature, degrees, s) for s in seeds]
    spread = {}
    for d in degrees:
        vals = [c[d] for c in curves]
        spread[d] = round(float(max(vals) - min(vals)), 1)
    return spread

# ----------------- Step 4: K 折交叉验证与留一法 (LOOCV) -----------------
def cv_mse(estimator, X, y, k=5, random_state=0):
    """计算 K 折交叉验证 MSE 的均值与标准误。"""
    cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = -cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_squared_error')
    mean_val = float(np.mean(scores))
    se_val = float(np.std(scores, ddof=1) / np.sqrt(k))
    return round(mean_val, 2), round(se_val, 2)

def loocv_mse(estimator, X, y):
    """留一交叉验证 (LOOCV) MSE。"""
    loo = LeaveOneOut()
    scores = -cross_val_score(estimator, X, y, cv=loo, scoring='neg_mean_squared_error')
    return round(float(np.mean(scores)), 2)

# ----------------- Step 5: 不同 K 值折划分随机性对比 -----------------
def cv_spread_by_k(estimator, X, y, ks, seeds):
    """对比不同 K 值在不同随机种子下的均值与标准差。"""
    spread = {}
    for k in ks:
        estimates = [cv_mse(estimator, X, y, k=k, random_state=s)[0] for s in seeds]
        mean_val = round(float(np.mean(estimates)), 1)
        std_val = round(float(np.std(estimates, ddof=1)), 1)
        spread[k] = (mean_val, std_val)
    return spread

def compare_with_loocv(estimator, X, y, ks, seeds):
    """对比 K 折交叉验证与无随机性的 LOOCV。"""
    return {
        'loocv': loocv_mse(estimator, X, y),
        'kfold': cv_spread_by_k(estimator, X, y, ks, seeds)
    }

# ----------------- Step 6: 自助法 (Bootstrap) 评估回归系数标准误 -----------------
def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    """有放回抽样拟合 OLS，提取回归系数分布。"""
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    coefs = []
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = LinearRegression().fit(X_arr[idx], y_arr[idx])
        coefs.append(m.coef_)
    return np.array(coefs)

def bootstrap_se(coefs):
    """计算 Bootstrap 回归系数标准差。"""
    return np.round(np.std(coefs, axis=0, ddof=1), 2)

def ols_standard_errors(X, y):
    """利用解析公式计算 OLS 斜率系数的标准误。"""
    n, p = X.shape
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    A = np.column_stack([np.ones(n), X_arr])
    beta = np.linalg.lstsq(A, y_arr, rcond=None)[0]
    residuals = y_arr - A @ beta
    rss = np.sum(residuals ** 2)
    sigma2 = rss / (n - p - 1)
    inv_ata = np.linalg.inv(A.T @ A)
    se = np.sqrt(sigma2 * np.diag(inv_ata))[1:]
    return np.round(se, 2)

# ----------------- Step 7: 逐步特征选择路径 (Stepwise Selection) -----------------
def select_features(X, y, k, direction, cv):
    """序列特征选择器，返回指定特征数量的特征名。"""
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction=direction, cv=cv, scoring='neg_mean_squared_error')
    sfs.fit(X, y)
    return list(X.columns[sfs.get_support()])

def stepwise_path(X, y, direction, cv):
    """生成从 1 个特征到全特征的逐步选择路径字典。"""
    p = X.shape[1]
    path = {}
    for k in range(1, p):
        path[k] = select_features(X, y, k, direction, cv)
    path[p] = list(X.columns)
    return path

# ----------------- Step 8: 特征子集路径评估 -----------------
def score_path(X, y, path, cv):
    """评估路径上每个尺寸特征集的交叉验证 MSE 及其标准误。"""
    sizes = sorted(path.keys())
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for k in sizes:
        feats = path[k]
        scores = -cross_val_score(LinearRegression(), X[feats], y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return sizes, means, ses

def best_size(sizes, means):
    """返回 MSE 最小时对应的特征数量。"""
    idx = int(np.argmin(means))
    return sizes[idx]

# ----------------- Step 9: 1-SE 准则与子集决策 -----------------
def one_se_rule(values, means, ses, prefer='smaller'):
    """应用 1-SE 准则在可容忍阈值内挑选更简化的超参数。"""
    i_min = int(np.argmin(means))
    threshold = means[i_min] + ses[i_min]
    eligible = [v for v, m in zip(values, means) if m <= threshold]
    return min(eligible) if prefer == 'smaller' else max(eligible)

def choose_subset(X, y, direction, cv):
    """使用 1-SE 准则选择最优前向特征子集。"""
    path = stepwise_path(X, y, direction, cv)
    sizes, means, ses = score_path(X, y, path, cv)
    size_min = best_size(sizes, means)
    size_1se = one_se_rule(sizes, means, ses, prefer='smaller')
    return size_min, size_1se, path[size_1se]

# ----------------- Step 10: 岭回归 (Ridge) 正则化路径 -----------------
def ridge_model(alpha):
    """标准化 + 岭回归 Pipeline。"""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def ridge_path(X, y, alphas):
    """提取不同 alpha 惩罚下的岭回归系数矩阵。"""
    coefs = []
    for a in alphas:
        m = ridge_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def coef_norms(path):
    """计算每组系数向量的 L2 范数。"""
    return np.round(np.linalg.norm(path, axis=1), 2)

# ----------------- Step 11: 交叉验证超参数曲线 -----------------
def cv_curve(make_model, X, y, values, cv):
    """计算模型随网格参数变化的 CV 均值和标准误。"""
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for val in values:
        m = make_model(val)
        scores = -cross_val_score(m, X, y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return means, ses

def choose_penalty(make_model, X, y, values, cv):
    """挑选岭回归惩罚项（1-SE 偏好更大的正则化系数）。"""
    means, ses = cv_curve(make_model, X, y, values, cv)
    v_min = values[int(np.argmin(means))]
    v_1se = one_se_rule(values, means, ses, prefer='larger')
    return v_min, v_1se

# ----------------- Step 12: Lasso 稀疏化路径与 LassoCV -----------------
def lasso_model(alpha):
    """标准化 + Lasso Pipeline。"""
    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))

def lasso_path(X, y, alphas):
    """提取 Lasso 系数矩阵。"""
    coefs = []
    for a in alphas:
        m = lasso_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def nonzero_features(coef, names, tol=1e-8):
    """提取绝对值大于容差的非零特征名。"""
    return [name for name, c in zip(names, coef) if abs(c) > tol]

def lasso_cv(X, y, cv):
    """使用内置 LassoCV 寻优并返回最佳 alpha 与保留的稀疏特征。"""
    pipe = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0))
    pipe.fit(X, y)
    alpha = round(float(pipe[-1].alpha_), 4)
    selected = nonzero_features(pipe[-1].coef_, list(X.columns))
    return alpha, selected

# ----------------- Step 13: 主成分回归 (PCR) -----------------
def pcr_model(n_components):
    """标准化 + PCA 降维 + OLS 回归。"""
    return make_pipeline(StandardScaler(), PCA(n_components=n_components), LinearRegression())

def explained_variance(X):
    """计算累计解释方差比。"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA().fit(X_scaled)
    return np.round(np.cumsum(pca.explained_variance_ratio_), 3)

def pcr_curve(X, y, cv):
    """计算 1 到 p 个主成分下的 PCR 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pcr_model, X, y, comps, cv)
    return comps, means, ses

# ----------------- Step 14: 偏最小二乘回归 (PLS) -----------------
def pls_model(n_components):
    """监督式降维 PLS 回归。"""
    return PLSRegression(n_components=n_components, scale=True)

def pls_curve(X, y, cv):
    """计算 PLS 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pls_model, X, y, comps, cv)
    return comps, means, ses

def pls_predict(model, X):
    """将 PLS 预测的列矩阵压平为 1-D 数组。"""
    return np.asarray(model.predict(X)).ravel()

def best_components(components, means, ses):
    """选择主成分数量（1-SE 偏好更少成分）。"""
    i_min = int(np.argmin(means))
    m_min = components[i_min]
    m_1se = one_se_rule(components, means, ses, prefer='smaller')
    return m_min, m_1se

# ----------------- Step 15: 统一拟合全家桶模型 -----------------
def fit_all(X, y, cv, alphas):
    """统一拟合 6 种方法在各自交叉验证最优设置下的模型。"""
    models = {}
    all_cols = list(X.columns)
    
    # 1. ols
    models['ols'] = (LinearRegression().fit(X, y), all_cols, None)
    
    # 2. forward_1se
    _, size_1se, f_feats = choose_subset(X, y, 'forward', cv)
    m_fwd = LinearRegression().fit(X[f_feats], y)
    models['forward_1se'] = (m_fwd, f_feats, size_1se)
    
    # 3. ridge_1se
    _, r_alpha_1se = choose_penalty(ridge_model, X, y, alphas, cv)
    m_ridge = ridge_model(r_alpha_1se).fit(X, y)
    models['ridge_1se'] = (m_ridge, all_cols, r_alpha_1se)
    
    # 4. lasso_cv
    p_lasso = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0)).fit(X, y)
    l_alpha = round(float(p_lasso[-1].alpha_), 4)
    models['lasso_cv'] = (p_lasso, all_cols, l_alpha)
    
    # 5. pcr_1se
    comps, pm, ps = pcr_curve(X, y, cv)
    _, pcr_c_1se = best_components(comps, pm, ps)
    m_pcr = pcr_model(pcr_c_1se).fit(X, y)
    models['pcr_1se'] = (m_pcr, all_cols, pcr_c_1se)
    
    # 6. pls_1se
    comps, lm, ls = pls_curve(X, y, cv)
    _, pls_c_1se = best_components(comps, lm, ls)
    m_pls = pls_model(pls_c_1se).fit(X, y)
    models['pls_1se'] = (m_pls, all_cols, pls_c_1se)
    
    return models

# ----------------- Step 16: 最终测试集统一评估与制表 -----------------
def test_report(models, X_test, y_test):
    """打开测试集计算各个模型的最终 RMSE、有效特征数与参数设置。"""
    report = {}
    for name, (m, feats, setting) in models.items():
        if name == 'pls_1se':
            preds = pls_predict(m, X_test[feats])
        else:
            preds = np.asarray(m.predict(X_test[feats])).ravel()
            
        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 1)
        n_feats = len(nonzero_features(m[-1].coef_, feats)) if name == 'lasso_cv' else len(feats)
            
        report[name] = {
            'rmse': rmse,
            'n_features': n_feats,
            'setting': setting
        }
    return report

def best_method(report):
    """选出测试集上 RMSE 最低的方法。"""
    return min(report, key=lambda k: report[k]['rmse'])

def format_table(report):
    """按 RMSE 从低到高排版输出格式化结果行。"""
    lines = []
    for name in sorted(report, key=lambda k: report[k]['rmse']):
        r = report[name]
        lines.append(f"{name:12s} rmse={r['rmse']:6.1f} features={r['n_features']:2d} setting={r['setting']}")
    return lines

# Step 2 - train_test
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# ----------------- Step 1: 数据加载与概览 -----------------
def load_data():
    """从 sklearn 加载糖尿病数据集 (DataFrame 格式)。"""
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    return X, y

def describe_data(X, y):
    """返回样本量、特征数、特征列表与目标均值。"""
    return {
        'n': len(X),
        'p': X.shape[1],
        'features': list(X.columns),
        'y_mean': round(float(y.mean()), 2)
    }

# ----------------- Step 2: 划分训练集与测试集 -----------------
def train_test(X, y, test_size=0.25, random_state=0):
    """一次性保留测试集，后续模型选择与调优只在训练集上进行。"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ----------------- Step 3: 单次验证集切分曲线与波动评估 -----------------
def poly_model(degree):
    """多项式回归 Pipeline。"""
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

def validation_set_curve(X, y, feature, degrees, random_state):
    """使用 50/50 单次切分评估不同多项式阶数的验证集 MSE。"""
    X_sub = X[[feature]]
    X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y, test_size=0.5, random_state=random_state)
    curve = {}
    for d in degrees:
        m = poly_model(d).fit(X_tr, y_tr)
        pred = m.predict(X_val)
        curve[d] = round(float(mean_squared_error(y_val, pred)), 1)
    return curve

def curve_spread(X, y, feature, degrees, seeds):
    """统计不同随机切分下验证集 MSE 的极差（最大值减最小值）。"""
    curves = [validation_set_curve(X, y, feature, degrees, s) for s in seeds]
    spread = {}
    for d in degrees:
        vals = [c[d] for c in curves]
        spread[d] = round(float(max(vals) - min(vals)), 1)
    return spread

# ----------------- Step 4: K 折交叉验证与留一法 (LOOCV) -----------------
def cv_mse(estimator, X, y, k=5, random_state=0):
    """计算 K 折交叉验证 MSE 的均值与标准误。"""
    cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = -cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_squared_error')
    mean_val = float(np.mean(scores))
    se_val = float(np.std(scores, ddof=1) / np.sqrt(k))
    return round(mean_val, 2), round(se_val, 2)

def loocv_mse(estimator, X, y):
    """留一交叉验证 (LOOCV) MSE。"""
    loo = LeaveOneOut()
    scores = -cross_val_score(estimator, X, y, cv=loo, scoring='neg_mean_squared_error')
    return round(float(np.mean(scores)), 2)

# ----------------- Step 5: 不同 K 值折划分随机性对比 -----------------
def cv_spread_by_k(estimator, X, y, ks, seeds):
    """对比不同 K 值在不同随机种子下的均值与标准差。"""
    spread = {}
    for k in ks:
        estimates = [cv_mse(estimator, X, y, k=k, random_state=s)[0] for s in seeds]
        mean_val = round(float(np.mean(estimates)), 1)
        std_val = round(float(np.std(estimates, ddof=1)), 1)
        spread[k] = (mean_val, std_val)
    return spread

def compare_with_loocv(estimator, X, y, ks, seeds):
    """对比 K 折交叉验证与无随机性的 LOOCV。"""
    return {
        'loocv': loocv_mse(estimator, X, y),
        'kfold': cv_spread_by_k(estimator, X, y, ks, seeds)
    }

# ----------------- Step 6: 自助法 (Bootstrap) 评估回归系数标准误 -----------------
def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    """有放回抽样拟合 OLS，提取回归系数分布。"""
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    coefs = []
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = LinearRegression().fit(X_arr[idx], y_arr[idx])
        coefs.append(m.coef_)
    return np.array(coefs)

def bootstrap_se(coefs):
    """计算 Bootstrap 回归系数标准差。"""
    return np.round(np.std(coefs, axis=0, ddof=1), 2)

def ols_standard_errors(X, y):
    """利用解析公式计算 OLS 斜率系数的标准误。"""
    n, p = X.shape
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    A = np.column_stack([np.ones(n), X_arr])
    beta = np.linalg.lstsq(A, y_arr, rcond=None)[0]
    residuals = y_arr - A @ beta
    rss = np.sum(residuals ** 2)
    sigma2 = rss / (n - p - 1)
    inv_ata = np.linalg.inv(A.T @ A)
    se = np.sqrt(sigma2 * np.diag(inv_ata))[1:]
    return np.round(se, 2)

# ----------------- Step 7: 逐步特征选择路径 (Stepwise Selection) -----------------
def select_features(X, y, k, direction, cv):
    """序列特征选择器，返回指定特征数量的特征名。"""
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction=direction, cv=cv, scoring='neg_mean_squared_error')
    sfs.fit(X, y)
    return list(X.columns[sfs.get_support()])

def stepwise_path(X, y, direction, cv):
    """生成从 1 个特征到全特征的逐步选择路径字典。"""
    p = X.shape[1]
    path = {}
    for k in range(1, p):
        path[k] = select_features(X, y, k, direction, cv)
    path[p] = list(X.columns)
    return path

# ----------------- Step 8: 特征子集路径评估 -----------------
def score_path(X, y, path, cv):
    """评估路径上每个尺寸特征集的交叉验证 MSE 及其标准误。"""
    sizes = sorted(path.keys())
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for k in sizes:
        feats = path[k]
        scores = -cross_val_score(LinearRegression(), X[feats], y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return sizes, means, ses

def best_size(sizes, means):
    """返回 MSE 最小时对应的特征数量。"""
    idx = int(np.argmin(means))
    return sizes[idx]

# ----------------- Step 9: 1-SE 准则与子集决策 -----------------
def one_se_rule(values, means, ses, prefer='smaller'):
    """应用 1-SE 准则在可容忍阈值内挑选更简化的超参数。"""
    i_min = int(np.argmin(means))
    threshold = means[i_min] + ses[i_min]
    eligible = [v for v, m in zip(values, means) if m <= threshold]
    return min(eligible) if prefer == 'smaller' else max(eligible)

def choose_subset(X, y, direction, cv):
    """使用 1-SE 准则选择最优前向特征子集。"""
    path = stepwise_path(X, y, direction, cv)
    sizes, means, ses = score_path(X, y, path, cv)
    size_min = best_size(sizes, means)
    size_1se = one_se_rule(sizes, means, ses, prefer='smaller')
    return size_min, size_1se, path[size_1se]

# ----------------- Step 10: 岭回归 (Ridge) 正则化路径 -----------------
def ridge_model(alpha):
    """标准化 + 岭回归 Pipeline。"""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def ridge_path(X, y, alphas):
    """提取不同 alpha 惩罚下的岭回归系数矩阵。"""
    coefs = []
    for a in alphas:
        m = ridge_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def coef_norms(path):
    """计算每组系数向量的 L2 范数。"""
    return np.round(np.linalg.norm(path, axis=1), 2)

# ----------------- Step 11: 交叉验证超参数曲线 -----------------
def cv_curve(make_model, X, y, values, cv):
    """计算模型随网格参数变化的 CV 均值和标准误。"""
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for val in values:
        m = make_model(val)
        scores = -cross_val_score(m, X, y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return means, ses

def choose_penalty(make_model, X, y, values, cv):
    """挑选岭回归惩罚项（1-SE 偏好更大的正则化系数）。"""
    means, ses = cv_curve(make_model, X, y, values, cv)
    v_min = values[int(np.argmin(means))]
    v_1se = one_se_rule(values, means, ses, prefer='larger')
    return v_min, v_1se

# ----------------- Step 12: Lasso 稀疏化路径与 LassoCV -----------------
def lasso_model(alpha):
    """标准化 + Lasso Pipeline。"""
    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))

def lasso_path(X, y, alphas):
    """提取 Lasso 系数矩阵。"""
    coefs = []
    for a in alphas:
        m = lasso_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def nonzero_features(coef, names, tol=1e-8):
    """提取绝对值大于容差的非零特征名。"""
    return [name for name, c in zip(names, coef) if abs(c) > tol]

def lasso_cv(X, y, cv):
    """使用内置 LassoCV 寻优并返回最佳 alpha 与保留的稀疏特征。"""
    pipe = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0))
    pipe.fit(X, y)
    alpha = round(float(pipe[-1].alpha_), 4)
    selected = nonzero_features(pipe[-1].coef_, list(X.columns))
    return alpha, selected

# ----------------- Step 13: 主成分回归 (PCR) -----------------
def pcr_model(n_components):
    """标准化 + PCA 降维 + OLS 回归。"""
    return make_pipeline(StandardScaler(), PCA(n_components=n_components), LinearRegression())

def explained_variance(X):
    """计算累计解释方差比。"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA().fit(X_scaled)
    return np.round(np.cumsum(pca.explained_variance_ratio_), 3)

def pcr_curve(X, y, cv):
    """计算 1 到 p 个主成分下的 PCR 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pcr_model, X, y, comps, cv)
    return comps, means, ses

# ----------------- Step 14: 偏最小二乘回归 (PLS) -----------------
def pls_model(n_components):
    """监督式降维 PLS 回归。"""
    return PLSRegression(n_components=n_components, scale=True)

def pls_curve(X, y, cv):
    """计算 PLS 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pls_model, X, y, comps, cv)
    return comps, means, ses

def pls_predict(model, X):
    """将 PLS 预测的列矩阵压平为 1-D 数组。"""
    return np.asarray(model.predict(X)).ravel()

def best_components(components, means, ses):
    """选择主成分数量（1-SE 偏好更少成分）。"""
    i_min = int(np.argmin(means))
    m_min = components[i_min]
    m_1se = one_se_rule(components, means, ses, prefer='smaller')
    return m_min, m_1se

# ----------------- Step 15: 统一拟合全家桶模型 -----------------
def fit_all(X, y, cv, alphas):
    """统一拟合 6 种方法在各自交叉验证最优设置下的模型。"""
    models = {}
    all_cols = list(X.columns)
    
    # 1. ols
    models['ols'] = (LinearRegression().fit(X, y), all_cols, None)
    
    # 2. forward_1se
    _, size_1se, f_feats = choose_subset(X, y, 'forward', cv)
    m_fwd = LinearRegression().fit(X[f_feats], y)
    models['forward_1se'] = (m_fwd, f_feats, size_1se)
    
    # 3. ridge_1se
    _, r_alpha_1se = choose_penalty(ridge_model, X, y, alphas, cv)
    m_ridge = ridge_model(r_alpha_1se).fit(X, y)
    models['ridge_1se'] = (m_ridge, all_cols, r_alpha_1se)
    
    # 4. lasso_cv
    p_lasso = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0)).fit(X, y)
    l_alpha = round(float(p_lasso[-1].alpha_), 4)
    models['lasso_cv'] = (p_lasso, all_cols, l_alpha)
    
    # 5. pcr_1se
    comps, pm, ps = pcr_curve(X, y, cv)
    _, pcr_c_1se = best_components(comps, pm, ps)
    m_pcr = pcr_model(pcr_c_1se).fit(X, y)
    models['pcr_1se'] = (m_pcr, all_cols, pcr_c_1se)
    
    # 6. pls_1se
    comps, lm, ls = pls_curve(X, y, cv)
    _, pls_c_1se = best_components(comps, lm, ls)
    m_pls = pls_model(pls_c_1se).fit(X, y)
    models['pls_1se'] = (m_pls, all_cols, pls_c_1se)
    
    return models

# ----------------- Step 16: 最终测试集统一评估与制表 -----------------
def test_report(models, X_test, y_test):
    """打开测试集计算各个模型的最终 RMSE、有效特征数与参数设置。"""
    report = {}
    for name, (m, feats, setting) in models.items():
        if name == 'pls_1se':
            preds = pls_predict(m, X_test[feats])
        else:
            preds = np.asarray(m.predict(X_test[feats])).ravel()
            
        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 1)
        n_feats = len(nonzero_features(m[-1].coef_, feats)) if name == 'lasso_cv' else len(feats)
            
        report[name] = {
            'rmse': rmse,
            'n_features': n_feats,
            'setting': setting
        }
    return report

def best_method(report):
    """选出测试集上 RMSE 最低的方法。"""
    return min(report, key=lambda k: report[k]['rmse'])

def format_table(report):
    """按 RMSE 从低到高排版输出格式化结果行。"""
    lines = []
    for name in sorted(report, key=lambda k: report[k]['rmse']):
        r = report[name]
        lines.append(f"{name:12s} rmse={r['rmse']:6.1f} features={r['n_features']:2d} setting={r['setting']}")
    return lines

# Step 3 - validation_set_curve
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# ----------------- Step 1: 数据加载与概览 -----------------
def load_data():
    """从 sklearn 加载糖尿病数据集 (DataFrame 格式)。"""
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    return X, y

def describe_data(X, y):
    """返回样本量、特征数、特征列表与目标均值。"""
    return {
        'n': len(X),
        'p': X.shape[1],
        'features': list(X.columns),
        'y_mean': round(float(y.mean()), 2)
    }

# ----------------- Step 2: 划分训练集与测试集 -----------------
def train_test(X, y, test_size=0.25, random_state=0):
    """一次性保留测试集，后续模型选择与调优只在训练集上进行。"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ----------------- Step 3: 单次验证集切分曲线与波动评估 -----------------
def poly_model(degree):
    """多项式回归 Pipeline。"""
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

def validation_set_curve(X, y, feature, degrees, random_state):
    """使用 50/50 单次切分评估不同多项式阶数的验证集 MSE。"""
    X_sub = X[[feature]]
    X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y, test_size=0.5, random_state=random_state)
    curve = {}
    for d in degrees:
        m = poly_model(d).fit(X_tr, y_tr)
        pred = m.predict(X_val)
        curve[d] = round(float(mean_squared_error(y_val, pred)), 1)
    return curve

def curve_spread(X, y, feature, degrees, seeds):
    """统计不同随机切分下验证集 MSE 的极差（最大值减最小值）。"""
    curves = [validation_set_curve(X, y, feature, degrees, s) for s in seeds]
    spread = {}
    for d in degrees:
        vals = [c[d] for c in curves]
        spread[d] = round(float(max(vals) - min(vals)), 1)
    return spread

# ----------------- Step 4: K 折交叉验证与留一法 (LOOCV) -----------------
def cv_mse(estimator, X, y, k=5, random_state=0):
    """计算 K 折交叉验证 MSE 的均值与标准误。"""
    cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = -cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_squared_error')
    mean_val = float(np.mean(scores))
    se_val = float(np.std(scores, ddof=1) / np.sqrt(k))
    return round(mean_val, 2), round(se_val, 2)

def loocv_mse(estimator, X, y):
    """留一交叉验证 (LOOCV) MSE。"""
    loo = LeaveOneOut()
    scores = -cross_val_score(estimator, X, y, cv=loo, scoring='neg_mean_squared_error')
    return round(float(np.mean(scores)), 2)

# ----------------- Step 5: 不同 K 值折划分随机性对比 -----------------
def cv_spread_by_k(estimator, X, y, ks, seeds):
    """对比不同 K 值在不同随机种子下的均值与标准差。"""
    spread = {}
    for k in ks:
        estimates = [cv_mse(estimator, X, y, k=k, random_state=s)[0] for s in seeds]
        mean_val = round(float(np.mean(estimates)), 1)
        std_val = round(float(np.std(estimates, ddof=1)), 1)
        spread[k] = (mean_val, std_val)
    return spread

def compare_with_loocv(estimator, X, y, ks, seeds):
    """对比 K 折交叉验证与无随机性的 LOOCV。"""
    return {
        'loocv': loocv_mse(estimator, X, y),
        'kfold': cv_spread_by_k(estimator, X, y, ks, seeds)
    }

# ----------------- Step 6: 自助法 (Bootstrap) 评估回归系数标准误 -----------------
def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    """有放回抽样拟合 OLS，提取回归系数分布。"""
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    coefs = []
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = LinearRegression().fit(X_arr[idx], y_arr[idx])
        coefs.append(m.coef_)
    return np.array(coefs)

def bootstrap_se(coefs):
    """计算 Bootstrap 回归系数标准差。"""
    return np.round(np.std(coefs, axis=0, ddof=1), 2)

def ols_standard_errors(X, y):
    """利用解析公式计算 OLS 斜率系数的标准误。"""
    n, p = X.shape
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    A = np.column_stack([np.ones(n), X_arr])
    beta = np.linalg.lstsq(A, y_arr, rcond=None)[0]
    residuals = y_arr - A @ beta
    rss = np.sum(residuals ** 2)
    sigma2 = rss / (n - p - 1)
    inv_ata = np.linalg.inv(A.T @ A)
    se = np.sqrt(sigma2 * np.diag(inv_ata))[1:]
    return np.round(se, 2)

# ----------------- Step 7: 逐步特征选择路径 (Stepwise Selection) -----------------
def select_features(X, y, k, direction, cv):
    """序列特征选择器，返回指定特征数量的特征名。"""
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction=direction, cv=cv, scoring='neg_mean_squared_error')
    sfs.fit(X, y)
    return list(X.columns[sfs.get_support()])

def stepwise_path(X, y, direction, cv):
    """生成从 1 个特征到全特征的逐步选择路径字典。"""
    p = X.shape[1]
    path = {}
    for k in range(1, p):
        path[k] = select_features(X, y, k, direction, cv)
    path[p] = list(X.columns)
    return path

# ----------------- Step 8: 特征子集路径评估 -----------------
def score_path(X, y, path, cv):
    """评估路径上每个尺寸特征集的交叉验证 MSE 及其标准误。"""
    sizes = sorted(path.keys())
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for k in sizes:
        feats = path[k]
        scores = -cross_val_score(LinearRegression(), X[feats], y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return sizes, means, ses

def best_size(sizes, means):
    """返回 MSE 最小时对应的特征数量。"""
    idx = int(np.argmin(means))
    return sizes[idx]

# ----------------- Step 9: 1-SE 准则与子集决策 -----------------
def one_se_rule(values, means, ses, prefer='smaller'):
    """应用 1-SE 准则在可容忍阈值内挑选更简化的超参数。"""
    i_min = int(np.argmin(means))
    threshold = means[i_min] + ses[i_min]
    eligible = [v for v, m in zip(values, means) if m <= threshold]
    return min(eligible) if prefer == 'smaller' else max(eligible)

def choose_subset(X, y, direction, cv):
    """使用 1-SE 准则选择最优前向特征子集。"""
    path = stepwise_path(X, y, direction, cv)
    sizes, means, ses = score_path(X, y, path, cv)
    size_min = best_size(sizes, means)
    size_1se = one_se_rule(sizes, means, ses, prefer='smaller')
    return size_min, size_1se, path[size_1se]

# ----------------- Step 10: 岭回归 (Ridge) 正则化路径 -----------------
def ridge_model(alpha):
    """标准化 + 岭回归 Pipeline。"""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def ridge_path(X, y, alphas):
    """提取不同 alpha 惩罚下的岭回归系数矩阵。"""
    coefs = []
    for a in alphas:
        m = ridge_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def coef_norms(path):
    """计算每组系数向量的 L2 范数。"""
    return np.round(np.linalg.norm(path, axis=1), 2)

# ----------------- Step 11: 交叉验证超参数曲线 -----------------
def cv_curve(make_model, X, y, values, cv):
    """计算模型随网格参数变化的 CV 均值和标准误。"""
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for val in values:
        m = make_model(val)
        scores = -cross_val_score(m, X, y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return means, ses

def choose_penalty(make_model, X, y, values, cv):
    """挑选岭回归惩罚项（1-SE 偏好更大的正则化系数）。"""
    means, ses = cv_curve(make_model, X, y, values, cv)
    v_min = values[int(np.argmin(means))]
    v_1se = one_se_rule(values, means, ses, prefer='larger')
    return v_min, v_1se

# ----------------- Step 12: Lasso 稀疏化路径与 LassoCV -----------------
def lasso_model(alpha):
    """标准化 + Lasso Pipeline。"""
    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))

def lasso_path(X, y, alphas):
    """提取 Lasso 系数矩阵。"""
    coefs = []
    for a in alphas:
        m = lasso_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def nonzero_features(coef, names, tol=1e-8):
    """提取绝对值大于容差的非零特征名。"""
    return [name for name, c in zip(names, coef) if abs(c) > tol]

def lasso_cv(X, y, cv):
    """使用内置 LassoCV 寻优并返回最佳 alpha 与保留的稀疏特征。"""
    pipe = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0))
    pipe.fit(X, y)
    alpha = round(float(pipe[-1].alpha_), 4)
    selected = nonzero_features(pipe[-1].coef_, list(X.columns))
    return alpha, selected

# ----------------- Step 13: 主成分回归 (PCR) -----------------
def pcr_model(n_components):
    """标准化 + PCA 降维 + OLS 回归。"""
    return make_pipeline(StandardScaler(), PCA(n_components=n_components), LinearRegression())

def explained_variance(X):
    """计算累计解释方差比。"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA().fit(X_scaled)
    return np.round(np.cumsum(pca.explained_variance_ratio_), 3)

def pcr_curve(X, y, cv):
    """计算 1 到 p 个主成分下的 PCR 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pcr_model, X, y, comps, cv)
    return comps, means, ses

# ----------------- Step 14: 偏最小二乘回归 (PLS) -----------------
def pls_model(n_components):
    """监督式降维 PLS 回归。"""
    return PLSRegression(n_components=n_components, scale=True)

def pls_curve(X, y, cv):
    """计算 PLS 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pls_model, X, y, comps, cv)
    return comps, means, ses

def pls_predict(model, X):
    """将 PLS 预测的列矩阵压平为 1-D 数组。"""
    return np.asarray(model.predict(X)).ravel()

def best_components(components, means, ses):
    """选择主成分数量（1-SE 偏好更少成分）。"""
    i_min = int(np.argmin(means))
    m_min = components[i_min]
    m_1se = one_se_rule(components, means, ses, prefer='smaller')
    return m_min, m_1se

# ----------------- Step 15: 统一拟合全家桶模型 -----------------
def fit_all(X, y, cv, alphas):
    """统一拟合 6 种方法在各自交叉验证最优设置下的模型。"""
    models = {}
    all_cols = list(X.columns)
    
    # 1. ols
    models['ols'] = (LinearRegression().fit(X, y), all_cols, None)
    
    # 2. forward_1se
    _, size_1se, f_feats = choose_subset(X, y, 'forward', cv)
    m_fwd = LinearRegression().fit(X[f_feats], y)
    models['forward_1se'] = (m_fwd, f_feats, size_1se)
    
    # 3. ridge_1se
    _, r_alpha_1se = choose_penalty(ridge_model, X, y, alphas, cv)
    m_ridge = ridge_model(r_alpha_1se).fit(X, y)
    models['ridge_1se'] = (m_ridge, all_cols, r_alpha_1se)
    
    # 4. lasso_cv
    p_lasso = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0)).fit(X, y)
    l_alpha = round(float(p_lasso[-1].alpha_), 4)
    models['lasso_cv'] = (p_lasso, all_cols, l_alpha)
    
    # 5. pcr_1se
    comps, pm, ps = pcr_curve(X, y, cv)
    _, pcr_c_1se = best_components(comps, pm, ps)
    m_pcr = pcr_model(pcr_c_1se).fit(X, y)
    models['pcr_1se'] = (m_pcr, all_cols, pcr_c_1se)
    
    # 6. pls_1se
    comps, lm, ls = pls_curve(X, y, cv)
    _, pls_c_1se = best_components(comps, lm, ls)
    m_pls = pls_model(pls_c_1se).fit(X, y)
    models['pls_1se'] = (m_pls, all_cols, pls_c_1se)
    
    return models

# ----------------- Step 16: 最终测试集统一评估与制表 -----------------
def test_report(models, X_test, y_test):
    """打开测试集计算各个模型的最终 RMSE、有效特征数与参数设置。"""
    report = {}
    for name, (m, feats, setting) in models.items():
        if name == 'pls_1se':
            preds = pls_predict(m, X_test[feats])
        else:
            preds = np.asarray(m.predict(X_test[feats])).ravel()
            
        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 1)
        n_feats = len(nonzero_features(m[-1].coef_, feats)) if name == 'lasso_cv' else len(feats)
            
        report[name] = {
            'rmse': rmse,
            'n_features': n_feats,
            'setting': setting
        }
    return report

def best_method(report):
    """选出测试集上 RMSE 最低的方法。"""
    return min(report, key=lambda k: report[k]['rmse'])

def format_table(report):
    """按 RMSE 从低到高排版输出格式化结果行。"""
    lines = []
    for name in sorted(report, key=lambda k: report[k]['rmse']):
        r = report[name]
        lines.append(f"{name:12s} rmse={r['rmse']:6.1f} features={r['n_features']:2d} setting={r['setting']}")
    return lines

# Step 4 - cv_mse
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# ----------------- Step 1: 数据加载与概览 -----------------
def load_data():
    """从 sklearn 加载糖尿病数据集 (DataFrame 格式)。"""
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    return X, y

def describe_data(X, y):
    """返回样本量、特征数、特征列表与目标均值。"""
    return {
        'n': len(X),
        'p': X.shape[1],
        'features': list(X.columns),
        'y_mean': round(float(y.mean()), 2)
    }

# ----------------- Step 2: 划分训练集与测试集 -----------------
def train_test(X, y, test_size=0.25, random_state=0):
    """一次性保留测试集，后续模型选择与调优只在训练集上进行。"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ----------------- Step 3: 单次验证集切分曲线与波动评估 -----------------
def poly_model(degree):
    """多项式回归 Pipeline。"""
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

def validation_set_curve(X, y, feature, degrees, random_state):
    """使用 50/50 单次切分评估不同多项式阶数的验证集 MSE。"""
    X_sub = X[[feature]]
    X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y, test_size=0.5, random_state=random_state)
    curve = {}
    for d in degrees:
        m = poly_model(d).fit(X_tr, y_tr)
        pred = m.predict(X_val)
        curve[d] = round(float(mean_squared_error(y_val, pred)), 1)
    return curve

def curve_spread(X, y, feature, degrees, seeds):
    """统计不同随机切分下验证集 MSE 的极差（最大值减最小值）。"""
    curves = [validation_set_curve(X, y, feature, degrees, s) for s in seeds]
    spread = {}
    for d in degrees:
        vals = [c[d] for c in curves]
        spread[d] = round(float(max(vals) - min(vals)), 1)
    return spread

# ----------------- Step 4: K 折交叉验证与留一法 (LOOCV) -----------------
def cv_mse(estimator, X, y, k=5, random_state=0):
    """计算 K 折交叉验证 MSE 的均值与标准误。"""
    cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = -cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_squared_error')
    mean_val = float(np.mean(scores))
    se_val = float(np.std(scores, ddof=1) / np.sqrt(k))
    return round(mean_val, 2), round(se_val, 2)

def loocv_mse(estimator, X, y):
    """留一交叉验证 (LOOCV) MSE。"""
    loo = LeaveOneOut()
    scores = -cross_val_score(estimator, X, y, cv=loo, scoring='neg_mean_squared_error')
    return round(float(np.mean(scores)), 2)

# ----------------- Step 5: 不同 K 值折划分随机性对比 -----------------
def cv_spread_by_k(estimator, X, y, ks, seeds):
    """对比不同 K 值在不同随机种子下的均值与标准差。"""
    spread = {}
    for k in ks:
        estimates = [cv_mse(estimator, X, y, k=k, random_state=s)[0] for s in seeds]
        mean_val = round(float(np.mean(estimates)), 1)
        std_val = round(float(np.std(estimates, ddof=1)), 1)
        spread[k] = (mean_val, std_val)
    return spread

def compare_with_loocv(estimator, X, y, ks, seeds):
    """对比 K 折交叉验证与无随机性的 LOOCV。"""
    return {
        'loocv': loocv_mse(estimator, X, y),
        'kfold': cv_spread_by_k(estimator, X, y, ks, seeds)
    }

# ----------------- Step 6: 自助法 (Bootstrap) 评估回归系数标准误 -----------------
def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    """有放回抽样拟合 OLS，提取回归系数分布。"""
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    coefs = []
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = LinearRegression().fit(X_arr[idx], y_arr[idx])
        coefs.append(m.coef_)
    return np.array(coefs)

def bootstrap_se(coefs):
    """计算 Bootstrap 回归系数标准差。"""
    return np.round(np.std(coefs, axis=0, ddof=1), 2)

def ols_standard_errors(X, y):
    """利用解析公式计算 OLS 斜率系数的标准误。"""
    n, p = X.shape
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    A = np.column_stack([np.ones(n), X_arr])
    beta = np.linalg.lstsq(A, y_arr, rcond=None)[0]
    residuals = y_arr - A @ beta
    rss = np.sum(residuals ** 2)
    sigma2 = rss / (n - p - 1)
    inv_ata = np.linalg.inv(A.T @ A)
    se = np.sqrt(sigma2 * np.diag(inv_ata))[1:]
    return np.round(se, 2)

# ----------------- Step 7: 逐步特征选择路径 (Stepwise Selection) -----------------
def select_features(X, y, k, direction, cv):
    """序列特征选择器，返回指定特征数量的特征名。"""
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction=direction, cv=cv, scoring='neg_mean_squared_error')
    sfs.fit(X, y)
    return list(X.columns[sfs.get_support()])

def stepwise_path(X, y, direction, cv):
    """生成从 1 个特征到全特征的逐步选择路径字典。"""
    p = X.shape[1]
    path = {}
    for k in range(1, p):
        path[k] = select_features(X, y, k, direction, cv)
    path[p] = list(X.columns)
    return path

# ----------------- Step 8: 特征子集路径评估 -----------------
def score_path(X, y, path, cv):
    """评估路径上每个尺寸特征集的交叉验证 MSE 及其标准误。"""
    sizes = sorted(path.keys())
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for k in sizes:
        feats = path[k]
        scores = -cross_val_score(LinearRegression(), X[feats], y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return sizes, means, ses

def best_size(sizes, means):
    """返回 MSE 最小时对应的特征数量。"""
    idx = int(np.argmin(means))
    return sizes[idx]

# ----------------- Step 9: 1-SE 准则与子集决策 -----------------
def one_se_rule(values, means, ses, prefer='smaller'):
    """应用 1-SE 准则在可容忍阈值内挑选更简化的超参数。"""
    i_min = int(np.argmin(means))
    threshold = means[i_min] + ses[i_min]
    eligible = [v for v, m in zip(values, means) if m <= threshold]
    return min(eligible) if prefer == 'smaller' else max(eligible)

def choose_subset(X, y, direction, cv):
    """使用 1-SE 准则选择最优前向特征子集。"""
    path = stepwise_path(X, y, direction, cv)
    sizes, means, ses = score_path(X, y, path, cv)
    size_min = best_size(sizes, means)
    size_1se = one_se_rule(sizes, means, ses, prefer='smaller')
    return size_min, size_1se, path[size_1se]

# ----------------- Step 10: 岭回归 (Ridge) 正则化路径 -----------------
def ridge_model(alpha):
    """标准化 + 岭回归 Pipeline。"""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def ridge_path(X, y, alphas):
    """提取不同 alpha 惩罚下的岭回归系数矩阵。"""
    coefs = []
    for a in alphas:
        m = ridge_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def coef_norms(path):
    """计算每组系数向量的 L2 范数。"""
    return np.round(np.linalg.norm(path, axis=1), 2)

# ----------------- Step 11: 交叉验证超参数曲线 -----------------
def cv_curve(make_model, X, y, values, cv):
    """计算模型随网格参数变化的 CV 均值和标准误。"""
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for val in values:
        m = make_model(val)
        scores = -cross_val_score(m, X, y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return means, ses

def choose_penalty(make_model, X, y, values, cv):
    """挑选岭回归惩罚项（1-SE 偏好更大的正则化系数）。"""
    means, ses = cv_curve(make_model, X, y, values, cv)
    v_min = values[int(np.argmin(means))]
    v_1se = one_se_rule(values, means, ses, prefer='larger')
    return v_min, v_1se

# ----------------- Step 12: Lasso 稀疏化路径与 LassoCV -----------------
def lasso_model(alpha):
    """标准化 + Lasso Pipeline。"""
    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))

def lasso_path(X, y, alphas):
    """提取 Lasso 系数矩阵。"""
    coefs = []
    for a in alphas:
        m = lasso_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def nonzero_features(coef, names, tol=1e-8):
    """提取绝对值大于容差的非零特征名。"""
    return [name for name, c in zip(names, coef) if abs(c) > tol]

def lasso_cv(X, y, cv):
    """使用内置 LassoCV 寻优并返回最佳 alpha 与保留的稀疏特征。"""
    pipe = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0))
    pipe.fit(X, y)
    alpha = round(float(pipe[-1].alpha_), 4)
    selected = nonzero_features(pipe[-1].coef_, list(X.columns))
    return alpha, selected

# ----------------- Step 13: 主成分回归 (PCR) -----------------
def pcr_model(n_components):
    """标准化 + PCA 降维 + OLS 回归。"""
    return make_pipeline(StandardScaler(), PCA(n_components=n_components), LinearRegression())

def explained_variance(X):
    """计算累计解释方差比。"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA().fit(X_scaled)
    return np.round(np.cumsum(pca.explained_variance_ratio_), 3)

def pcr_curve(X, y, cv):
    """计算 1 到 p 个主成分下的 PCR 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pcr_model, X, y, comps, cv)
    return comps, means, ses

# ----------------- Step 14: 偏最小二乘回归 (PLS) -----------------
def pls_model(n_components):
    """监督式降维 PLS 回归。"""
    return PLSRegression(n_components=n_components, scale=True)

def pls_curve(X, y, cv):
    """计算 PLS 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pls_model, X, y, comps, cv)
    return comps, means, ses

def pls_predict(model, X):
    """将 PLS 预测的列矩阵压平为 1-D 数组。"""
    return np.asarray(model.predict(X)).ravel()

def best_components(components, means, ses):
    """选择主成分数量（1-SE 偏好更少成分）。"""
    i_min = int(np.argmin(means))
    m_min = components[i_min]
    m_1se = one_se_rule(components, means, ses, prefer='smaller')
    return m_min, m_1se

# ----------------- Step 15: 统一拟合全家桶模型 -----------------
def fit_all(X, y, cv, alphas):
    """统一拟合 6 种方法在各自交叉验证最优设置下的模型。"""
    models = {}
    all_cols = list(X.columns)
    
    # 1. ols
    models['ols'] = (LinearRegression().fit(X, y), all_cols, None)
    
    # 2. forward_1se
    _, size_1se, f_feats = choose_subset(X, y, 'forward', cv)
    m_fwd = LinearRegression().fit(X[f_feats], y)
    models['forward_1se'] = (m_fwd, f_feats, size_1se)
    
    # 3. ridge_1se
    _, r_alpha_1se = choose_penalty(ridge_model, X, y, alphas, cv)
    m_ridge = ridge_model(r_alpha_1se).fit(X, y)
    models['ridge_1se'] = (m_ridge, all_cols, r_alpha_1se)
    
    # 4. lasso_cv
    p_lasso = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0)).fit(X, y)
    l_alpha = round(float(p_lasso[-1].alpha_), 4)
    models['lasso_cv'] = (p_lasso, all_cols, l_alpha)
    
    # 5. pcr_1se
    comps, pm, ps = pcr_curve(X, y, cv)
    _, pcr_c_1se = best_components(comps, pm, ps)
    m_pcr = pcr_model(pcr_c_1se).fit(X, y)
    models['pcr_1se'] = (m_pcr, all_cols, pcr_c_1se)
    
    # 6. pls_1se
    comps, lm, ls = pls_curve(X, y, cv)
    _, pls_c_1se = best_components(comps, lm, ls)
    m_pls = pls_model(pls_c_1se).fit(X, y)
    models['pls_1se'] = (m_pls, all_cols, pls_c_1se)
    
    return models

# ----------------- Step 16: 最终测试集统一评估与制表 -----------------
def test_report(models, X_test, y_test):
    """打开测试集计算各个模型的最终 RMSE、有效特征数与参数设置。"""
    report = {}
    for name, (m, feats, setting) in models.items():
        if name == 'pls_1se':
            preds = pls_predict(m, X_test[feats])
        else:
            preds = np.asarray(m.predict(X_test[feats])).ravel()
            
        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 1)
        n_feats = len(nonzero_features(m[-1].coef_, feats)) if name == 'lasso_cv' else len(feats)
            
        report[name] = {
            'rmse': rmse,
            'n_features': n_feats,
            'setting': setting
        }
    return report

def best_method(report):
    """选出测试集上 RMSE 最低的方法。"""
    return min(report, key=lambda k: report[k]['rmse'])

def format_table(report):
    """按 RMSE 从低到高排版输出格式化结果行。"""
    lines = []
    for name in sorted(report, key=lambda k: report[k]['rmse']):
        r = report[name]
        lines.append(f"{name:12s} rmse={r['rmse']:6.1f} features={r['n_features']:2d} setting={r['setting']}")
    return lines

# Step 5 - cv_spread_by_k
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# ----------------- Step 1: 数据加载与概览 -----------------
def load_data():
    """从 sklearn 加载糖尿病数据集 (DataFrame 格式)。"""
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    return X, y

def describe_data(X, y):
    """返回样本量、特征数、特征列表与目标均值。"""
    return {
        'n': len(X),
        'p': X.shape[1],
        'features': list(X.columns),
        'y_mean': round(float(y.mean()), 2)
    }

# ----------------- Step 2: 划分训练集与测试集 -----------------
def train_test(X, y, test_size=0.25, random_state=0):
    """一次性保留测试集，后续模型选择与调优只在训练集上进行。"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ----------------- Step 3: 单次验证集切分曲线与波动评估 -----------------
def poly_model(degree):
    """多项式回归 Pipeline。"""
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

def validation_set_curve(X, y, feature, degrees, random_state):
    """使用 50/50 单次切分评估不同多项式阶数的验证集 MSE。"""
    X_sub = X[[feature]]
    X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y, test_size=0.5, random_state=random_state)
    curve = {}
    for d in degrees:
        m = poly_model(d).fit(X_tr, y_tr)
        pred = m.predict(X_val)
        curve[d] = round(float(mean_squared_error(y_val, pred)), 1)
    return curve

def curve_spread(X, y, feature, degrees, seeds):
    """统计不同随机切分下验证集 MSE 的极差（最大值减最小值）。"""
    curves = [validation_set_curve(X, y, feature, degrees, s) for s in seeds]
    spread = {}
    for d in degrees:
        vals = [c[d] for c in curves]
        spread[d] = round(float(max(vals) - min(vals)), 1)
    return spread

# ----------------- Step 4: K 折交叉验证与留一法 (LOOCV) -----------------
def cv_mse(estimator, X, y, k=5, random_state=0):
    """计算 K 折交叉验证 MSE 的均值与标准误。"""
    cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = -cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_squared_error')
    mean_val = float(np.mean(scores))
    se_val = float(np.std(scores, ddof=1) / np.sqrt(k))
    return round(mean_val, 2), round(se_val, 2)

def loocv_mse(estimator, X, y):
    """留一交叉验证 (LOOCV) MSE。"""
    loo = LeaveOneOut()
    scores = -cross_val_score(estimator, X, y, cv=loo, scoring='neg_mean_squared_error')
    return round(float(np.mean(scores)), 2)

# ----------------- Step 5: 不同 K 值折划分随机性对比 -----------------
def cv_spread_by_k(estimator, X, y, ks, seeds):
    """对比不同 K 值在不同随机种子下的均值与标准差。"""
    spread = {}
    for k in ks:
        estimates = [cv_mse(estimator, X, y, k=k, random_state=s)[0] for s in seeds]
        mean_val = round(float(np.mean(estimates)), 1)
        std_val = round(float(np.std(estimates, ddof=1)), 1)
        spread[k] = (mean_val, std_val)
    return spread

def compare_with_loocv(estimator, X, y, ks, seeds):
    """对比 K 折交叉验证与无随机性的 LOOCV。"""
    return {
        'loocv': loocv_mse(estimator, X, y),
        'kfold': cv_spread_by_k(estimator, X, y, ks, seeds)
    }

# ----------------- Step 6: 自助法 (Bootstrap) 评估回归系数标准误 -----------------
def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    """有放回抽样拟合 OLS，提取回归系数分布。"""
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    coefs = []
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = LinearRegression().fit(X_arr[idx], y_arr[idx])
        coefs.append(m.coef_)
    return np.array(coefs)

def bootstrap_se(coefs):
    """计算 Bootstrap 回归系数标准差。"""
    return np.round(np.std(coefs, axis=0, ddof=1), 2)

def ols_standard_errors(X, y):
    """利用解析公式计算 OLS 斜率系数的标准误。"""
    n, p = X.shape
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    A = np.column_stack([np.ones(n), X_arr])
    beta = np.linalg.lstsq(A, y_arr, rcond=None)[0]
    residuals = y_arr - A @ beta
    rss = np.sum(residuals ** 2)
    sigma2 = rss / (n - p - 1)
    inv_ata = np.linalg.inv(A.T @ A)
    se = np.sqrt(sigma2 * np.diag(inv_ata))[1:]
    return np.round(se, 2)

# ----------------- Step 7: 逐步特征选择路径 (Stepwise Selection) -----------------
def select_features(X, y, k, direction, cv):
    """序列特征选择器，返回指定特征数量的特征名。"""
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction=direction, cv=cv, scoring='neg_mean_squared_error')
    sfs.fit(X, y)
    return list(X.columns[sfs.get_support()])

def stepwise_path(X, y, direction, cv):
    """生成从 1 个特征到全特征的逐步选择路径字典。"""
    p = X.shape[1]
    path = {}
    for k in range(1, p):
        path[k] = select_features(X, y, k, direction, cv)
    path[p] = list(X.columns)
    return path

# ----------------- Step 8: 特征子集路径评估 -----------------
def score_path(X, y, path, cv):
    """评估路径上每个尺寸特征集的交叉验证 MSE 及其标准误。"""
    sizes = sorted(path.keys())
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for k in sizes:
        feats = path[k]
        scores = -cross_val_score(LinearRegression(), X[feats], y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return sizes, means, ses

def best_size(sizes, means):
    """返回 MSE 最小时对应的特征数量。"""
    idx = int(np.argmin(means))
    return sizes[idx]

# ----------------- Step 9: 1-SE 准则与子集决策 -----------------
def one_se_rule(values, means, ses, prefer='smaller'):
    """应用 1-SE 准则在可容忍阈值内挑选更简化的超参数。"""
    i_min = int(np.argmin(means))
    threshold = means[i_min] + ses[i_min]
    eligible = [v for v, m in zip(values, means) if m <= threshold]
    return min(eligible) if prefer == 'smaller' else max(eligible)

def choose_subset(X, y, direction, cv):
    """使用 1-SE 准则选择最优前向特征子集。"""
    path = stepwise_path(X, y, direction, cv)
    sizes, means, ses = score_path(X, y, path, cv)
    size_min = best_size(sizes, means)
    size_1se = one_se_rule(sizes, means, ses, prefer='smaller')
    return size_min, size_1se, path[size_1se]

# ----------------- Step 10: 岭回归 (Ridge) 正则化路径 -----------------
def ridge_model(alpha):
    """标准化 + 岭回归 Pipeline。"""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def ridge_path(X, y, alphas):
    """提取不同 alpha 惩罚下的岭回归系数矩阵。"""
    coefs = []
    for a in alphas:
        m = ridge_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def coef_norms(path):
    """计算每组系数向量的 L2 范数。"""
    return np.round(np.linalg.norm(path, axis=1), 2)

# ----------------- Step 11: 交叉验证超参数曲线 -----------------
def cv_curve(make_model, X, y, values, cv):
    """计算模型随网格参数变化的 CV 均值和标准误。"""
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for val in values:
        m = make_model(val)
        scores = -cross_val_score(m, X, y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return means, ses

def choose_penalty(make_model, X, y, values, cv):
    """挑选岭回归惩罚项（1-SE 偏好更大的正则化系数）。"""
    means, ses = cv_curve(make_model, X, y, values, cv)
    v_min = values[int(np.argmin(means))]
    v_1se = one_se_rule(values, means, ses, prefer='larger')
    return v_min, v_1se

# ----------------- Step 12: Lasso 稀疏化路径与 LassoCV -----------------
def lasso_model(alpha):
    """标准化 + Lasso Pipeline。"""
    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))

def lasso_path(X, y, alphas):
    """提取 Lasso 系数矩阵。"""
    coefs = []
    for a in alphas:
        m = lasso_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def nonzero_features(coef, names, tol=1e-8):
    """提取绝对值大于容差的非零特征名。"""
    return [name for name, c in zip(names, coef) if abs(c) > tol]

def lasso_cv(X, y, cv):
    """使用内置 LassoCV 寻优并返回最佳 alpha 与保留的稀疏特征。"""
    pipe = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0))
    pipe.fit(X, y)
    alpha = round(float(pipe[-1].alpha_), 4)
    selected = nonzero_features(pipe[-1].coef_, list(X.columns))
    return alpha, selected

# ----------------- Step 13: 主成分回归 (PCR) -----------------
def pcr_model(n_components):
    """标准化 + PCA 降维 + OLS 回归。"""
    return make_pipeline(StandardScaler(), PCA(n_components=n_components), LinearRegression())

def explained_variance(X):
    """计算累计解释方差比。"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA().fit(X_scaled)
    return np.round(np.cumsum(pca.explained_variance_ratio_), 3)

def pcr_curve(X, y, cv):
    """计算 1 到 p 个主成分下的 PCR 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pcr_model, X, y, comps, cv)
    return comps, means, ses

# ----------------- Step 14: 偏最小二乘回归 (PLS) -----------------
def pls_model(n_components):
    """监督式降维 PLS 回归。"""
    return PLSRegression(n_components=n_components, scale=True)

def pls_curve(X, y, cv):
    """计算 PLS 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pls_model, X, y, comps, cv)
    return comps, means, ses

def pls_predict(model, X):
    """将 PLS 预测的列矩阵压平为 1-D 数组。"""
    return np.asarray(model.predict(X)).ravel()

def best_components(components, means, ses):
    """选择主成分数量（1-SE 偏好更少成分）。"""
    i_min = int(np.argmin(means))
    m_min = components[i_min]
    m_1se = one_se_rule(components, means, ses, prefer='smaller')
    return m_min, m_1se

# ----------------- Step 15: 统一拟合全家桶模型 -----------------
def fit_all(X, y, cv, alphas):
    """统一拟合 6 种方法在各自交叉验证最优设置下的模型。"""
    models = {}
    all_cols = list(X.columns)
    
    # 1. ols
    models['ols'] = (LinearRegression().fit(X, y), all_cols, None)
    
    # 2. forward_1se
    _, size_1se, f_feats = choose_subset(X, y, 'forward', cv)
    m_fwd = LinearRegression().fit(X[f_feats], y)
    models['forward_1se'] = (m_fwd, f_feats, size_1se)
    
    # 3. ridge_1se
    _, r_alpha_1se = choose_penalty(ridge_model, X, y, alphas, cv)
    m_ridge = ridge_model(r_alpha_1se).fit(X, y)
    models['ridge_1se'] = (m_ridge, all_cols, r_alpha_1se)
    
    # 4. lasso_cv
    p_lasso = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0)).fit(X, y)
    l_alpha = round(float(p_lasso[-1].alpha_), 4)
    models['lasso_cv'] = (p_lasso, all_cols, l_alpha)
    
    # 5. pcr_1se
    comps, pm, ps = pcr_curve(X, y, cv)
    _, pcr_c_1se = best_components(comps, pm, ps)
    m_pcr = pcr_model(pcr_c_1se).fit(X, y)
    models['pcr_1se'] = (m_pcr, all_cols, pcr_c_1se)
    
    # 6. pls_1se
    comps, lm, ls = pls_curve(X, y, cv)
    _, pls_c_1se = best_components(comps, lm, ls)
    m_pls = pls_model(pls_c_1se).fit(X, y)
    models['pls_1se'] = (m_pls, all_cols, pls_c_1se)
    
    return models

# ----------------- Step 16: 最终测试集统一评估与制表 -----------------
def test_report(models, X_test, y_test):
    """打开测试集计算各个模型的最终 RMSE、有效特征数与参数设置。"""
    report = {}
    for name, (m, feats, setting) in models.items():
        if name == 'pls_1se':
            preds = pls_predict(m, X_test[feats])
        else:
            preds = np.asarray(m.predict(X_test[feats])).ravel()
            
        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 1)
        n_feats = len(nonzero_features(m[-1].coef_, feats)) if name == 'lasso_cv' else len(feats)
            
        report[name] = {
            'rmse': rmse,
            'n_features': n_feats,
            'setting': setting
        }
    return report

def best_method(report):
    """选出测试集上 RMSE 最低的方法。"""
    return min(report, key=lambda k: report[k]['rmse'])

def format_table(report):
    """按 RMSE 从低到高排版输出格式化结果行。"""
    lines = []
    for name in sorted(report, key=lambda k: report[k]['rmse']):
        r = report[name]
        lines.append(f"{name:12s} rmse={r['rmse']:6.1f} features={r['n_features']:2d} setting={r['setting']}")
    return lines

# Step 6 - bootstrap_coefficients
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# ----------------- Step 1: 数据加载与概览 -----------------
def load_data():
    """从 sklearn 加载糖尿病数据集 (DataFrame 格式)。"""
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    return X, y

def describe_data(X, y):
    """返回样本量、特征数、特征列表与目标均值。"""
    return {
        'n': len(X),
        'p': X.shape[1],
        'features': list(X.columns),
        'y_mean': round(float(y.mean()), 2)
    }

# ----------------- Step 2: 划分训练集与测试集 -----------------
def train_test(X, y, test_size=0.25, random_state=0):
    """一次性保留测试集，后续模型选择与调优只在训练集上进行。"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ----------------- Step 3: 单次验证集切分曲线与波动评估 -----------------
def poly_model(degree):
    """多项式回归 Pipeline。"""
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

def validation_set_curve(X, y, feature, degrees, random_state):
    """使用 50/50 单次切分评估不同多项式阶数的验证集 MSE。"""
    X_sub = X[[feature]]
    X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y, test_size=0.5, random_state=random_state)
    curve = {}
    for d in degrees:
        m = poly_model(d).fit(X_tr, y_tr)
        pred = m.predict(X_val)
        curve[d] = round(float(mean_squared_error(y_val, pred)), 1)
    return curve

def curve_spread(X, y, feature, degrees, seeds):
    """统计不同随机切分下验证集 MSE 的极差（最大值减最小值）。"""
    curves = [validation_set_curve(X, y, feature, degrees, s) for s in seeds]
    spread = {}
    for d in degrees:
        vals = [c[d] for c in curves]
        spread[d] = round(float(max(vals) - min(vals)), 1)
    return spread

# ----------------- Step 4: K 折交叉验证与留一法 (LOOCV) -----------------
def cv_mse(estimator, X, y, k=5, random_state=0):
    """计算 K 折交叉验证 MSE 的均值与标准误。"""
    cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = -cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_squared_error')
    mean_val = float(np.mean(scores))
    se_val = float(np.std(scores, ddof=1) / np.sqrt(k))
    return round(mean_val, 2), round(se_val, 2)

def loocv_mse(estimator, X, y):
    """留一交叉验证 (LOOCV) MSE。"""
    loo = LeaveOneOut()
    scores = -cross_val_score(estimator, X, y, cv=loo, scoring='neg_mean_squared_error')
    return round(float(np.mean(scores)), 2)

# ----------------- Step 5: 不同 K 值折划分随机性对比 -----------------
def cv_spread_by_k(estimator, X, y, ks, seeds):
    """对比不同 K 值在不同随机种子下的均值与标准差。"""
    spread = {}
    for k in ks:
        estimates = [cv_mse(estimator, X, y, k=k, random_state=s)[0] for s in seeds]
        mean_val = round(float(np.mean(estimates)), 1)
        std_val = round(float(np.std(estimates, ddof=1)), 1)
        spread[k] = (mean_val, std_val)
    return spread

def compare_with_loocv(estimator, X, y, ks, seeds):
    """对比 K 折交叉验证与无随机性的 LOOCV。"""
    return {
        'loocv': loocv_mse(estimator, X, y),
        'kfold': cv_spread_by_k(estimator, X, y, ks, seeds)
    }

# ----------------- Step 6: 自助法 (Bootstrap) 评估回归系数标准误 -----------------
def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    """有放回抽样拟合 OLS，提取回归系数分布。"""
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    coefs = []
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = LinearRegression().fit(X_arr[idx], y_arr[idx])
        coefs.append(m.coef_)
    return np.array(coefs)

def bootstrap_se(coefs):
    """计算 Bootstrap 回归系数标准差。"""
    return np.round(np.std(coefs, axis=0, ddof=1), 2)

def ols_standard_errors(X, y):
    """利用解析公式计算 OLS 斜率系数的标准误。"""
    n, p = X.shape
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    A = np.column_stack([np.ones(n), X_arr])
    beta = np.linalg.lstsq(A, y_arr, rcond=None)[0]
    residuals = y_arr - A @ beta
    rss = np.sum(residuals ** 2)
    sigma2 = rss / (n - p - 1)
    inv_ata = np.linalg.inv(A.T @ A)
    se = np.sqrt(sigma2 * np.diag(inv_ata))[1:]
    return np.round(se, 2)

# ----------------- Step 7: 逐步特征选择路径 (Stepwise Selection) -----------------
def select_features(X, y, k, direction, cv):
    """序列特征选择器，返回指定特征数量的特征名。"""
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction=direction, cv=cv, scoring='neg_mean_squared_error')
    sfs.fit(X, y)
    return list(X.columns[sfs.get_support()])

def stepwise_path(X, y, direction, cv):
    """生成从 1 个特征到全特征的逐步选择路径字典。"""
    p = X.shape[1]
    path = {}
    for k in range(1, p):
        path[k] = select_features(X, y, k, direction, cv)
    path[p] = list(X.columns)
    return path

# ----------------- Step 8: 特征子集路径评估 -----------------
def score_path(X, y, path, cv):
    """评估路径上每个尺寸特征集的交叉验证 MSE 及其标准误。"""
    sizes = sorted(path.keys())
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for k in sizes:
        feats = path[k]
        scores = -cross_val_score(LinearRegression(), X[feats], y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return sizes, means, ses

def best_size(sizes, means):
    """返回 MSE 最小时对应的特征数量。"""
    idx = int(np.argmin(means))
    return sizes[idx]

# ----------------- Step 9: 1-SE 准则与子集决策 -----------------
def one_se_rule(values, means, ses, prefer='smaller'):
    """应用 1-SE 准则在可容忍阈值内挑选更简化的超参数。"""
    i_min = int(np.argmin(means))
    threshold = means[i_min] + ses[i_min]
    eligible = [v for v, m in zip(values, means) if m <= threshold]
    return min(eligible) if prefer == 'smaller' else max(eligible)

def choose_subset(X, y, direction, cv):
    """使用 1-SE 准则选择最优前向特征子集。"""
    path = stepwise_path(X, y, direction, cv)
    sizes, means, ses = score_path(X, y, path, cv)
    size_min = best_size(sizes, means)
    size_1se = one_se_rule(sizes, means, ses, prefer='smaller')
    return size_min, size_1se, path[size_1se]

# ----------------- Step 10: 岭回归 (Ridge) 正则化路径 -----------------
def ridge_model(alpha):
    """标准化 + 岭回归 Pipeline。"""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def ridge_path(X, y, alphas):
    """提取不同 alpha 惩罚下的岭回归系数矩阵。"""
    coefs = []
    for a in alphas:
        m = ridge_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def coef_norms(path):
    """计算每组系数向量的 L2 范数。"""
    return np.round(np.linalg.norm(path, axis=1), 2)

# ----------------- Step 11: 交叉验证超参数曲线 -----------------
def cv_curve(make_model, X, y, values, cv):
    """计算模型随网格参数变化的 CV 均值和标准误。"""
    means, ses = [], []
    n_folds = cv.get_n_splits(X)
    for val in values:
        m = make_model(val)
        scores = -cross_val_score(m, X, y, cv=cv, scoring='neg_mean_squared_error')
        means.append(round(float(np.mean(scores)), 1))
        ses.append(round(float(np.std(scores, ddof=1) / np.sqrt(n_folds)), 1))
    return means, ses

def choose_penalty(make_model, X, y, values, cv):
    """挑选岭回归惩罚项（1-SE 偏好更大的正则化系数）。"""
    means, ses = cv_curve(make_model, X, y, values, cv)
    v_min = values[int(np.argmin(means))]
    v_1se = one_se_rule(values, means, ses, prefer='larger')
    return v_min, v_1se

# ----------------- Step 12: Lasso 稀疏化路径与 LassoCV -----------------
def lasso_model(alpha):
    """标准化 + Lasso Pipeline。"""
    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))

def lasso_path(X, y, alphas):
    """提取 Lasso 系数矩阵。"""
    coefs = []
    for a in alphas:
        m = lasso_model(a).fit(X, y)
        coefs.append(m[-1].coef_)
    return np.array(coefs)

def nonzero_features(coef, names, tol=1e-8):
    """提取绝对值大于容差的非零特征名。"""
    return [name for name, c in zip(names, coef) if abs(c) > tol]

def lasso_cv(X, y, cv):
    """使用内置 LassoCV 寻优并返回最佳 alpha 与保留的稀疏特征。"""
    pipe = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0))
    pipe.fit(X, y)
    alpha = round(float(pipe[-1].alpha_), 4)
    selected = nonzero_features(pipe[-1].coef_, list(X.columns))
    return alpha, selected

# ----------------- Step 13: 主成分回归 (PCR) -----------------
def pcr_model(n_components):
    """标准化 + PCA 降维 + OLS 回归。"""
    return make_pipeline(StandardScaler(), PCA(n_components=n_components), LinearRegression())

def explained_variance(X):
    """计算累计解释方差比。"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA().fit(X_scaled)
    return np.round(np.cumsum(pca.explained_variance_ratio_), 3)

def pcr_curve(X, y, cv):
    """计算 1 到 p 个主成分下的 PCR 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pcr_model, X, y, comps, cv)
    return comps, means, ses

# ----------------- Step 14: 偏最小二乘回归 (PLS) -----------------
def pls_model(n_components):
    """监督式降维 PLS 回归。"""
    return PLSRegression(n_components=n_components, scale=True)

def pls_curve(X, y, cv):
    """计算 PLS 交叉验证曲线。"""
    comps = list(range(1, X.shape[1] + 1))
    means, ses = cv_curve(pls_model, X, y, comps, cv)
    return comps, means, ses

def pls_predict(model, X):
    """将 PLS 预测的列矩阵压平为 1-D 数组。"""
    return np.asarray(model.predict(X)).ravel()

def best_components(components, means, ses):
    """选择主成分数量（1-SE 偏好更少成分）。"""
    i_min = int(np.argmin(means))
    m_min = components[i_min]
    m_1se = one_se_rule(components, means, ses, prefer='smaller')
    return m_min, m_1se

# ----------------- Step 15: 统一拟合全家桶模型 -----------------
def fit_all(X, y, cv, alphas):
    """统一拟合 6 种方法在各自交叉验证最优设置下的模型。"""
    models = {}
    all_cols = list(X.columns)
    
    # 1. ols
    models['ols'] = (LinearRegression().fit(X, y), all_cols, None)
    
    # 2. forward_1se
    _, size_1se, f_feats = choose_subset(X, y, 'forward', cv)
    m_fwd = LinearRegression().fit(X[f_feats], y)
    models['forward_1se'] = (m_fwd, f_feats, size_1se)
    
    # 3. ridge_1se
    _, r_alpha_1se = choose_penalty(ridge_model, X, y, alphas, cv)
    m_ridge = ridge_model(r_alpha_1se).fit(X, y)
    models['ridge_1se'] = (m_ridge, all_cols, r_alpha_1se)
    
    # 4. lasso_cv
    p_lasso = make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=20000, random_state=0)).fit(X, y)
    l_alpha = round(float(p_lasso[-1].alpha_), 4)
    models['lasso_cv'] = (p_lasso, all_cols, l_alpha)
    
    # 5. pcr_1se
    comps, pm, ps = pcr_curve(X, y, cv)
    _, pcr_c_1se = best_components(comps, pm, ps)
    m_pcr = pcr_model(pcr_c_1se).fit(X, y)
    models['pcr_1se'] = (m_pcr, all_cols, pcr_c_1se)
    
    # 6. pls_1se
    comps, lm, ls = pls_curve(X, y, cv)
    _, pls_c_1se = best_components(comps, lm, ls)
    m_pls = pls_model(pls_c_1se).fit(X, y)
    models['pls_1se'] = (m_pls, all_cols, pls_c_1se)
    
    return models

# ----------------- Step 16: 最终测试集统一评估与制表 -----------------
def test_report(models, X_test, y_test):
    """打开测试集计算各个模型的最终 RMSE、有效特征数与参数设置。"""
    report = {}
    for name, (m, feats, setting) in models.items():
        if name == 'pls_1se':
            preds = pls_predict(m, X_test[feats])
        else:
            preds = np.asarray(m.predict(X_test[feats])).ravel()
            
        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 1)
        n_feats = len(nonzero_features(m[-1].coef_, feats)) if name == 'lasso_cv' else len(feats)
            
        report[name] = {
            'rmse': rmse,
            'n_features': n_feats,
            'setting': setting
        }
    return report

def best_method(report):
    """选出测试集上 RMSE 最低的方法。"""
    return min(report, key=lambda k: report[k]['rmse'])

def format_table(report):
    """按 RMSE 从低到高排版输出格式化结果行。"""
    lines = []
    for name in sorted(report, key=lambda k: report[k]['rmse']):
        r = report[name]
        lines.append(f"{name:12s} rmse={r['rmse']:6.1f} features={r['n_features']:2d} setting={r['setting']}")
    return lines

# Step 7 - stepwise_path (not yet solved)
# TODO: implement

# Step 8 - score_path (not yet solved)
# TODO: implement

# Step 9 - one_se_rule (not yet solved)
# TODO: implement

# Step 10 - ridge_path (not yet solved)
# TODO: implement

# Step 11 - cv_curve (not yet solved)
# TODO: implement

# Step 12 - lasso_path (not yet solved)
# TODO: implement

# Step 13 - pcr_model (not yet solved)
# TODO: implement

# Step 14 - pls_model (not yet solved)
# TODO: implement

# Step 15 - fit_all (not yet solved)
# TODO: implement

# Step 16 - test_report (not yet solved)
# TODO: implement

