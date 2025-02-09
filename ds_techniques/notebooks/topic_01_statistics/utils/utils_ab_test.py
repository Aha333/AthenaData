"""
Intuition:
--------------------------------
Delta Method 

Context - A/B testing , delta method for ratio of two normal distributions

The ratio of two normal distributions follows a Cauchy distribution if the denominator's mean is non-zero.
However, this is only true under certain conditions:
1. The variables must be independent
2. The denominator must not be close to zero (to avoid division by zero issues)
3. The denominator should have small variance relative to its mean

在 **Delta 方法** 中，确实存在两个关键的假设：

### 1. **样本均值本身近似正态分布**：

- Delta 方法依赖于 **中心极限定理（Central Limit Theorem, CLT）**，它表明，当样本量足够大时，样本均值的分布会趋向于正态分布，不管原始数据的分布是什么样的。
- 也就是说，**对于大样本**，即使你原始数据不服从正态分布，样本均值（例如，X1 和 X2）也会接近正态分布，因此我们可以使用 **正态分布** 来近似样本均值的分布。
    
    X1‾\overline{X_1}
    
    X2‾\overline{X_2}
    

### 2. **两个正态分布的比率近似正态分布**：

- 当我们使用 Delta 方法来计算两个正态分布比率的 **标准误差** 时，我们假设 **比率的分布在大样本情况下是正态的**。这意味着，在比率计算中，假设 **样本量足够大**，**样本均值的比率** 近似服从正态分布。
- 这个假设来自于 **Delta 方法的线性近似**：我们通过对比率函数（X2X1）进行一阶泰勒展开，得到该比率的近似标准误差，进而推断其置信区间。对于大样本量，Delta 方法的线性近似在统计上是合理的，即使原始数据不完全是正态分布。
    
    X1X2\frac{X_1}{X_2}
    

### 简单总结：

- **第一个假设**：中心极限定理告诉我们，当样本量足够大时，样本均值会接近正态分布。
- **第二个假设**：通过 Delta 方法的泰勒展开近似，假设样本均值比率的分布也是正态的，尤其是在 **大样本** 的条件下。

"""

import numpy as np
from scipy import stats
# For the delta method:
# - The individual variables should be approximately normal, but the ratio itself doesn't need to be
# - This is because the delta method uses a first-order Taylor expansion around the means
#   of the individual variables
#
# For bootstrap:
# - No normality assumptions needed since it's non-parametric
# - Works with any distribution as it uses empirical sampling
#
# For Monte Carlo:
# - Assumes normality of individual variables since we sample from normal distributions
# - The ratio distribution itself doesn't need to be normal


def progress_decorator(func):
    """
    Decorator to show progress bar for iterative calculations using tqdm.
    """
    def wrapper(*args, **kwargs):
        from tqdm import tqdm
        
        # Get number of iterations from kwargs
        n_iter = kwargs.get('n_bootstrap', kwargs.get('n_simulations', 1000))
        
        ratios = []
        # Create progress bar
        with tqdm(total=n_iter, desc=f'Running {func.__name__}') as pbar:
            for _ in range(n_iter):
                if 'n_bootstrap' in kwargs:
                    # Bootstrap case
                    data1, data2 = args[0], args[1]
                    sample1 = np.random.choice(data1, size=len(data1), replace=True)
                    sample2 = np.random.choice(data2, size=len(data2), replace=True)
                    ratios.append(np.mean(sample1) / np.mean(sample2))
                else:
                    # Monte Carlo case
                    mean1, mean2, std1, std2, n1, n2 = args
                    sample1 = np.random.normal(mean1, std1/np.sqrt(n1))
                    sample2 = np.random.normal(mean2, std2/np.sqrt(n2))
                    ratios.append(sample1 / sample2)
                pbar.update(1)
                
        return np.percentile(ratios, [kwargs.get('alpha', 0.05)/2*100, 
                                    (1-kwargs.get('alpha', 0.05)/2)*100])
    
    return wrapper


def delta_method_ci(mean1, mean2, std1, std2, n1, n2, alpha=0.05):
    """Calculate CI using delta method for ratio of two normal distributions"""
    ratio = mean1 / mean2
    # Variance of ratio using delta method
    var_ratio = (std1**2/n1)/(mean2**2) + (mean1**2 * std2**2/n2)/(mean2**4)
    z_score = stats.norm.ppf(1 - alpha/2)
    ci_lower = ratio - z_score * np.sqrt(var_ratio)
    ci_upper = ratio + z_score * np.sqrt(var_ratio)
    return ci_lower, ci_upper

@progress_decorator
def bootstrap_ci(data1, data2, n_bootstrap=1000, alpha=0.05):
    """Calculate CI using bootstrap sampling"""
    pass

@progress_decorator
def monte_carlo_ci(mean1, mean2, std1, std2, n1, n2, n_simulations=1000, alpha=0.05):
    """Calculate CI using Monte Carlo simulation"""
    pass

def calculate_ratio_confidence_intervals(df, value_col, alpha=0.05, n_bootstrap=1000, n_simulations=1000):
    """
    Calculate confidence intervals for the ratio of treatment to control using three methods:
    Delta method, Bootstrap, and Monte Carlo simulation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the A/B test data with 'group' and 'converted' columns
    alpha : float
        Significance level (default: 0.05 for 95% confidence interval)
    n_bootstrap : int
        Number of bootstrap iterations (default: 1000)
    n_simulations : int
        Number of Monte Carlo simulations (default: 1000)
        
    Returns:
    --------
    dict : Dictionary containing the ratio and confidence intervals from all three methods
    """
    # Split data into treatment and control groups
    treatment_data = df[df['group'] == 'treatment'][value_col]
    control_data = df[df['group'] == 'control'][value_col]
    
    # Calculate statistics
    treatment_mean = treatment_data.mean()
    control_mean = control_data.mean()
    treatment_std = treatment_data.std()
    control_std = control_data.std()
    n_treatment = len(treatment_data)
    n_control = len(control_data)

    
    
    print(f"Control Group - Mean: {control_mean:.2f}, Std: {control_std:.2f}")
    print(f"Treatment Group - Mean: {treatment_mean:.2f}, Std: {treatment_std:.2f}")
    print(f"Treatment/Control Rel Lift: {treatment_mean/control_mean-1:.2f}")
    # Calculate ratio
    ratio = treatment_mean / control_mean
    
    # Calculate CIs using all three methods
    delta_ci = delta_method_ci(
        treatment_mean, control_mean,
        treatment_std, control_std,
        n_treatment, n_control,
        alpha=alpha
    )
    
    bootstrap_ci_result = bootstrap_ci(
        treatment_data, control_data,
        n_bootstrap=n_bootstrap,
        alpha=alpha
    )
    
    monte_carlo_ci_result = monte_carlo_ci(
        treatment_mean, control_mean,
        treatment_std, control_std,
        n_treatment, n_control,
        n_simulations=n_simulations,
        alpha=alpha
    )
    
    return {
        'ratio': ratio,
        'delta_method_ci': delta_ci,
        'bootstrap_ci': bootstrap_ci_result,
        'monte_carlo_ci': monte_carlo_ci_result
    }

def calculate_sample_size(baseline_metric, mde_relative, metric_type='conversion', variance=None, 
                         alpha=0.05, power=0.8):
    """
    Calculate required sample size per group for an A/B test for both conversion and continuous metrics.
    
    Parameters:
    -----------
    baseline_metric : float
        Current (control) metric value - either conversion rate or mean for continuous
    mde_relative : float
        Minimum detectable effect as relative change (e.g., 0.1 for 10% increase)
        For example:
        - If baseline is 0.2 (20%) and mde_relative is 0.1, we want to detect a change to 0.22 (22%)
        - If baseline is 100 and mde_relative is 0.15, we want to detect a change to 115
    metric_type : str, optional
        Type of metric - either 'conversion' or 'continuous', default 'conversion'
    variance : float, optional
        Variance of the continuous metric, required if metric_type='continuous'
    alpha : float, optional
        Significance level (Type I error rate), default 0.05
    power : float, optional
        Statistical power (1 - Type II error rate), default 0.8
        
    Returns:
    --------
    int : Required sample size per variant
    
    Raises:
    -------
    ValueError
        If metric_type='continuous' and variance is None
        If metric_type is not 'conversion' or 'continuous'
        If mde_relative is not between -1 and 1
    """
    if not -1 < mde_relative < 1:
        raise ValueError("mde_relative must be between -1 and 1")
        
    # Standard normal distribution values
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate absolute change based on relative MDE
    absolute_change = baseline_metric * mde_relative
    treatment_metric = baseline_metric + absolute_change
    
    if metric_type == 'conversion':
        # Pooled standard error for conversion metric
        p_pooled = (baseline_metric + treatment_metric) / 2
        se_pooled = np.sqrt(2 * p_pooled * (1 - p_pooled))
        
    elif metric_type == 'continuous':
        if variance is None:
            raise ValueError("Variance must be provided for continuous metrics")
        # Standard error for continuous metric
        se_pooled = np.sqrt(2 * variance)
        
    else:
        raise ValueError("metric_type must be either 'conversion' or 'continuous'")
    
    # Calculate sample size per group
    numerator = (z_alpha + z_beta)**2 * se_pooled**2
    denominator = absolute_change**2
    
    sample_size = np.ceil(numerator / denominator)
    
    return int(sample_size)


