import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data with sheet name
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Compare with FE results\Statistical test\Statistical test_DFF_input.xlsx'
sheet_name = 'high difference'  # CHANGE to your sheet name, e.g., 'Data', 0, or 1
df = pd.read_excel(file_path, sheet_name=sheet_name).dropna(subset=['FEA', 'ML Prediction'])
print(f"Loaded {len(df)} paired samples from '{sheet_name}'")

fe = df['FEA'].values
ml = df['ML Prediction'].values
diff = fe - ml

# Normality
print(f'Shapiro-Wilk p-value: {stats.shapiro(diff)[1]:.4f}')

# Paired t-test
t_result = stats.ttest_rel(fe, ml)
print(f't-test: t={t_result.statistic:.2f}, p={t_result.pvalue:.4f}')
ci_95 = t_result.confidence_interval(0.95)
print(f'95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}] days')

# Wilcoxon
wilcox_result = stats.wilcoxon(fe, ml)
print(f'Wilcoxon: statistic={wilcox_result.statistic}, p={wilcox_result.pvalue:.4f}')

# Errors
mae = np.mean(np.abs(diff))
rmse = np.sqrt(np.mean(diff**2))
print(f'MAE: {mae:.2f} days, RMSE: {rmse:.2f} days')

# Bland-Altman
ax = sm.graphics.mean_diff_plot(fe, ml)
ax.set_title('Bland-Altman Plot: FE vs ML Days to Failure')
plt.show()

# Wilcoxon bootstrap CI
print(f'Wilcoxon p= {wilcox_result.pvalue:.4f} (95% confidence level)')
boot_res = stats.bootstrap((diff,), np.median, confidence_level=0.95, n_resamples=10000)
ci_95_wilcox = boot_res.confidence_interval
print(f'Wilcoxon Bootstrap 95% CI: [{ci_95_wilcox[0]:.2f}, {ci_95_wilcox[1]:.2f}] days')
