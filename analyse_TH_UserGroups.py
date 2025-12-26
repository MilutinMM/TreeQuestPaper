import numpy as np
import pandas as pd
import geopandas as gpd
import pingouin as pg
import scipy.stats
#
import seaborn as sns
sns.set_theme(style='white', font_scale=1.2)
#
import matplotlib
import matplotlib.pyplot as plt
#
# set the matplotlib backend to enable plotting
matplotlib.use('TkAgg',force=True)

# ---------------------------
# read data
# ---------------------------
# path to the data folder
dataFolder = r'c:\Path\to\folder\with\data\downloaded\from\Zenodo'
# read the high quality:
my_qHML = pd.read_parquet(dataFolder + r'\qHML_TH_LaxPark_UserGroups.parquet')

# -------------------------------
# Distribution per quality group
# -------------------------------
# group by quality and count (%) per user group:
myStat = my_qHML.groupby('qMin').apply(lambda aa: 100*aa.groupby('UserCatName').count()/aa.shape[0])['CreatorId']
# prepare for plotting
myStat2 = myStat.unstack()
# rename:
myStat2.columns.set_names('User Groups', inplace=True)
myStat2.index.set_names('', inplace=True)
# rename index value
myStat2.rename(index={'Med':'Medium'}, inplace=True)
# reorder
myStat2 = myStat2.reindex(index = ['High','Medium','Low'])

# plotting:
fig, ax = plt.subplots(1, 1, figsize=(12./3, 12./3))
myStat2.plot(kind='bar', ax=ax)
ax.grid(True, lw=0.4, axis='y')
ax.set_ylim([0, 75])
ax.set_yticks(np.arange(0, 75, 10))
ax.set_ylabel('Num. of Measurements [%]')
ax.set_xticklabels(['High', 'Medium', 'Low'], rotation=20, ha='right', rotation_mode='anchor')
fig.tight_layout()
fig.show()

# ##########################
# analyse Experts group
# ##########################
# select medium-quality
my_Exp = my_qHML[my_qHML['UserCatName'] == 'Experts']

m_exp, b_exp, r_value_exp, p_value, std_err = scipy.stats.linregress(my_Exp['refTH_man'], my_Exp['TreeHeight'])
stat_out = pg.corr(x=my_Exp['refTH_man'], y=my_Exp['TreeHeight'])
my_lm = pg.linear_regression(my_Exp['refTH_man'], my_Exp['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [m]:
my_err =  my_Exp['refTH_man'] - my_Exp['TreeHeight']
my_rmse_exp = ((my_err**2).mean()**0.5)
my_mae_exp = (my_err.abs().mean())

# ############################
# analyse Practitioners group
# ############################
# select medium-quality
my_Pra = my_qHML[my_qHML['UserCatName'] == 'Practitioners']

m_pra, b_pra, r_value_pra, p_value, std_err = scipy.stats.linregress(my_Pra['refTH_man'], my_Pra['TreeHeight'])
stat_out = pg.corr(x=my_Pra['refTH_man'], y=my_Pra['TreeHeight'])
my_lm = pg.linear_regression(my_Pra['refTH_man'], my_Pra['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [m]:
my_err =  my_Pra['refTH_man'] - my_Pra['TreeHeight']
my_rmse_pra = ((my_err**2).mean()**0.5)
my_mae_pra = (my_err.abs().mean())

# ############################
# analyse Students group
# ############################
# select medium-quality
my_Stu = my_qHML[my_qHML['UserCatName'] == 'Students']

m_stu, b_stu, r_value_stu, p_value, std_err = scipy.stats.linregress(my_Stu['refTH_man'], my_Stu['TreeHeight'])
stat_out = pg.corr(x=my_Stu['refTH_man'], y=my_Stu['TreeHeight'])
my_lm = pg.linear_regression(my_Stu['refTH_man'], my_Stu['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [m]:
my_err =  my_Stu['refTH_man'] - my_Stu['TreeHeight']
my_rmse_stu = ((my_err**2).mean()**0.5)
my_mae_stu = (my_err.abs().mean())

# ------------------------------------------------
# make a joint figure ---- Figure 9
# ------------------------------------------------
# make input lists:
myDF = [my_Exp, my_Pra , my_Stu]
myX = ['refTH_man', 'refTH_man', 'refTH_man']
myY = ['TreeHeight', 'TreeHeight', 'TreeHeight']
myR = [r_value_exp, r_value_pra, r_value_stu]
my_m = [m_exp, m_pra, m_stu]
my_b = [b_exp, b_pra, b_stu]
my_MAE = [my_mae_exp, my_mae_pra, my_mae_stu]
my_label = ['(a)', '(b)', '(c)']
my_xlabel = ['FI TH [m]', 'FI TH [m]', 'FI TH [m]']
my_ylabel = ['Experts Group TH [m]', 'Practitioners Group TH [m]', 'Students Group TH [m]']
# colors:
myColors = ['tab:green', 'tab:blue', 'tab:orange']
# plotting
fig, axx = plt.subplots(1, 3, figsize=(12, 12./3))
for my_df1, ax, x1, y1, r1, m1, b1, mae1, label1, xlabel1, ylabel1 in (zip(myDF, axx, myX, myY, myR, my_m, my_b, my_MAE, my_label, my_xlabel, my_ylabel)):
    ax.axline((0, 0), slope=1, color='k', linestyle=':')
    sns.scatterplot(data=my_df1, x=x1, y=y1, hue='qMin', color='royalblue', palette=myColors ,ax=ax)
    sns.regplot(data=my_df1, x=x1, y=y1, color='k', scatter=False, ax=ax)
    ax.annotate('$R^2$: ' + str("{:.2f}".format(r1 ** 2)), xy=(5, 32))
    ax.annotate('$y=$' + str("{:.2f}".format(m1)) + '$x$ + ' + str("{:.2f}".format(b1)), xy=(5, 30))
    ax.annotate('MAE: ' + str("{:.1f}".format(mae1)) + ' m', xy=(5, 28))
    ax.annotate('$n = $' + str("{}".format(len(my_df1[y1]))), xy=(5, 26), size=15)
    ax.text(-0.15, -0.1, label1, transform=ax.transAxes, size=17)
    ax.set_xlabel(xlabel1)
    ax.set_ylabel(ylabel1)
    ax.set_xlim([3, 35])
    ax.set_ylim([3, 35])
    ax.grid(True, lw=0.4)
    ax.set_aspect('equal')
    ax.set_yticks(np.arange(5, 35, 5))
    ax.set_xticks(np.arange(5, 35, 5))
    # remove the legend title:
    ax.legend_.set_title('')

fig.tight_layout()
fig.show()

figPath = r'c:\path\to\folder\where\figures\will\be\stored'
fig.savefig(figPath + r'\TH_LaxPark_UserGroups.png', dpi=300)


# --------------------------------------------------------
# Check if normally distributed and do the stat test
# --------------------------------------------------------

# check if the data normally distributed:
my_Exp['TreeHeight'].hist(bins=20)
my_Pra['TreeHeight'].hist(bins=20)
my_Stu['TreeHeight'].hist(bins=20)
#
scipy.stats.probplot(my_Stu['TreeHeight'], dist="norm", plot=plt)
#
scipy.stats.shapiro(my_Exp['TreeHeight'])
#
scipy.stats.shapiro(my_Pra['TreeHeight'])
# --> not normally distributed
scipy.stats.shapiro(my_Stu['TreeHeight'])
# --> not normally distributed

# check if the data distributions are the same (a Wilcoxon signed-rank test for paired data)
# non-parametric alternative for paired t-test
stat1, p_value1 =scipy.stats.wilcoxon(my_Exp['TreeHeight'], my_Exp['refTH_man'], alternative='two-sided')
stat2, p_value2 =scipy.stats.wilcoxon(my_Pra['TreeHeight'], my_Pra['refTH_man'], alternative='two-sided')
stat3, p_value3 =scipy.stats.wilcoxon(my_Stu['TreeHeight'], my_Stu['refTH_man'], alternative='two-sided')
#
results1 = pg.wilcoxon(my_Exp['TreeHeight'], my_Exp['refTH_man'])
print(results1)
results2 = pg.wilcoxon(my_Pra['TreeHeight'], my_Pra['refTH_man'])
print(results2)
results3 = pg.wilcoxon(my_Stu['TreeHeight'], my_Stu['refTH_man'])
print(results3)

# if p-values are < 0.05 --> the measurements are statistically different --> we have to apply the correction functions
# if p-values are >= 0.05 --> not statistically different, Ho rejected

# examine the differences:
ax = pg.plot_blandaltman(my_Exp['TreeHeight'], my_Exp['refTH_man'])
plt.tight_layout()

# TQ - FI bias
bias1 = (my_Exp['TreeHeight'] - my_Exp['refTH_man']).median()
bias11 = (100*(my_Exp['TreeHeight'] - my_Exp['refTH_man'])/my_Exp['refTH_man']).median()

# TQ - TLS bias
bias2 = (my_Pra['TreeHeight'] - my_Pra['refTH_man']).median()
bias22 = (100*(my_Pra['TreeHeight'] - my_Pra['refTH_man'])/my_Pra['refTH_man']).median()

# TLS - FI bias
bias3 = (my_Stu['TreeHeight'] - my_Stu['refTH_man']).median()
bias33 = (100*(my_Stu['TreeHeight'] - my_Stu['refTH_man'])/my_Stu['refTH_man']).median()

#----------------------------
# test all tree TQ gropus:
#----------------------------
minNum = min(len(my_Exp['TreeHeight']), len(my_Pra['TreeHeight']), len(my_Stu['TreeHeight']))

#
mySatResTQ_groups = scipy.stats.friedmanchisquare(my_Exp['TreeHeight'].sample(n=minNum),
                                                  my_Pra['TreeHeight'].sample(n=minNum),
                                                  my_Stu['TreeHeight'].sample(n=minNum)
                                                  )

mySatResTQ_groups.pvalue
mySatResTQ_groups.statistic
