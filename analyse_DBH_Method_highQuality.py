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
#
import hvplot.pandas
# ---------------------------
# read
# ---------------------------
# path to the data folder
dataFolder = r'c:\Path\to\folder\with\data\downloaded\from\Zenodo'

# read the high-quality and outlier filtered DBH measurements:
my_qH = pd.read_parquet(dataFolder + r'\qualHigh_outFiltered_dbh_LaxPark.parquet')

# --------------------------------------
# select a user and make user datasets
# --------------------------------------
my_Auto = my_qH.loc[my_qH['DiameterMethod_dbh0'] == 'CircleFit']
my_Man = my_qH.loc[my_qH['DiameterMethod_dbh0'] == 'Angle']
#
# print some statistics:
my_Auto['errDBH_man'].describe()
my_Man['errDBH_man'].describe()

# ------------------------------------------------
#  AutoMethod: TQ vs. FI get the regression parameters:
# ------------------------------------------------
m_auto, b_auto, r_value_auto, p_value, std_err = scipy.stats.linregress(my_Auto['refDBH_man'], my_Auto['Diameter_dbh0'])
stat_out = pg.corr(x=my_Auto['refDBH_man'], y=my_Auto['Diameter_dbh0'])
my_lm = pg.linear_regression(my_Auto['refDBH_man'], my_Auto['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_Auto['refDBH_man'] - my_Auto['Diameter_dbh0']
my_rmse_auto = ((my_err**2).mean()**0.5)
my_mae_auto = (my_err.abs().mean())

# ------------------------------------------------
#  ManualMethod: TQ vs. FI get the regression parameters:
# ------------------------------------------------
m_man, b_man, r_value_man, p_value, std_err = scipy.stats.linregress(my_Man['refDBH_man'], my_Man['Diameter_dbh0'])
stat_out = pg.corr(x=my_Man['refDBH_man'], y=my_Man['Diameter_dbh0'])
my_lm = pg.linear_regression(my_Man['refDBH_man'], my_Man['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_Man['refDBH_man'] - my_Man['Diameter_dbh0']
my_rmse_man = ((my_err**2).mean()**0.5)
my_mae_man = (my_err.abs().mean())

# ------------------------------------------------
#  AutoMethod: TQ vs. FI get the regression parameters:
# ------------------------------------------------
# select only trees with DBH smaller than 60cm
my_Auto60 = my_Auto.loc[my_Auto['refDBH_man'] <= 65]
my_Auto60_ = my_Auto.loc[my_Auto['refDBH_man'] > 65]
#
m_auto60, b_auto60, r_value_auto60, p_value, std_err = scipy.stats.linregress(my_Auto60['refDBH_man'], my_Auto60['Diameter_dbh0'])
stat_out = pg.corr(x=my_Auto60['refDBH_man'], y=my_Auto60['Diameter_dbh0'])
my_lm = pg.linear_regression(my_Auto60['refDBH_man'], my_Auto60['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_Auto60['refDBH_man'] - my_Auto60['Diameter_dbh0']
my_rmse_auto60 = ((my_err**2).mean()**0.5)
my_mae_auto60 = (my_err.abs().mean())

my_Auto60['errDBH_man_Rel'].mean()

# ------------------------------------------------
# make a joint figure --- Figure 10
# ------------------------------------------------
# make input lists:
myDF = [my_Man, my_Auto]
myX = ['refDBH_man', 'refDBH_man']
myY = ['Diameter_dbh0', 'Diameter_dbh0']
myR = [r_value_man, r_value_auto]
my_m = [m_man, m_auto]
my_b = [b_man, b_auto]
my_MAE = [my_mae_man, my_mae_auto]
my_label = ['(a)', '(b)']
my_xlabel = ['FI DBH [cm]', 'FI DBH [cm]']
my_ylabel = ['Tree-Quest Manual DBH [cm]', 'Tree-Quest Automatic DBH [cm]']
# plotting
fig, axx = plt.subplots(1, 2, figsize=(8, 12./3))
for my_df1, ax, x1, y1, r1, m1, b1, mae1, label1, xlabel1, ylabel1 in (zip(myDF, axx, myX, myY, myR, my_m, my_b, my_MAE, my_label, my_xlabel, my_ylabel)):
    if label1 == '(a)':
        ax.axline((0, 0), slope=1, color='k', linestyle=':', label='1-1 line')
    else:
        ax.plot([0, 100], [0, 100], color='k', linestyle=':')
    #
    sns.scatterplot(data=my_df1, x=x1, y=y1, color='royalblue', ax=ax)
    sns.regplot(data=my_df1, x=x1, y=y1, color='k', scatter=False, ax=ax)
    #
    if label1 == '(b)':
        sns.scatterplot(data=my_Auto60, x=x1, y=y1, color='forestgreen', ax=ax, label= 'DBH < 60 cm')
        ax.annotate('MAE: ' + str("{:.1f}".format(my_mae_auto60)) + ' cm', xy=(10, 100), fontsize=12, weight='bold')
        ax.annotate(', for DBH < 60 cm', xy=(60, 100), style='italic', fontsize=11)
    #
    ax.annotate('$R^2$: ' + str("{:.2f}".format(r1 ** 2)), xy=(10, 140))
    ax.annotate('$y=$' + str("{:.2f}".format(m1)) + '$x$ + ' + str("{:.2f}".format(b1)), xy=(10, 130))
    ax.annotate('MAE: ' + str("{:.1f}".format(mae1)) + ' cm', xy=(10, 120))
    ax.annotate('$n = $' + str("{}".format(len(my_df1[y1]))), xy=(10, 110), size=15)
    ax.text(-0.15, -0.1, label1, transform=ax.transAxes, size=17)
    ax.set_xlabel(xlabel1)
    ax.set_ylabel(ylabel1)
    ax.set_xlim([0, 150])
    ax.set_ylim([0, 150])
    ax.grid(True, lw=0.4)
    ax.set_aspect('equal')
    ax.set_yticks(np.arange(0, 150, 20))
    ax.set_xticks(np.arange(0, 150, 20))

fig.tight_layout()
fig.show()

figPath = r'c:\path\to\folder\where\figures\will\be\stored'
fig.savefig(figPath + r'\Manual_Auto_DBH_LaxPark_highQuality.png', dpi=300)

# --------------------------------------------------------
# Check if normally distributed and do the stat test
# --------------------------------------------------------

# check if the data normally distributed:
my_Man['Diameter_dbh0'].hist(bins=20)
my_Auto['Diameter_dbh0'].hist(bins=20)
#
scipy.stats.probplot(my_Man['Diameter_dbh0'], dist="norm", plot=plt)
scipy.stats.probplot(my_Auto['Diameter_dbh0'], dist="norm", plot=plt)
#
scipy.stats.shapiro(my_Man['Diameter_dbh0'])
# --> not normally distributed
scipy.stats.shapiro(my_Auto['refDBH_man'])
# --> not normally distributed

# check if the data distributions are the same (a Wilcoxon signed-rank test for paired data)
# non-parametric alternative for paired t-test
stat1, p_value1 =scipy.stats.wilcoxon(my_Man['Diameter_dbh0'], my_Man['refDBH_man'], alternative='two-sided')
stat2, p_value2 =scipy.stats.wilcoxon(my_Auto['Diameter_dbh0'], my_Auto['refDBH_man'], alternative='two-sided')
#
results1 = pg.wilcoxon(my_Man['Diameter_dbh0'], my_Man['refDBH_man'])
print(results1)
results2 = pg.wilcoxon(my_Auto['Diameter_dbh0'], my_Auto['refDBH_man'])
print(results2)

# if p-values are < 0.05 --> the measurements are statistically different --> we have to apply the correction functions
# if p-values are >= 0.05 --> not statistically different, Ho rejected

# examine the differences:
ax = pg.plot_blandaltman(my_Man['Diameter_dbh0'], my_Man['refDBH_man'])
plt.tight_layout()

ax = pg.plot_blandaltman(my_Auto['Diameter_dbh0'], my_Auto['refDBH_man'])
plt.tight_layout()

# TQ - FI bias
bias1 = (my_Man['Diameter_dbh0'] - my_Man['refDBH_man']).median()
bias11 = (100*(my_Man['Diameter_dbh0'] - my_Man['refDBH_man'])/my_Man['refDBH_man']).median()

# TQ - TLS bias
bias2 = (my_Auto['Diameter_dbh0'] - my_Auto['refDBH_man']).median()
bias22 = (100*(my_Auto['Diameter_dbh0'] - my_Auto['refDBH_man'])/my_Auto['refDBH_man']).median()

#----------------------------
# test two TQ groups:
#----------------------------
my_Man1 = my_Man.drop_duplicates(subset=['treeId'])
my_Auto1 = my_Auto.drop_duplicates(subset=['treeId'])

my_ManAutoPaired = pd.merge(my_Auto1, my_Man1, on='treeId', how='left', validate="one_to_one",
                            suffixes=('_a', '_m'))
my_ManAutoPaired.shape

my_ManAutoPaired.dropna(subset=['Diameter_dbh0_m'])

results3 = pg.wilcoxon(my_ManAutoPaired['Diameter_dbh0_m'], my_ManAutoPaired['Diameter_dbh0_a'])
print(results3)

# TQ - TLS bias
bias3 = (my_ManAutoPaired['Diameter_dbh0_m'] - my_ManAutoPaired['Diameter_dbh0_a']).median()
bias33 = (100*(my_ManAutoPaired['Diameter_dbh0_m'] - my_ManAutoPaired['Diameter_dbh0_a'])/my_ManAutoPaired['Diameter_dbh0_a']).median()