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
# read the ard data Stadtpark:
my_df = pd.read_parquet(dataFolder + r'\ard_gdf_2024_04_14_Stadtpark.parquet')

# ###############################
# analyse Geo-Quest measurements
# ###############################
# get the regression parameters:
m, b, r_value, p_value, std_err = scipy.stats.linregress(my_df['refTH_man'], my_df['TreeHeight'])
stat_out = pg.corr(x=my_df['refTH_man'], y=my_df['TreeHeight'])
my_lm = pg.linear_regression(my_df['refTH_man'], my_df['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_df, x='refTH_man', y='TreeHeight', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_df, x="refTH_man", y="TreeHeight", ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(8, 30))
ax.set_xlabel('FI TH')
ax.set_ylabel('Tree-Quest TH')
fig.show()

# ------------------------------------------------
# find and exclude outliers (TQ-FI)
# ------------------------------------------------
# get the difference from the mean diameter:
myMultiDiff = pd.Series(my_lm.residuals_)
# get iqr:
threshold = 0.5
q3 = myMultiDiff.quantile(0.75)
q1 = myMultiDiff.quantile(0.25)
iqr =  q3 - q1
lower_bound = q1 - threshold * iqr
upper_bound = q3 + threshold * iqr
# remove according to the iqr:
ind_goodPts = (myMultiDiff >= lower_bound) & (myMultiDiff <= upper_bound)
ind_outPts = (myMultiDiff < lower_bound) | (myMultiDiff > upper_bound)
#
my_tqIn = my_df[ind_goodPts.to_list()]
my_tqOut = my_df[ind_outPts.to_list()]

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_tqIn, x='refTH_man', y='TreeHeight', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_tqIn, x="refTH_man", y="TreeHeight", color='royalblue' , ax=ax)
sns.scatterplot(data=my_tqOut, x="refTH_man", y="TreeHeight", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(8, 30))
ax.set_xlabel('FI TH')
ax.set_ylabel('Tree-Quest TH')
fig.show()

# ------------------------------------------------
#  get the regression parameters:
# ------------------------------------------------
m_tq, b_tq, r_value_tq, p_value, std_err = scipy.stats.linregress(my_tqIn['refTH_man'], my_tqIn['TreeHeight'])
stat_out = pg.corr(x=my_tqIn['refTH_man'], y=my_tqIn['TreeHeight'])
my_lm = pg.linear_regression(my_tqIn['refTH_man'], my_tqIn['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [m]:
my_err =  my_tqIn['refTH_man'] - my_tqIn['TreeHeight']
my_rmse_tq = ((my_err**2).mean()**0.5)
my_mae_tq = (my_err.abs().mean())

my_tqIn['errTH_man_Rel'].mean()

# ###############################
# analyse WorkingTrees measurements
# ###############################
# get the regression parameters:
m, b, r_value, p_value, std_err = scipy.stats.linregress(my_df['refTH_man'], my_df['Height_WorkingTrees'])
stat_out = pg.corr(x=my_df['refTH_man'], y=my_df['Height_WorkingTrees'])
my_lm = pg.linear_regression(my_df['refTH_man'], my_df['Height_WorkingTrees'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_df, x='refTH_man', y='Height_WorkingTrees', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_df, x="refTH_man", y="Height_WorkingTrees",  palette='royalblue', ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(10, 25))
ax.set_xlabel('FI TH')
ax.set_ylabel('WT TH')
fig.show()

# ------------------------------------------------
# find and exclude outliers
# ------------------------------------------------
# get the difference from the mean diameter:
myMultiDiff = pd.Series(my_lm.residuals_)
# get iqr:
threshold = 0.5
q3 = myMultiDiff.quantile(0.75)
q1 = myMultiDiff.quantile(0.25)
iqr =  q3 - q1
lower_bound = q1 - threshold * iqr
upper_bound = q3 + threshold * iqr
# remove according to the iqr:
ind_goodPts = (myMultiDiff >= lower_bound) & (myMultiDiff <= upper_bound)
ind_outPts = (myMultiDiff < lower_bound) | (myMultiDiff > upper_bound)
#
my_wtIn = my_df[ind_goodPts.to_list()]
my_wtOut = my_df[ind_outPts.to_list()]

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_wtIn, x='refTH_man', y='Height_WorkingTrees', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_wtIn, x="refTH_man", y="Height_WorkingTrees", color='royalblue' , ax=ax)
sns.scatterplot(data=my_wtOut, x="refTH_man", y="Height_WorkingTrees", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(10, 25))
ax.set_xlabel('FI TH')
ax.set_ylabel('WT TH')
fig.show()

# ------------------------------------------------
#  get the regression parameters:
# ------------------------------------------------
m_wt, b_wt, r_value_wt, p_value, std_err = scipy.stats.linregress(my_wtIn['refTH_man'], my_wtIn['Height_WorkingTrees'])
stat_out = pg.corr(x=my_wtIn['refTH_man'], y=my_wtIn['Height_WorkingTrees'])
my_lm = pg.linear_regression(my_wtIn['refTH_man'], my_wtIn['Height_WorkingTrees'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [m]:
my_err =  my_wtIn['refTH_man'] - my_wtIn['Height_WorkingTrees']
my_rmse_wt = ((my_err**2).mean()**0.5)
my_mae_wt = (my_err.abs().mean())

# ###############################
# analyse GlobOBSERVE measurements
# ###############################
# get the regression parameters:
m, b, r_value, p_value, std_err = scipy.stats.linregress(my_df['refTH_man'], my_df['Height_Globeobserver'])
stat_out = pg.corr(x=my_df['refTH_man'], y=my_df['Height_Globeobserver'])
my_lm = pg.linear_regression(my_df['refTH_man'], my_df['Height_Globeobserver'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_df, x='refTH_man', y='Height_Globeobserver', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_df, x="refTH_man", y="Height_Globeobserver",  palette='royalblue', ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(10, 25))
ax.set_xlabel('FI TH')
ax.set_ylabel('GL TH')
fig.show()

# ------------------------------------------------
# find and exclude outliers
# ------------------------------------------------
# get the difference from the mean diameter:
myMultiDiff = pd.Series(my_lm.residuals_)
# get iqr:
threshold = 0.5
q3 = myMultiDiff.quantile(0.75)
q1 = myMultiDiff.quantile(0.25)
iqr =  q3 - q1
lower_bound = q1 - threshold * iqr
upper_bound = q3 + threshold * iqr
# remove according to the iqr:
ind_goodPts = (myMultiDiff >= lower_bound) & (myMultiDiff <= upper_bound)
ind_outPts = (myMultiDiff < lower_bound) | (myMultiDiff > upper_bound)
#
my_glIn = my_df[ind_goodPts.to_list()]
my_glOut = my_df[ind_outPts.to_list()]

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_glIn, x='refTH_man', y='Height_Globeobserver', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_glIn, x="refTH_man", y="Height_Globeobserver", color='royalblue' , ax=ax)
sns.scatterplot(data=my_glOut, x="refTH_man", y="Height_Globeobserver", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(60, 120))
ax.set_xlabel('FI TH')
ax.set_ylabel('GL TH')
fig.show()

# ------------------------------------------------
#  get the regression parameters:
# ------------------------------------------------
m_gl, b_gl, r_value_gl, p_value, std_err = scipy.stats.linregress(my_glIn['refTH_man'], my_glIn['Height_Globeobserver'])
stat_out = pg.corr(x=my_glIn['refTH_man'], y=my_glIn['Height_Globeobserver'])
my_lm = pg.linear_regression(my_glIn['refTH_man'], my_glIn['Height_Globeobserver'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [m]:
my_err =  my_glIn['refTH_man'] - my_glIn['Height_Globeobserver']
my_rmse_gl = ((my_err**2).mean()**0.5)
my_mae_gl = (my_err.abs().mean())

# ------------------------------------------------
# make a joint figure ---- Figure 12
# ------------------------------------------------
# make input lists:
myDF = [my_tqIn, my_glIn , my_wtIn]
myX = ['refTH_man', 'refTH_man', 'refTH_man']
myY = ['TreeHeight', 'Height_Globeobserver', 'Height_WorkingTrees']
myR = [r_value_tq, r_value_gl, r_value_wt]
my_m = [m_tq, m_gl, m_wt]
my_b = [b_tq, b_gl, b_wt]
my_MAE = [my_mae_tq, my_mae_gl, my_mae_wt]
my_label = ['(a)', '(b)', '(c)']
my_xlabel = ['FI TH [m]', 'FI TH [m]', 'FI TH [m]']
my_ylabel = ['Tree-Quest TH [m]', 'GLOBE Observer TH [m]', 'Working Trees TH [m]']
# plotting
fig, axx = plt.subplots(1, 3, figsize=(12, 12./3))
for my_df1, ax, x1, y1, r1, m1, b1, mae1, label1, xlabel1, ylabel1 in (zip(myDF, axx, myX, myY, myR, my_m, my_b, my_MAE, my_label, my_xlabel, my_ylabel)):
    ax.axline((0, 0), slope=1, color='k', linestyle=':', label='1-1 line')
    sns.scatterplot(data=my_df1, x=x1, y=y1, color='royalblue', ax=ax)
    sns.regplot(data=my_df1, x=x1, y=y1, color='k', scatter=False, ax=ax)
    ax.annotate('$R^2$: ' + str("{:.2f}".format(r1 ** 2)), xy=(8.4, 26.5))
    ax.annotate('$y=$' + str("{:.2f}".format(m1)) + '$x$ + ' + str("{:.2f}".format(b1)), xy=(8.4, 25))
    ax.annotate('MAE: ' + str("{:.1f}".format(mae1)) + ' m', xy=(8.4, 23.5))
    ax.annotate('$n = $' + str("{}".format(len(my_df1[y1]))), xy=(8.4, 22), size=15)
    ax.text(-0.15, -0.1, label1, transform=ax.transAxes, size=17)
    ax.set_xlabel(xlabel1)
    ax.set_ylabel(ylabel1)
    ax.set_xlim([8, 22])
    ax.set_ylim([8, 28])
    ax.grid(True, lw=0.4)
    #ax.set_aspect('equal')
    ax.set_yticks(np.arange(10, 30, 5))
    ax.set_xticks(np.arange(10, 25, 5))
    ax.legend(loc='lower right')

fig.tight_layout()
fig.show()

figPath = r'c:\path\to\folder\where\figures\will\be\stored'
fig.savefig(figPath + r'\TH_Stadtpark_otherApps.png', dpi=300)

# --------------------------------------------------------
# Check if normally distributed and do the stat test
# --------------------------------------------------------

# check if the data normally distributed:
my_tqIn['TreeHeight'].hist(bins=20)
my_glIn['TreeHeight'].hist(bins=20)
my_wtIn['TreeHeight'].hist(bins=20)
#
scipy.stats.probplot(my_wtIn['TreeHeight'], dist="norm", plot=plt)
#
scipy.stats.shapiro(my_glIn['TreeHeight'])
# --> not normally distributed
scipy.stats.shapiro(my_wtIn['refTH_man'])
# --> not normally distributed

# check if the data distributions are the same (a Wilcoxon signed-rank test for paired data)
# non-parametric alternative for paired t-test
stat1, p_value1 =scipy.stats.wilcoxon(my_tqIn['TreeHeight'], my_tqIn['refTH_man'], alternative='two-sided')
stat2, p_value2 =scipy.stats.wilcoxon(my_glIn['TreeHeight'], my_glIn['refTH_man'], alternative='two-sided')
stat3, p_value3 =scipy.stats.wilcoxon(my_wtIn['TreeHeight'], my_wtIn['refTH_man'], alternative='two-sided')
#
results1 = pg.wilcoxon(my_tqIn['TreeHeight'], my_tqIn['refTH_man'])
print(results1)
results2 = pg.wilcoxon(my_glIn['TreeHeight'], my_glIn['refTH_man'])
print(results2)
results3 = pg.wilcoxon(my_wtIn['TreeHeight'], my_wtIn['refTH_man'])
print(results3)

# if p-values are < 0.05 --> the measurements are statistically different --> we have to apply the correction functions
# if p-values are >= 0.05 --> not statistically different, Ho rejected

# examine the differences:
ax = pg.plot_blandaltman(my_tqIn['TreeHeight'], my_tqIn['refTH_man'])
plt.tight_layout()

# TQ - FI bias
bias1 = (my_tqIn['TreeHeight'] - my_tqIn['refTH_man']).median()
bias11 = (100*(my_tqIn['TreeHeight'] - my_tqIn['refTH_man'])/my_tqIn['refTH_man']).median()

# TQ - TLS bias
bias2 = (my_glIn['TreeHeight'] - my_glIn['refTH_man']).median()
bias22 = (100*(my_glIn['TreeHeight'] - my_glIn['refTH_man'])/my_glIn['refTH_man']).median()

# TLS - FI bias
bias3 = (my_wtIn['TreeHeight'] - my_wtIn['refTH_man']).median()
bias33 = (100*(my_wtIn['TreeHeight'] - my_wtIn['refTH_man'])/my_wtIn['refTH_man']).median()

#----------------------------
# test all tree TQ gropus:
#----------------------------
minNum = min(len(my_tqIn['TreeHeight']), len(my_glIn['TreeHeight']), len(my_wtIn['TreeHeight']))

#
mySatResTQ_groups = scipy.stats.friedmanchisquare(my_tqIn['TreeHeight'].sample(n=minNum),
                                                  my_glIn['TreeHeight'].sample(n=minNum),
                                                  my_wtIn['TreeHeight'].sample(n=minNum)
                                                  )

mySatResTQ_groups.pvalue
mySatResTQ_groups.statistic