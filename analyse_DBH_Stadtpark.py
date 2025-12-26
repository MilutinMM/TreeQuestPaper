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
m, b, r_value, p_value, std_err = scipy.stats.linregress(my_df['refDBH_man'], my_df['Diameter_dbh0'])
stat_out = pg.corr(x=my_df['refDBH_man'], y=my_df['Diameter_dbh0'])
my_lm = pg.linear_regression(my_df['refDBH_man'], my_df['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_df, x='refDBH_man', y='Diameter_dbh0', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_df, x="refDBH_man", y="Diameter_dbh0",  palette='royalblue', ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(60, 120))
ax.set_xlabel('FI DBH')
ax.set_ylabel('Tree-Quest DBH')
fig.show()

# ------------------------------------------------
# find and exclude outliers (TQ-FI)
# ------------------------------------------------
# get the difference from the mean diameter:
myMultiDiff = pd.Series(my_lm.residuals_)
# get iqr:
threshold = 0.8
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
sns.regplot(data=my_tqIn, x='refDBH_man', y='Diameter_dbh0', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_tqIn, x="refDBH_man", y="Diameter_dbh0", color='royalblue' , ax=ax)
sns.scatterplot(data=my_tqOut, x="refDBH_man", y="Diameter_dbh0", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(60, 120))
ax.set_xlabel('FI DBH')
ax.set_ylabel('Tree-Quest DBH')
fig.show()

# ------------------------------------------------
#  get the regression parameters:
# ------------------------------------------------
m_tq, b_tq, r_value_tq, p_value, std_err = scipy.stats.linregress(my_tqIn['refDBH_man'], my_tqIn['Diameter_dbh0'])
stat_out = pg.corr(x=my_tqIn['refDBH_man'], y=my_tqIn['Diameter_dbh0'])
my_lm = pg.linear_regression(my_tqIn['refDBH_man'], my_tqIn['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_tqIn['refDBH_man'] - my_tqIn['Diameter_dbh0']
my_rmse_tq = ((my_err**2).mean()**0.5)
my_mae_tq = (my_err.abs().mean())

# ###############################
# analyse WorkingTrees measurements
# ###############################
# get the regression parameters:
m, b, r_value, p_value, std_err = scipy.stats.linregress(my_df['refDBH_man'], my_df['DBH_WorkingTrees'])
stat_out = pg.corr(x=my_df['refDBH_man'], y=my_df['DBH_WorkingTrees'])
my_lm = pg.linear_regression(my_df['refDBH_man'], my_df['DBH_WorkingTrees'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_df, x='refDBH_man', y='DBH_WorkingTrees', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_df, x="refDBH_man", y="DBH_WorkingTrees",  palette='royalblue', ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(60, 120))
ax.set_xlabel('FI DBH')
ax.set_ylabel('WT DBH')
fig.show()

# ------------------------------------------------
# find and exclude outliers
# ------------------------------------------------
# get the difference from the mean diameter:
myMultiDiff = pd.Series(my_lm.residuals_)
# get iqr:
threshold = 0.8
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
sns.regplot(data=my_wtIn, x='refDBH_man', y='DBH_WorkingTrees', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_wtIn, x="refDBH_man", y="DBH_WorkingTrees", color='royalblue' , ax=ax)
sns.scatterplot(data=my_wtOut, x="refDBH_man", y="DBH_WorkingTrees", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(60, 120))
ax.set_xlabel('FI DBH')
ax.set_ylabel('WT DBH')
fig.show()

# ------------------------------------------------
#  get the regression parameters:
# ------------------------------------------------
m_wt, b_wt, r_value_wt, p_value, std_err = scipy.stats.linregress(my_wtIn['refDBH_man'], my_wtIn['DBH_WorkingTrees'])
stat_out = pg.corr(x=my_wtIn['refDBH_man'], y=my_wtIn['DBH_WorkingTrees'])
my_lm = pg.linear_regression(my_wtIn['refDBH_man'], my_wtIn['DBH_WorkingTrees'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_wtIn['refDBH_man'] - my_wtIn['DBH_WorkingTrees']
my_rmse_wt = ((my_err**2).mean()**0.5)
my_mae_wt = (my_err.abs().mean())



# ###############################
# analyse Greenlens measurements
# ###############################
# get the regression parameters:
m, b, r_value, p_value, std_err = scipy.stats.linregress(my_df['refDBH_man'], my_df['DBH_Greenlens'])
stat_out = pg.corr(x=my_df['refDBH_man'], y=my_df['DBH_Greenlens'])
my_lm = pg.linear_regression(my_df['refDBH_man'], my_df['DBH_Greenlens'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_df, x='refDBH_man', y='DBH_Greenlens', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_df, x="refDBH_man", y="DBH_Greenlens",  palette='royalblue', ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(60, 120))
ax.set_xlabel('FI DBH')
ax.set_ylabel('GL DBH')
fig.show()

# ------------------------------------------------
# find and exclude outliers
# ------------------------------------------------
# get the difference from the mean diameter:
myMultiDiff = pd.Series(my_lm.residuals_)
# get iqr:
threshold = 0.8
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
sns.regplot(data=my_glIn, x='refDBH_man', y='DBH_Greenlens', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_glIn, x="refDBH_man", y="DBH_Greenlens", color='royalblue' , ax=ax)
sns.scatterplot(data=my_glOut, x="refDBH_man", y="DBH_Greenlens", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(60, 120))
ax.set_xlabel('FI DBH')
ax.set_ylabel('GL DBH')
fig.show()

# ------------------------------------------------
#  get the regression parameters:
# ------------------------------------------------
m_gl, b_gl, r_value_gl, p_value, std_err = scipy.stats.linregress(my_glIn['refDBH_man'], my_glIn['DBH_Greenlens'])
stat_out = pg.corr(x=my_glIn['refDBH_man'], y=my_glIn['DBH_Greenlens'])
my_lm = pg.linear_regression(my_glIn['refDBH_man'], my_glIn['DBH_Greenlens'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err = my_glIn['refDBH_man'] - my_glIn['DBH_Greenlens']
my_rmse_gl = ((my_err**2).mean()**0.5)
my_mae_gl = (my_err.abs().mean())

# ------------------------------------------------
# make a joint figure -- Figure 11
# ------------------------------------------------
# make input lists:
myDF = [my_tqIn, my_glIn , my_wtIn]
myX = ['refDBH_man', 'refDBH_man', 'refDBH_man']
myY = ['Diameter_dbh0', 'DBH_Greenlens', 'DBH_WorkingTrees']
myR = [r_value_tq, r_value_gl, r_value_wt]
my_m = [m_tq, m_gl, m_wt]
my_b = [b_tq, b_gl, b_wt]
my_MAE = [my_mae_tq, my_mae_gl, my_mae_wt]
my_label = ['(a)', '(b)', '(c)']
my_xlabel = ['FI DBH [cm]', 'FI DBH [cm]', 'FI DBH [cm]']
my_ylabel = ['Tree-Quest DBH [cm]', 'GreenLens DBH [cm]', 'Working Trees DBH [cm]']
# plotting
fig, axx = plt.subplots(1, 3, figsize=(12, 12./3))
for my_df1, ax, x1, y1, r1, m1, b1, mae1, label1, xlabel1, ylabel1 in (zip(myDF, axx, myX, myY, myR, my_m, my_b, my_MAE, my_label, my_xlabel, my_ylabel)):
    ax.axline((0, 0), slope=1, color='k', linestyle=':')
    sns.scatterplot(data=my_df1, x=x1, y=y1, color='royalblue', ax=ax)
    sns.regplot(data=my_df1, x=x1, y=y1, color='k', scatter=False, ax=ax)
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
fig.savefig(figPath + r'\DBH_Stadtpark_otherApps.png', dpi=300)


# --------------------------------------------------------
# Check if normally distributed and do the stat test
# --------------------------------------------------------

# check if the data normally distributed:
my_tqIn['Diameter_dbh0'].hist(bins=20)
my_glIn['Diameter_dbh0'].hist(bins=20)
my_wtIn['Diameter_dbh0'].hist(bins=20)
#
scipy.stats.probplot(my_wtIn['Diameter_dbh0'], dist="norm", plot=plt)
#
scipy.stats.shapiro(my_glIn['Diameter_dbh0'])
# --> not normally distributed
scipy.stats.shapiro(my_wtIn['refDBH_man'])
# --> not normally distributed

# check if the data distributions are the same (a Wilcoxon signed-rank test for paired data)
# non-parametric alternative for paired t-test
stat1, p_value1 =scipy.stats.wilcoxon(my_tqIn['Diameter_dbh0'], my_tqIn['refDBH_man'], alternative='two-sided')
stat2, p_value2 =scipy.stats.wilcoxon(my_glIn['Diameter_dbh0'], my_glIn['refDBH_man'], alternative='two-sided')
stat3, p_value3 =scipy.stats.wilcoxon(my_wtIn['Diameter_dbh0'], my_wtIn['refDBH_man'], alternative='two-sided')
#
results1 = pg.wilcoxon(my_tqIn['Diameter_dbh0'], my_tqIn['refDBH_man'])
print(results1)
results2 = pg.wilcoxon(my_glIn['Diameter_dbh0'], my_glIn['refDBH_man'])
print(results2)
results3 = pg.wilcoxon(my_wtIn['Diameter_dbh0'], my_wtIn['refDBH_man'])
print(results3)

# if p-values are < 0.05 --> the measurements are statistically different --> we have to apply the correction functions
# if p-values are >= 0.05 --> not statistically different, Ho rejected

# examine the differences:
ax = pg.plot_blandaltman(my_tqIn['Diameter_dbh0'], my_tqIn['refDBH_man'])
plt.tight_layout()

# TQ - FI bias
bias1 = (my_tqIn['Diameter_dbh0'] - my_tqIn['refDBH_man']).median()
bias11 = (100*(my_tqIn['Diameter_dbh0'] - my_tqIn['refDBH_man'])/my_tqIn['refDBH_man']).median()

# TQ - TLS bias
bias2 = (my_glIn['Diameter_dbh0'] - my_glIn['refDBH_man']).median()
bias22 = (100*(my_glIn['Diameter_dbh0'] - my_glIn['refDBH_man'])/my_glIn['refDBH_man']).median()

# TLS - FI bias
bias3 = (my_wtIn['Diameter_dbh0'] - my_wtIn['refDBH_man']).median()
bias33 = (100*(my_wtIn['Diameter_dbh0'] - my_wtIn['refDBH_man'])/my_wtIn['refDBH_man']).median()

#----------------------------
# test all tree TQ gropus:
#----------------------------
minNum = min(len(my_tqIn['Diameter_dbh0']), len(my_glIn['Diameter_dbh0']), len(my_wtIn['Diameter_dbh0']))

#
mySatResTQ_groups = scipy.stats.friedmanchisquare(my_tqIn['Diameter_dbh0'].sample(n=minNum),
                                                  my_glIn['Diameter_dbh0'].sample(n=minNum),
                                                  my_wtIn['Diameter_dbh0'].sample(n=minNum)
                                                  )

mySatResTQ_groups.pvalue
mySatResTQ_groups.statistic