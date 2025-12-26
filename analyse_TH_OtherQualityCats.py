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
# read
# ---------------------------
# path to the data folder
dataFolder = r'c:\Path\to\folder\with\data\downloaded\from\Zenodo'
# read the high quality:
my_qH = pd.read_parquet(dataFolder + r'\qualHigh_outFiltered_TH_LaxPark.parquet')
# read non high-quality:
my_non_qH = pd.read_parquet(dataFolder + r'\not_qualHigh_TH_LaxPark.parquet')

# ---------------------------------------------------
# split non high-quality into individual categories
# ----------------------------------------------------
# select medium-quality
my_Med = my_non_qH[my_non_qH['qMin'] == 'Med']
# select low-quality
my_Low = my_non_qH[my_non_qH['qMin'] == 'Low']
# select cant tell:
my_other = my_non_qH[my_non_qH['qMin'] == 'Cant Tell']
# select those with none score:
my_None = my_non_qH[my_non_qH['qMin'].isna()]

# check the total number:
myNum_not_qH = my_Med.shape[0] + my_Low.shape[0] + my_other.shape[0] + my_None.shape[0]

# ##########################
# analyse MEDIUM quality
# ##########################

# get the regression parameters:
m, b, r_value, p_value, std_err = scipy.stats.linregress(my_Med['refTH_man'], my_Med['TreeHeight'])
stat_out = pg.corr(x=my_Med['refTH_man'], y=my_Med['TreeHeight'])
my_lm = pg.linear_regression(my_Med['refTH_man'], my_Med['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_Med, x='refTH_man', y='TreeHeight', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_Med, x="refTH_man", y="TreeHeight", palette='royalblue' , ax=ax)
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
threshold = 1.5
q3 = myMultiDiff.quantile(0.75)
q1 = myMultiDiff.quantile(0.25)
iqr =  q3 - q1
lower_bound = q1 - threshold * iqr
upper_bound = q3 + threshold * iqr
# remove according to the iqr:
ind_goodPts = (myMultiDiff >= lower_bound) & (myMultiDiff <= upper_bound)
ind_outPts = (myMultiDiff < lower_bound) | (myMultiDiff > upper_bound)
#
my_MedIn = my_Med[ind_goodPts.to_list()]
myMedOut = my_Med[ind_outPts.to_list()]

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_MedIn, x='refTH_man', y='TreeHeight', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_MedIn, x="refTH_man", y="TreeHeight", color='royalblue' , ax=ax)
sns.scatterplot(data=myMedOut, x="refTH_man", y="TreeHeight", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(8, 30))
ax.set_xlabel('FI TH')
ax.set_ylabel('Tree-Quest TH')
fig.show()

# ------------------------------------------------
#  TQ vs. FI: get the regression parameters:
# ------------------------------------------------
m_med, b_med, r_value_med, p_value, std_err = scipy.stats.linregress(my_MedIn['refTH_man'], my_MedIn['TreeHeight'])
stat_out = pg.corr(x=my_MedIn['refTH_man'], y=my_MedIn['TreeHeight'])
my_lm = pg.linear_regression(my_MedIn['refTH_man'], my_MedIn['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [m]:
my_err =  my_MedIn['refTH_man'] - my_MedIn['TreeHeight']
my_rmse_med = ((my_err**2).mean()**0.5)
my_mae_med = (my_err.abs().mean())

my_MedIn['errTH_man_Rel'].mean()

# ##########################
# analyse LOW quality
# ##########################

# get the regression parameters:
m, b, r_value, p_value, std_err = scipy.stats.linregress(my_Low['refTH_man'], my_Low['TreeHeight'])
stat_out = pg.corr(x=my_Low['refTH_man'], y=my_Low['TreeHeight'])
my_lm = pg.linear_regression(my_Low['refTH_man'], my_Low['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_Low, x='refTH_man', y='TreeHeight', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_Low, x="refTH_man", y="TreeHeight", palette='royalblue' , ax=ax)
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
threshold = 0.4
q3 = myMultiDiff.quantile(0.75)
q1 = myMultiDiff.quantile(0.25)
iqr =  q3 - q1
lower_bound = q1 - threshold * iqr
upper_bound = q3 + threshold * iqr
# remove according to the iqr:
ind_goodPts = (myMultiDiff >= lower_bound) & (myMultiDiff <= upper_bound)
ind_outPts = (myMultiDiff < lower_bound) | (myMultiDiff > upper_bound)
#
my_LowIn = my_Low[ind_goodPts.to_list()]
myLowOut = my_Low[ind_outPts.to_list()]

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_LowIn, x='refTH_man', y='TreeHeight', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_LowIn, x="refTH_man", y="TreeHeight", color='royalblue' , ax=ax)
sns.scatterplot(data=myLowOut, x="refTH_man", y="TreeHeight", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(0.6, 1.4))
ax.set_xlabel('FI TH')
ax.set_ylabel('Tree-Quest TH')
fig.show()

# ------------------------------------------------
#  TQ vs. FI: get the regression parameters:
# ------------------------------------------------
m_low, b_low, r_value_low, p_value, std_err = scipy.stats.linregress(my_LowIn['refTH_man'], my_LowIn['TreeHeight'])
stat_out = pg.corr(x=my_LowIn['refTH_man'], y=my_LowIn['TreeHeight'])
my_lm = pg.linear_regression(my_LowIn['refTH_man'], my_LowIn['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [m]:
my_err =  my_LowIn['refTH_man'] - my_LowIn['TreeHeight']
my_rmse_low = ((my_err**2).mean()**0.5)
my_mae_low = (my_err.abs().mean())

# ###############################
# combine HIGH and MEDIUM
# ###############################
my_HM = pd.concat([my_qH, my_MedIn], axis='index')

#get the regression parameters:
m_hm, b_hm, r_value_hm, p_value, std_err = scipy.stats.linregress(my_HM['refTH_man'], my_HM['TreeHeight'])
stat_out = pg.corr(x=my_HM['refTH_man'], y=my_HM['TreeHeight'])
my_lm = pg.linear_regression(my_HM['refTH_man'], my_HM['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [m]:
my_err =  my_HM['refTH_man'] - my_HM['TreeHeight']
my_rmse_hm = ((my_err**2).mean()**0.5)
my_mae_hm = (my_err.abs().mean())

# ###############################
# combine HIGH, MEDIUM, and LOW
# ###############################
my_HML = pd.concat([my_HM, my_LowIn], axis='index')
#export:
my_HML.to_parquet(dataFolder + r'\qHML_TH_LaxPark_UserGroups.parquet')

#get the regression parameters:
m_hml, b_hml, r_value_hml, p_value, std_err = scipy.stats.linregress(my_HML['refTH_man'], my_HML['TreeHeight'])
stat_out = pg.corr(x=my_HML['refTH_man'], y=my_HML['TreeHeight'])
my_lm = pg.linear_regression(my_HML['refTH_man'], my_HML['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [m]:
my_err =  my_HML['refTH_man'] - my_HML['TreeHeight']
my_rmse_hml = ((my_err**2).mean()**0.5)
my_mae_hml = (my_err.abs().mean())

my_HML['errTH_man_Rel'].mean()

# ------------------------------------------------
# make a joint figure --- Figure 7
# ------------------------------------------------
# make input lists:
myDF = [my_MedIn, my_HM , my_HML]
myX = ['refTH_man', 'refTH_man', 'refTH_man']
myY = ['TreeHeight', 'TreeHeight', 'TreeHeight']
myR = [r_value_med, r_value_hm, r_value_hml]
my_m = [m_med, m_hm, m_hml]
my_b = [b_med, b_hm, b_hml]
my_MAE = [my_mae_med, my_mae_hm, my_mae_hml]
my_label = ['(a)', '(b)', '(c)']
my_xlabel = ['FI TH [m]', 'FI TH [m]', 'FI TH [m]']
my_ylabel = ['Medium TH [m]', 'Medium & High TH [m]', 'Medium, High, and Low TH [m]']
# plotting
fig, axx = plt.subplots(1, 3, figsize=(12, 12./3))
for my_df1, ax, x1, y1, r1, m1, b1, mae1, label1, xlabel1, ylabel1 in (zip(myDF, axx, myX, myY, myR, my_m, my_b, my_MAE, my_label, my_xlabel, my_ylabel)):
    ax.axline((0, 0), slope=1, color='k', linestyle=':')
    sns.scatterplot(data=my_df1, x=x1, y=y1, color='royalblue', ax=ax)
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

fig.tight_layout()
fig.show()

figPath = r'c:\path\to\folder\where\figures\will\be\stored'
fig.savefig(figPath + r'\TH_LaxPark_otherQualities.png', dpi=300)


# --------------------------------------------------------
# Check if normally distributed and do the stat test
# --------------------------------------------------------

# check if the data normally distributed:
my_MedIn['TreeHeight'].hist(bins=20)
my_HM['TreeHeight'].hist(bins=20)
my_HML['TreeHeight'].hist(bins=20)
#
scipy.stats.probplot(my_HML['TreeHeight'], dist="norm", plot=plt)
#
scipy.stats.shapiro(my_HM['TreeHeight'])
# --> not normally distributed
scipy.stats.shapiro(my_HML['TreeHeight'])
# --> not normally distributed

# check if the data distributions are the same (a Wilcoxon signed-rank test for paired data)
# non-parametric alternative for paired t-test
stat1, p_value1 =scipy.stats.wilcoxon(my_MedIn['TreeHeight'], my_MedIn['refTH_man'], alternative='two-sided')
stat2, p_value2 =scipy.stats.wilcoxon(my_HM['TreeHeight'], my_HM['refTH_man'], alternative='two-sided')
stat3, p_value3 =scipy.stats.wilcoxon(my_HML['TreeHeight'], my_HML['refTH_man'], alternative='two-sided')
#
results1 = pg.wilcoxon(my_MedIn['TreeHeight'], my_MedIn['refTH_man'])
print(results1)
results2 = pg.wilcoxon(my_HM['TreeHeight'], my_HM['refTH_man'])
print(results2)
results3 = pg.wilcoxon(my_HML['TreeHeight'], my_HML['refTH_man'])
print(results3)

# if p-values are < 0.05 --> the measurements are statistically different --> we have to apply the correction functions
# if p-values are >= 0.05 --> not statistically different, Ho rejected

# examine the differences:
ax = pg.plot_blandaltman(my_MedIn['TreeHeight'], my_MedIn['refTH_man'])
plt.tight_layout()

# TQ - FI bias
bias1 = (my_MedIn['TreeHeight'] - my_MedIn['refTH_man']).median()
bias11 = (100*(my_MedIn['TreeHeight'] - my_MedIn['refTH_man'])/my_MedIn['refTH_man']).median()

# TQ - TLS bias
bias2 = (my_HM['TreeHeight'] - my_HM['refTH_man']).median()
bias22 = (100*(my_HM['TreeHeight'] - my_HM['refTH_man'])/my_HM['refTH_man']).median()

# TLS - FI bias
bias3 = (my_HML['TreeHeight'] - my_HML['refTH_man']).median()
bias33 = (100*(my_HML['TreeHeight'] - my_HML['refTH_man'])/my_HML['refTH_man']).median()

#----------------------------
# test all tree TQ gropus:
#----------------------------
minNum = min(len(my_qH['TreeHeight']), len(my_MedIn['TreeHeight']), len(my_LowIn['TreeHeight']))

#
mySatResTQ_groups = scipy.stats.friedmanchisquare(my_qH['Diameter_dbh0'].sample(n=minNum),
                                                  my_MedIn['Diameter_dbh0'].sample(n=minNum),
                                                  my_LowIn['Diameter_dbh0'].sample(n=minNum)
                                                  )

mySatResTQ_groups.pvalue
mySatResTQ_groups.statistic