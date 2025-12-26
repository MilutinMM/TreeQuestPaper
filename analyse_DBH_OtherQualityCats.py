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
# read the high quality:
my_qH = pd.read_parquet(dataFolder + r'\qualHigh_outFiltered_dbh_LaxPark.parquet')
# read non high-quality (medium, low, and 'can tell'):
my_non_qH = pd.read_parquet(dataFolder + r'\not_qualHigh_dbh_LaxPark.parquet')

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
m, b, r_value, p_value, std_err = scipy.stats.linregress(my_Med['refDBH_man'], my_Med['Diameter_dbh0'])
stat_out = pg.corr(x=my_Med['refDBH_man'], y=my_Med['Diameter_dbh0'])
my_lm = pg.linear_regression(my_Med['refDBH_man'], my_Med['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_Med, x='refDBH_man', y='Diameter_dbh0', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_Med, x="refDBH_man", y="Diameter_dbh0", palette='royalblue' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(0.6, 1.4))
ax.set_xlabel('FI DBH')
ax.set_ylabel('Tree-Quest DBH')
fig.show()

# ------------------------------------------------
# find and exclude outliers (TQ-FI)
# ------------------------------------------------
# get the difference from the mean diameter:
myMultiDiff = pd.Series(my_lm.residuals_)
# get iqr:
threshold = 2.5
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
sns.regplot(data=my_MedIn, x='refDBH_man', y='Diameter_dbh0', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_MedIn, x="refDBH_man", y="Diameter_dbh0", color='royalblue' , ax=ax)
sns.scatterplot(data=myMedOut, x="refDBH_man", y="Diameter_dbh0", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(0.6, 1.4))
ax.set_xlabel('FI DBH')
ax.set_ylabel('Tree-Quest DBH')
fig.show()

# ------------------------------------------------
# convert to cm:
# ------------------------------------------------
my_MedIn['refDBH_man'] = my_MedIn['refDBH_man'].multiply(100)
my_MedIn['Diameter_dbh0'] = my_MedIn['Diameter_dbh0'].multiply(100)

# export:
my_MedIn.to_parquet(dataFolder + r'\qualMedium_outFiltered_dbh_LaxPark.parquet')

# ------------------------------------------------
#  TQ vs. FI: get the regression parameters:
# ------------------------------------------------
m_med, b_med, r_value_med, p_value, std_err = scipy.stats.linregress(my_MedIn['refDBH_man'], my_MedIn['Diameter_dbh0'])
stat_out = pg.corr(x=my_MedIn['refDBH_man'], y=my_MedIn['Diameter_dbh0'])
my_lm = pg.linear_regression(my_MedIn['refDBH_man'], my_MedIn['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_MedIn['refDBH_man'] - my_MedIn['Diameter_dbh0']
my_rmse_med = ((my_err**2).mean()**0.5)
my_mae_med = (my_err.abs().mean())

# rMAE
rMAE_med = my_MedIn['errDBH_man_Rel'].mean()

# ##########################
# analyse LOW quality
# ##########################

# get the regression parameters:
m, b, r_value, p_value, std_err = scipy.stats.linregress(my_Low['refDBH_man'], my_Low['Diameter_dbh0'])
stat_out = pg.corr(x=my_Low['refDBH_man'], y=my_Low['Diameter_dbh0'])
my_lm = pg.linear_regression(my_Low['refDBH_man'], my_Low['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_Low, x='refDBH_man', y='Diameter_dbh0', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_Low, x="refDBH_man", y="Diameter_dbh0", palette='royalblue' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(0.6, 1.4))
ax.set_xlabel('FI DBH')
ax.set_ylabel('Tree-Quest DBH')
fig.show()

# ------------------------------------------------
# find and exclude outliers (TQ-FI)
# ------------------------------------------------
# get the difference from the mean diameter:
myMultiDiff = pd.Series(my_lm.residuals_)
# get iqr:
threshold = 5
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
sns.regplot(data=my_LowIn, x='refDBH_man', y='Diameter_dbh0', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_LowIn, x="refDBH_man", y="Diameter_dbh0", color='royalblue' , ax=ax)
sns.scatterplot(data=myLowOut, x="refDBH_man", y="Diameter_dbh0", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(0.6, 1.4))
ax.set_xlabel('FI DBH')
ax.set_ylabel('Tree-Quest DBH')
fig.show()

# ------------------------------------------------
# convert to cm:
# ------------------------------------------------
my_LowIn['refDBH_man'] = my_LowIn['refDBH_man'].multiply(100)
my_LowIn['Diameter_dbh0'] = my_LowIn['Diameter_dbh0'].multiply(100)

# export:
my_LowIn.to_parquet(dataFolder + r'\qualLow_outFiltered_dbh_LaxPark.parquet')

# ------------------------------------------------
#  TQ vs. FI: get the regression parameters:
# ------------------------------------------------
m_low, b_low, r_value_low, p_value, std_err = scipy.stats.linregress(my_LowIn['refDBH_man'], my_LowIn['Diameter_dbh0'])
stat_out = pg.corr(x=my_LowIn['refDBH_man'], y=my_LowIn['Diameter_dbh0'])
my_lm = pg.linear_regression(my_LowIn['refDBH_man'], my_LowIn['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_LowIn['refDBH_man'] - my_LowIn['Diameter_dbh0']
my_rmse_low = ((my_err**2).mean()**0.5)
my_mae_low = (my_err.abs().mean())

# ###############################
# combine MEDIUM and LOW quality
# ###############################
my_ML = pd.concat([my_MedIn, my_LowIn], axis='index')

#get the regression parameters:
m_ml, b_ml, r_value_ml, p_value, std_err = scipy.stats.linregress(my_ML['refDBH_man'], my_ML['Diameter_dbh0'])
stat_out = pg.corr(x=my_ML['refDBH_man'], y=my_ML['Diameter_dbh0'])
my_lm = pg.linear_regression(my_ML['refDBH_man'], my_ML['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_ML['refDBH_man'] - my_ML['Diameter_dbh0']
my_rmse_ml = ((my_err**2).mean()**0.5)
my_mae_ml = (my_err.abs().mean())

# ###############################
# combine HIGH and MEDIUM
# ###############################
my_HM = pd.concat([my_qH, my_MedIn], axis='index')

#get the regression parameters:
m_hm, b_hm, r_value_hm, p_value, std_err = scipy.stats.linregress(my_HM['refDBH_man'], my_HM['Diameter_dbh0'])
stat_out = pg.corr(x=my_HM['refDBH_man'], y=my_HM['Diameter_dbh0'])
my_lm = pg.linear_regression(my_HM['refDBH_man'], my_HM['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_HM['refDBH_man'] - my_HM['Diameter_dbh0']
my_rmse_hm = ((my_err**2).mean()**0.5)
my_mae_hm = (my_err.abs().mean())

my_HM['errDBH_man_Rel'].mean()

# ###############################
# combine HIGH, MEDIUM, and LOW
# ###############################
my_HML = pd.concat([my_HM, my_LowIn], axis='index')

#get the regression parameters:
m_hml, b_hml, r_value_hml, p_value, std_err = scipy.stats.linregress(my_HML['refDBH_man'], my_HML['Diameter_dbh0'])
stat_out = pg.corr(x=my_HML['refDBH_man'], y=my_HML['Diameter_dbh0'])
my_lm = pg.linear_regression(my_HML['refDBH_man'], my_HML['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_HML['refDBH_man'] - my_HML['Diameter_dbh0']
my_rmse_hml = ((my_err**2).mean()**0.5)
my_mae_hml = (my_err.abs().mean())

my_HML['errDBH_man_Rel'].mean()
# ------------------------------------------------
# make a joint figure --- Figure 6 ----
# ------------------------------------------------
# make input lists:
myDF = [my_MedIn, my_HM , my_HML]
myX = ['refDBH_man', 'refDBH_man', 'refDBH_man']
myY = ['Diameter_dbh0', 'Diameter_dbh0', 'Diameter_dbh0']
myR = [r_value_med, r_value_hm, r_value_hml]
my_m = [m_med, m_hm, m_hml]
my_b = [b_med, b_hm, b_hml]
my_MAE = [my_mae_med, my_mae_hm, my_mae_hml]
my_label = ['(a)', '(b)', '(c)']
my_xlabel = ['FI DBH [cm]', 'FI DBH [cm]', 'FI DBH [cm]']
my_ylabel = ['Medium DBH [cm]', 'Medium & High DBH [cm]', 'Medium, High, and Low DBH [cm]']
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
fig.savefig(figPath + r'\DBH_LaxPark_otherQualities.png', dpi=300)

# --------------------------------------------------------
# Check if normally distributed and do the stat test
# --------------------------------------------------------

# check if the data normally distributed:
my_MedIn['Diameter_dbh0'].hist(bins=20)
my_HM['Diameter_dbh0'].hist(bins=20)
my_HML['Diameter_dbh0'].hist(bins=20)
#
scipy.stats.probplot(my_HML['Diameter_dbh0'], dist="norm", plot=plt)
#
scipy.stats.shapiro(my_HM['Diameter_dbh0'])
# --> not normally distributed
scipy.stats.shapiro(my_HML['refDBH_man'])
# --> not normally distributed

# check if the data distributions are the same (a Wilcoxon signed-rank test for paired data)
# non-parametric alternative for paired t-test
stat1, p_value1 =scipy.stats.wilcoxon(my_MedIn['Diameter_dbh0'], my_MedIn['refDBH_man'], alternative='two-sided')
stat2, p_value2 =scipy.stats.wilcoxon(my_HM['Diameter_dbh0'], my_HM['refDBH_man'], alternative='two-sided')
stat3, p_value3 =scipy.stats.wilcoxon(my_HML['Diameter_dbh0'], my_HML['refDBH_man'], alternative='two-sided')
#
results1 = pg.wilcoxon(my_MedIn['Diameter_dbh0'], my_MedIn['refDBH_man'])
print(results1)
results2 = pg.wilcoxon(my_HM['Diameter_dbh0'], my_HM['refDBH_man'])
print(results2)
results3 = pg.wilcoxon(my_HML['Diameter_dbh0'], my_HML['refDBH_man'])
print(results3)

# if p-values are < 0.05 --> the measurements are statistically different --> we have to apply the correction functions
# if p-values are >= 0.05 --> not statistically different, Ho rejected

# examine the differences:
ax = pg.plot_blandaltman(my_MedIn['Diameter_dbh0'], my_MedIn['refDBH_man'])
plt.tight_layout()

# TQ - FI bias
bias1 = (my_MedIn['Diameter_dbh0'] - my_MedIn['refDBH_man']).median()
bias11 = (100*(my_MedIn['Diameter_dbh0'] - my_MedIn['refDBH_man'])/my_MedIn['refDBH_man']).median()

# TQ - TLS bias
bias2 = (my_HM['Diameter_dbh0'] - my_HM['refDBH_man']).median()
bias22 = (100*(my_HM['Diameter_dbh0'] - my_HM['refDBH_man'])/my_HM['refDBH_man']).median()

# TLS - FI bias
bias3 = (my_HML['Diameter_dbh0'] - my_HML['refDBH_man']).median()
bias33 = (100*(my_HML['Diameter_dbh0'] - my_HML['refDBH_man'])/my_HML['refDBH_man']).median()

#----------------------------
# test all tree TQ gropus:
#----------------------------
minNum = min(len(my_qH['Diameter_dbh0']), len(my_MedIn['Diameter_dbh0']), len(my_LowIn['Diameter_dbh0']))

#
mySatResTQ_groups = scipy.stats.friedmanchisquare(my_qH['Diameter_dbh0'].sample(n=minNum),
                                                  my_MedIn['Diameter_dbh0'].sample(n=minNum),
                                                  my_LowIn['Diameter_dbh0'].sample(n=minNum)
                                                  )

mySatResTQ_groups.pvalue
mySatResTQ_groups.statistic