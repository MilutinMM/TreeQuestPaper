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
# read ARD geopandas file with Geo-Quest data
gdf = gpd.read_parquet(dataFolder + r'\ard_gdf_2024_03_07_LaxPark.parquet')
# read the quality scores:
df_qDBH = pd.read_parquet(dataFolder + r'\quality_dbh_LaxPark.parquet')

# ------------------------------------------------
# assign the quality values from the quality file
# ------------------------------------------------
gdf_qDBH = pd.merge(gdf, df_qDBH, on=['QuestSurveySubmissionId'], how='left', validate="one_to_many")

# ----------------------------------------
# only High quality observations:
# ----------------------------------------
gdf_qDBH_qH = gdf_qDBH[gdf_qDBH['qMin'] == 'High']
# export all others
gdf_qDBH_not_qH = gdf_qDBH[gdf_qDBH['qMin'] != 'High']
gdf_qDBH_not_qH.to_parquet(dataFolder + r'\not_qualHigh_dbh_LaxPark.parquet')
#
# get the regression parameters:
m, b, r_value, p_value, std_err = scipy.stats.linregress(gdf_qDBH_qH['refDBH_man'], gdf_qDBH_qH['Diameter_dbh0'])
stat_out = pg.corr(x=gdf_qDBH_qH['refDBH_man'], y=gdf_qDBH_qH['Diameter_dbh0'])
my_lm = pg.linear_regression(gdf_qDBH_qH['refDBH_man'], gdf_qDBH_qH['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=gdf_qDBH_qH, x='refDBH_man', y='Diameter_dbh0', color='k',  scatter=False , ax=ax)
myColors=['limegreen', 'royalblue', 'red']
sns.scatterplot(data=gdf_qDBH_qH, x="refDBH_man", y="Diameter_dbh0", palette=myColors , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(0.6, 1.4))
ax.set_xlabel('FI DBH')
ax.set_ylabel('Tree-Quest DBH')
fig.show()

# interactive plot:
# hvplot.show(gdf_qDBH_qH.hvplot(x='refDBH_man', y='Diameter_dbh0', kind='scatter', hover_cols=['refDBH_man', 'Diameter_dbh0', 'QuestSurveySubmissionId', 'treeId']))

# ------------------------------------------------
# correct the submission with wrong TreeID
# ------------------------------------------------
# select the submission according to dbh values:
mySub = gdf_qDBH_qH[(gdf_qDBH_qH['refDBH_man'].round(3) == 0.119) & (gdf_qDBH_qH['Diameter_dbh0'] >= 0.527)]
mySubID = mySub['QuestSurveySubmissionId'].values[0]
# export it:
#mySub.to_parquet(dataFolder + r'\missAssigned_dbh_LaxPark.parquet')
# remove the submission with the miss-assigned tree ID:
gdf_qDBH_qH = gdf_qDBH_qH[gdf_qDBH_qH['QuestSurveySubmissionId'] != mySubID]

# ------------------------------------------------
# find and exclude outliers
# (submissions with multiple dbh measurements and miss-assigned quality score)
# ------------------------------------------------
# inspect some outlier:
myOut1 = gdf_qDBH_qH[(gdf_qDBH_qH['refDBH_man'].round(3) == 0.476) & (gdf_qDBH_qH['Diameter_dbh0'].round(3) <=0.214)]

# split to multiple and single dbh measurements:
myMulti = gdf_qDBH_qH[gdf_qDBH_qH['Diameter_dbh0'] != gdf_qDBH_qH['DiameterMean']]
mySingle = gdf_qDBH_qH[gdf_qDBH_qH['Diameter_dbh0'] == gdf_qDBH_qH['DiameterMean']]
# get the difference from the mean diameter:
myMultiDiff = myMulti['Diameter_dbh0'] - myMulti['DiameterMean']
# get iqr:
threshold = 0.95
q3 = myMultiDiff.quantile(0.75)
q1 = myMultiDiff.quantile(0.25)
iqr =  q3 - q1
lower_bound = q1 - threshold * iqr
upper_bound = q3 + threshold * iqr
# remove according to the iqr:
myOutRemoved = myMulti[(myMultiDiff >= lower_bound) & (myMultiDiff <= upper_bound)]
myOut = myMulti[(myMultiDiff < lower_bound) | (myMultiDiff > upper_bound)]

# plot
fig, ax = plt.subplots()
sns.regplot(data=myOutRemoved, x='refDBH_man', y='Diameter_dbh0', color='k',  scatter=False , ax=ax)
myColors=['limegreen', 'royalblue', 'red']
sns.scatterplot(data=myOutRemoved, x="refDBH_man", y="Diameter_dbh0", color='royalblue' , ax=ax)
sns.scatterplot(data=myOut, x="refDBH_man", y="Diameter_dbh0", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(0.6, 1.4))
ax.set_xlabel('FI DBH')
ax.set_ylabel('Tree-Quest DBH')
fig.show()

# export outliers
#myOut.to_parquet(dataFolder + r'\outliers_in_qualHigh_dbh_LaxPark.parquet')
# ------------------------------------------------
# join with single dbh measurements:
# ------------------------------------------------
my_qH = pd.concat([myOutRemoved, mySingle], axis='index')

# convert to cm:
my_qH['refDBH_tls'] = my_qH['refDBH_tls'].multiply(100)
my_qH['refDBH_man'] = my_qH['refDBH_man'].multiply(100)
my_qH['Diameter_dbh0'] = my_qH['Diameter_dbh0'].multiply(100)

# ------------------------------------------------
#  TQ vs. FI: get the regression parameters:
# ------------------------------------------------
m_tq_fi, b_tq_fi, r_value_tq_fi, p_value, std_err = scipy.stats.linregress(my_qH['refDBH_man'], my_qH['Diameter_dbh0'])
stat_out = pg.corr(x=my_qH['refDBH_man'], y=my_qH['Diameter_dbh0'])
my_lm = pg.linear_regression(my_qH['refDBH_man'], my_qH['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_qH['refDBH_man'] - my_qH['Diameter_dbh0']
my_rmse_tq_fi = ((my_err**2).mean()**0.5)
my_mae_tq_fi = (my_err.abs().mean())

# get rMAE
rMAE_tq_fi = my_qH['errDBH_man_Rel'].mean()
rMAE_tq_tls = my_qH['errDBH_tls_Rel'].mean()

# ------------------------------------------------
#  TQ vs. TLS: get the regression parameters:
# ------------------------------------------------
#
m_tq_tls, b_tq_tls, r_value_tq_tls, p_value, std_err = scipy.stats.linregress(my_qH['refDBH_tls'], my_qH['Diameter_dbh0'])
stat_out = pg.corr(x=my_qH['refDBH_tls'], y=my_qH['Diameter_dbh0'])
my_lm = pg.linear_regression(my_qH['refDBH_tls'], my_qH['Diameter_dbh0'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_qH['refDBH_tls'] - my_qH['Diameter_dbh0']
my_rmse_tq_tls = ((my_err**2).mean()**0.5)
my_mae_tq_tls = (my_err.abs().mean())

# ------------------------------------------------
#  TLS vs. FI: get the regression parameters:
# ------------------------------------------------
m_tls_fi, b_tls_fi, r_value_tls_fi, p_value, std_err = scipy.stats.linregress(my_qH['refDBH_man'], my_qH['refDBH_tls'])
stat_out = pg.corr(x=my_qH['refDBH_man'], y=my_qH['refDBH_tls'])
my_lm = pg.linear_regression(my_qH['refDBH_man'], my_qH['refDBH_tls'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_qH['refDBH_man'] - my_qH['refDBH_tls']
my_rmse_tls_fi = ((my_err**2).mean()**0.5)
my_mae_tls_fi = (my_err.abs().mean())

# get rMAE
rMAE_tls_fi = my_qH['errDBH_tlsMan_Rel'].mean()

# ------------------------------------------------
# make a joint figure -- Figure 4
# ------------------------------------------------
# make input lists:
myX = ['refDBH_man', 'refDBH_tls', 'refDBH_man']
myY = ['Diameter_dbh0', 'Diameter_dbh0', 'refDBH_tls']
myR = [r_value_tq_fi, r_value_tq_tls, r_value_tls_fi]
my_m = [m_tq_fi, m_tq_tls, m_tls_fi]
my_b = [b_tq_fi, b_tq_tls, b_tls_fi]
my_MAE = [my_mae_tq_fi, my_mae_tq_tls, my_mae_tls_fi]
#my_ptNum = [my_mae_tq_fi, my_mae_tq_tls, my_mae_tls_fi]
my_label = ['(a)', '(b)', '(c)']
my_xlabel = ['FI DBH [cm]', 'TLS DBH [cm]', 'FI DBH [cm]']
my_ylabel = ['Tree-Quest DBH [cm]', 'Tree-Quest DBH [cm]', 'TLS DBH [cm]']
# plotting
fig, axx = plt.subplots(1, 3, figsize=(12, 12./3))
for ax, x1, y1, r1, m1, b1, mae1, label1, xlabel1, ylabel1 in (zip(axx, myX, myY, myR, my_m, my_b, my_MAE, my_label, my_xlabel, my_ylabel)):
    ax.axline((0, 0), slope=1, color='k', linestyle=':', label='1-1 line')
    sns.scatterplot(data=my_qH, x=x1, y=y1, color='royalblue', ax=ax)
    sns.regplot(data=my_qH, x=x1, y=y1, color='k', scatter=False, ax=ax)
    ax.annotate('$R^2$: ' + str("{:.2f}".format(r1 ** 2)), xy=(10, 140))
    ax.annotate('$y=$' + str("{:.2f}".format(m1)) + '$x$ + ' + str("{:.2f}".format(b1)), xy=(10, 130))
    ax.annotate('MAE: ' + str("{:.1f}".format(mae1)) + ' cm', xy=(10, 120))
    ax.annotate('$n = $' + str("{}".format(len(my_qH[y1]))), xy=(10, 110), size=15)
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
fig.savefig(figPath + r'\DBH_LaxPark_highQuality.png', dpi=300)

# --------------------------------------------------------
# export the high-quality and outlier filtered measurements
# --------------------------------------------------------
my_qH.to_parquet(dataFolder + r'\qualHigh_outFiltered_dbh_LaxPark.parquet')

# --------------------------------------------------------
# Check if normally distributed and do the stat test
# --------------------------------------------------------

# check if the data normally distributed:
my_qH['Diameter_dbh0'].hist(bins=20)
my_qH['refDBH_man'].hist(bins=20)
#
scipy.stats.probplot(my_qH['Diameter_dbh0'], dist="norm", plot=plt)

#
scipy.stats.shapiro(my_qH['Diameter_dbh0'])
# --> not normally distributed
scipy.stats.shapiro(my_qH['refDBH_man'])
# --> not normally distributed

# check if the data distributions are the same (a Wilcoxon signed-rank test for paired data)
# non-parametric alternative for paired t-test
stat1, p_value1 =scipy.stats.wilcoxon(my_qH['Diameter_dbh0'], my_qH['refDBH_man'], alternative='two-sided')
stat2, p_value2 =scipy.stats.wilcoxon(my_qH['refDBH_tls'], my_qH['refDBH_man'], alternative='two-sided')
stat3, p_value3 =scipy.stats.wilcoxon(my_qH['Diameter_dbh0'], my_qH['refDBH_tls'], alternative='two-sided')
#
results1 = pg.wilcoxon(my_qH['Diameter_dbh0'], my_qH['refDBH_man'])
print(results1)
results2 = pg.wilcoxon(my_qH['refDBH_tls'], my_qH['refDBH_man'])
print(results2)
results3 = pg.wilcoxon(my_qH['Diameter_dbh0'], my_qH['refDBH_tls'])
print(results3)

# both p-values are < 0.05 --> the measurements are different --> we have to apply the correction functions

# examine the differences:
ax = pg.plot_blandaltman(my_qH['Diameter_dbh0'], my_qH['refDBH_man'])
plt.tight_layout()

# TQ - FI bias
bias1 = (my_qH['Diameter_dbh0'] - my_qH['refDBH_man']).median()
bias11 = (100*(my_qH['Diameter_dbh0'] - my_qH['refDBH_man'])/my_qH['refDBH_man']).median()

# TQ - TLS bias
bias2 = (my_qH['Diameter_dbh0'] - my_qH['refDBH_tls']).median()
bias22 = (100*(my_qH['Diameter_dbh0'] - my_qH['refDBH_tls'])/my_qH['refDBH_tls']).median()

# TLS - FI bias
bias3 = (my_qH['refDBH_tls'] - my_qH['refDBH_man']).median()
bias33 = (100*(my_qH['refDBH_tls'] - my_qH['refDBH_man'])/my_qH['refDBH_man']).median()