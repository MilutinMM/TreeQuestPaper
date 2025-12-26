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
df_qTH = pd.read_parquet(dataFolder + r'\quality_th_LaxPark.parquet')

# ------------------------------------------------
# assign the quality values from the quality file
# ------------------------------------------------
gdf_qTH = pd.merge(gdf, df_qTH, on=['QuestSurveySubmissionId'], how='left', validate="one_to_many")

# ----------------------------------------
# only High quality observations:
# ----------------------------------------
# select the high-quality:
gdf_qTH_qH = gdf_qTH[gdf_qTH['qMin'] == 'High']
# select the other qualities:
gdf_not_qH = gdf_qTH[gdf_qTH['qMin'] != 'High']
gdf_not_qH.to_parquet(dataFolder + r'\not_qualHigh_TH_LaxPark.parquet')

# get the regression parameters:
m, b, r_value, p_value, std_err = scipy.stats.linregress(gdf_qTH_qH['refTH_man'], gdf_qTH_qH['TreeHeight'])
stat_out = pg.corr(x=gdf_qTH_qH['refTH_man'], y=gdf_qTH_qH['TreeHeight'])
my_lm = pg.linear_regression(gdf_qTH_qH['refTH_man'], gdf_qTH_qH['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# plot
fig, ax = plt.subplots()
sns.regplot(data=gdf_qTH_qH, x='refTH_man', y='TreeHeight', color='k',  scatter=False , ax=ax)
myColors=['limegreen', 'royalblue', 'red']
sns.scatterplot(data=gdf_qTH_qH, x="refTH_man", y="TreeHeight", palette=myColors , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(8, 30))
ax.set_xlabel('FI Height')
ax.set_ylabel('Tree-Quest Height')
fig.show()


# inspect some outlier:
myOut1 = gdf_qTH_qH[(gdf_qTH_qH['refTH_man'].round(1) >= 22.0) & (gdf_qTH_qH['TreeHeight'].round(1) >=29)]
myOut1.QuestSurveySubmissionId

# ------------------------------------------------
# find and exclude outliers (TQ-FI)
# ------------------------------------------------
# get the difference from the mean diameter:
myMultiDiff = pd.Series(my_lm.residuals_)
# get iqr:
threshold = 0.95
q3 = myMultiDiff.quantile(0.75)
q1 = myMultiDiff.quantile(0.25)
iqr =  q3 - q1
lower_bound = q1 - threshold * iqr
upper_bound = q3 + threshold * iqr
# remove according to the iqr:
ind_goodPts = (myMultiDiff >= lower_bound) & (myMultiDiff <= upper_bound)
ind_outPts = (myMultiDiff < lower_bound) | (myMultiDiff > upper_bound)
#
my_qH = gdf_qTH_qH[ind_goodPts.to_list()]
myOut = gdf_qTH_qH[ind_outPts.to_list()]

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_qH, x='refTH_man', y='TreeHeight', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_qH, x="refTH_man", y="TreeHeight", color='royalblue' , ax=ax)
sns.scatterplot(data=myOut, x="refTH_man", y="TreeHeight", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(8, 30))
ax.set_xlabel('FI Height')
ax.set_ylabel('Tree-Quest Height')
fig.show()

# export the outliers:
#myOut.to_parquet(dataFolder + r'\outliers_in_high_TH_LaxPark_part1.parquet')

# ------------------------------------------------
# find and exclude outliers (TQ-TLS)
# ------------------------------------------------
# get the regression parameters:
m, b, r_value, p_value, std_err = scipy.stats.linregress(my_qH['refTH_tls'], my_qH['TreeHeight'])
stat_out = pg.corr(x=my_qH['refTH_tls'], y=my_qH['TreeHeight'])
my_lm = pg.linear_regression(my_qH['refTH_tls'], my_qH['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the difference from the mean diameter:
myMultiDiff = pd.Series(my_lm.residuals_)
# get iqr:
threshold = 0.95
q3 = myMultiDiff.quantile(0.75)
q1 = myMultiDiff.quantile(0.25)
iqr =  q3 - q1
lower_bound = q1 - threshold * iqr
upper_bound = q3 + threshold * iqr
# remove according to the iqr:
ind_goodPts = (myMultiDiff >= lower_bound) & (myMultiDiff <= upper_bound)
ind_outPts = (myMultiDiff < lower_bound) | (myMultiDiff > upper_bound)
#
my_qH2 = my_qH[ind_goodPts.to_list()]
myOut2 = my_qH[ind_outPts.to_list()]

# plot
fig, ax = plt.subplots()
sns.regplot(data=my_qH2, x='refTH_tls', y='TreeHeight', color='k',  scatter=False , ax=ax)
sns.scatterplot(data=my_qH2, x="refTH_tls", y="TreeHeight", color='royalblue' , ax=ax)
sns.scatterplot(data=myOut2, x="refTH_tls", y="TreeHeight", color='red' , ax=ax)
ax.annotate('$R^2$: ' + str("{:.2f}".format(r_value**2)), xy=(8, 30))
ax.set_xlabel('TLS Height')
ax.set_ylabel('Tree-Quest Height')
fig.show()

# export the outliers:
#myOut2.to_parquet(dataFolder + r'\outliers_in_high_TH_LaxPark_part2.parquet')

# ------------------------------------------------
#  TQ vs. FI: get the regression parameters:
# ------------------------------------------------
m_tq_fi, b_tq_fi, r_value_tq_fi, p_value, std_err = scipy.stats.linregress(my_qH2['refTH_man'], my_qH2['TreeHeight'])
stat_out = pg.corr(x=my_qH2['refTH_man'], y=my_qH2['TreeHeight'])
my_lm = pg.linear_regression(my_qH2['refTH_man'], my_qH2['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_qH2['refTH_man'] - my_qH2['TreeHeight']
my_rmse_tq_fi = ((my_err**2).mean()**0.5)
my_mae_tq_fi = (my_err.abs().mean())

# get rMAE
rMAE_tq_fi = my_qH2['errTH_man_Rel'].mean()

# ------------------------------------------------
#  TQ vs. TLS: get the regression parameters:
# ------------------------------------------------
#
m_tq_tls, b_tq_tls, r_value_tq_tls, p_value, std_err = scipy.stats.linregress(my_qH2['refTH_tls'], my_qH2['TreeHeight'])
stat_out = pg.corr(x=my_qH2['refTH_tls'], y=my_qH2['TreeHeight'])
my_lm = pg.linear_regression(my_qH2['refTH_tls'], my_qH2['TreeHeight'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_qH2['refTH_tls'] - my_qH2['TreeHeight']
my_rmse_tq_tls = ((my_err**2).mean()**0.5)
my_mae_tq_tls = (my_err.abs().mean())

# ------------------------------------------------
#  TLS vs. FI: get the regression parameters:
# ------------------------------------------------
m_tls_fi, b_tls_fi, r_value_tls_fi, p_value, std_err = scipy.stats.linregress(my_qH2['refTH_man'], my_qH2['refTH_tls'])
stat_out = pg.corr(x=my_qH2['refTH_man'], y=my_qH2['refTH_tls'])
my_lm = pg.linear_regression(my_qH2['refTH_man'], my_qH2['refTH_tls'])
my_lm.round(2)
pd.Series(my_lm.residuals_).describe().round(3)

# get the RMSE amd MAE in [cm]:
my_err =  my_qH2['refTH_man'] - my_qH2['refTH_tls']
my_rmse_tls_fi = ((my_err**2).mean()**0.5)
my_mae_tls_fi = (my_err.abs().mean())

# get rMAE
rMAE_tls_fi = my_qH2['errTH_tlsMan_Rel'].mean()

# ------------------------------------------------
# make a joint figure -- Figure 5
# ------------------------------------------------
# make input lists:
myX = ['refTH_man', 'refTH_tls', 'refTH_man']
myY = ['TreeHeight', 'TreeHeight', 'refTH_tls']
myR = [r_value_tq_fi, r_value_tq_tls, r_value_tls_fi]
my_m = [m_tq_fi, m_tq_tls, m_tls_fi]
my_b = [b_tq_fi, b_tq_tls, b_tls_fi]
my_MAE = [my_mae_tq_fi, my_mae_tq_tls, my_mae_tls_fi]
my_label = ['(a)', '(b)', '(c)']
my_xlabel = ['FI Height [m]', 'TLS Height [m]', 'FI Height [m]']
my_ylabel = ['Tree-Quest Height [m]', 'Tree-Quest Height [m]', 'TLS Height [m]']
# plotting
fig, axx = plt.subplots(1, 3, figsize=(12, 12./3))
for ax, x1, y1, r1, m1, b1, mae1, label1, xlabel1, ylabel1 in (zip(axx, myX, myY, myR, my_m, my_b, my_MAE, my_label, my_xlabel, my_ylabel)):
    ax.axline((0, 0), slope=1, color='k', linestyle=':', label='1-1 line')
    sns.scatterplot(data=my_qH2, x=x1, y=y1, color='royalblue', ax=ax)
    sns.regplot(data=my_qH2, x=x1, y=y1, color='k', scatter=False, ax=ax)
    ax.annotate('$R^2$: ' + str("{:.2f}".format(r1 ** 2)), xy=(8, 32))
    ax.annotate('$y=$' + str("{:.2f}".format(m1)) + '$x$ + ' + str("{:.2f}".format(b1)), xy=(8, 30))
    ax.annotate('MAE: ' + str("{:.1f}".format(mae1)) + ' m', xy=(8, 28))
    ax.annotate('$n = $' + str("{}".format(len(my_qH2[y1]))), xy=(8, 26), size=15)
    ax.text(-0.15, -0.1, label1, transform=ax.transAxes, size=17)
    ax.set_xlabel(xlabel1)
    ax.set_ylabel(ylabel1)
    ax.set_xlim([5, 35])
    ax.set_ylim([5, 35])
    ax.grid(True, lw=0.4)
    ax.set_aspect('equal')
    ax.set_yticks(np.arange(10, 35, 5))
    ax.set_xticks(np.arange(10, 35, 5))

fig.tight_layout()
fig.show()

figPath = r'c:\path\to\folder\where\figures\will\be\stored'
fig.savefig(figPath + r'\TH_LaxPark_highQuality.png', dpi=300)

# --------------------------------------------------------
# export the high-quality and outlier filtered measurements
# --------------------------------------------------------
my_qH2.to_parquet(dataFolder + r'\qualHigh_outFiltered_TH_LaxPark.parquet')

# --------------------------------------------------------
# Check if normally distributed and do the stat test
# --------------------------------------------------------

# check if the data normally distributed:
my_qH2['TreeHeight'].hist(bins=20)
my_qH2['refTH_man'].hist(bins=20)
#
scipy.stats.probplot(my_qH2['TreeHeight'], dist="norm", plot=plt)

#
scipy.stats.shapiro(my_qH2['TreeHeight'])
# --> not normally distributed
scipy.stats.shapiro(my_qH2['refTH_man'])
# --> not normally distributed

# check if the data distributions are the same (a Wilcoxon signed-rank test for paired data)
# non-parametric alternative for paired t-test
stat1, p_value1 =scipy.stats.wilcoxon(my_qH2['TreeHeight'], my_qH2['refTH_man'], alternative='two-sided')
stat2, p_value2 =scipy.stats.wilcoxon(my_qH2['refTH_tls'], my_qH2['refTH_man'], alternative='two-sided')
stat3, p_value3 =scipy.stats.wilcoxon(my_qH2['TreeHeight'], my_qH2['refTH_tls'], alternative='two-sided')
#
results1 = pg.wilcoxon(my_qH2['TreeHeight'], my_qH2['refTH_man'])
print(results1)
results2 = pg.wilcoxon(my_qH2['refTH_tls'], my_qH2['refTH_man'])
print(results2)
results3 = pg.wilcoxon(my_qH2['TreeHeight'], my_qH2['refTH_tls'])
print(results3)

# both p-values are < 0.05 --> the measurements are different --> we have to apply the correction functions

# examine the differences:
ax = pg.plot_blandaltman(my_qH2['TreeHeight'], my_qH2['refTH_man'])
plt.tight_layout()

# TQ - FI bias
bias1 = (my_qH2['TreeHeight'] - my_qH2['refTH_man']).median()
bias11 = (100*(my_qH2['TreeHeight'] - my_qH2['refTH_man'])/my_qH2['refTH_man']).median()

# TQ - TLS bias
bias2 = (my_qH2['TreeHeight'] - my_qH2['refTH_tls']).median()
bias22 = (100*(my_qH2['TreeHeight'] - my_qH2['refTH_tls'])/my_qH2['refTH_tls']).median()

# TLS - FI bias
bias3 = (my_qH2['refTH_tls'] - my_qH2['refTH_man']).median()
bias33 = (100*(my_qH2['refTH_tls'] - my_qH2['refTH_man'])/my_qH2['refTH_man']).median()