
# coding: utf-8

# In[1]:

import pandas as pd
from scipy.stats import norm
import csv
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
import probscale
from matplotlib.backends.backend_pdf import PdfPages


# In[2]:

df = pd.read_csv(r'C:\Users\andrew.collord\Documents\PCRB\CESL Cross Qualification\Python Sameness\CESL Cross Qual Sameness Data.csv')

splits = pd.read_csv(r'C:\Users\andrew.collord\Documents\PCRB\CESL Cross Qualification\Python Sameness\SplitLotList.txt', sep=" ", header = None, names = ['Lot'])
# splits.Lot.apply(lambda x: x[:6])
splits = list(splits.Lot)

# Catagorize the data as Split or POR
df.loc[df.SOURCE_LOT.isin(splits), 'split'] = 'Split'
df.loc[~df.SOURCE_LOT.isin(splits), 'split'] = 'POR'
df['split'] = df['split'].astype('category')
df.split = df.split.cat.set_categories(['POR', 'Split'])


# Import split lot data

# In[4]:

# Catagorize The Data for Opens / Shorts / Bad
bad_values = [99995, 99996, 99997, 99999]

def bad_map(x):
    x = int(x)
    if x == 99995:
        return 'Bad'
    elif x == 99999:
        return 'Bad'
    elif x == 99996:
        return 'Short'
    elif x == 99997:
        return 'Open'
    
    return 'Good'


df['value_cat'] = df.SITE_VALUE.apply(bad_map)

# Filter parameters with less than 25% of data points (detail parameters)
count_pivot = df.pivot_table(values='SITE_VALUE', index=['PARAMETER'], aggfunc=[len] , margins=False)['len']
count_filter = count_pivot > count_pivot.quantile(0.75) *0.25
filtered_parameters = count_pivot[count_filter].index.tolist()


# In[5]:

# Summary table of Bad data
bad_pivot = df.pivot_table(index=['PARAMETER', 'split'], columns='value_cat' , values='SITE_VALUE',  aggfunc='count', fill_value=0)
bad_pivot['Percent_Bad'] = bad_pivot.Bad/(bad_pivot.Bad + bad_pivot.Good) * 100
bad_pivot['Percent_Open'] = bad_pivot.Open/(bad_pivot.Open + bad_pivot.Good) * 100
bad_pivot['Percent_Short'] = bad_pivot.Short/(bad_pivot.Short + bad_pivot.Good) * 100

# Make the filter values categorical 
# This can't be done before making the pivot tables or you can't append the pivot table with the percent values
df['value_cat'] = df['value_cat'].astype('category')
df['value_cat'] = df['value_cat'].cat.set_categories(['Good', 'Bad', 'Short', 'Open'])


# Add Data 6 IQR Data Filter

# In[6]:

def pseudofilter(x):
    ''' Pseudo Sigma filter by spec
    
    parameter x: dataframe
    return dataframe with extra columns (low limit and high limit)
    '''
    
    filtered_value = x[x.value_cat == 'Good'].SITE_VALUE
    def doMath(value):
        six_iqr =  ((value.quantile(0.75) - value.quantile(0.25)) * 6 / 1.33)
        pseudo_high =  value.median() + six_iqr
        pseudo_low =  value.median() - six_iqr
        return pseudo_low, pseudo_high
    
    x['pseudo_low'], x['pseudo_high'] = doMath(filtered_value)
    return x

groups = df.groupby(['PARAMETER', 'split'])
df = groups.apply(pseudofilter)


# Create the Filtered dataset and generate stats

# In[7]:

# Filters values x where -6IQR < x < +6IQR and values are "Good"
f = (df.SITE_VALUE > df.pseudo_low) & (df.SITE_VALUE < df.pseudo_high) & (df.value_cat == 'Good')
df_filtered = df[f]

# Calculate the mean, stdev, and number of datapoints of the primary data
stats_pivot = df_filtered.pivot_table(values='SITE_VALUE',
                                      index=['PARAMETER', 'split'],
                                      aggfunc=[np.mean, np.std, len])

stats_pivot = stats_pivot.unstack(level = 1)

# Calculate the number of filtered values by parameter
filtered_pivot = df[~f].pivot_table(values='SITE_VALUE',
                                    index=['PARAMETER', 'split'],
                                    aggfunc=[len, ],
                                    fill_value=0)

filtered_pivot = filtered_pivot.rename(index=str, columns={'len': 'Filtered_Count'})
filtered_pivot = filtered_pivot.unstack(level=1)

#Join the filtered parameter stats table to the bulk dataset
combined_stats = filtered_pivot.join(stats_pivot)
combined_stats.columns = combined_stats.columns.droplevel(level = 1)


# In[8]:

# Calculate the % difference in filtered data between split and POR, append as a new column
fc = combined_stats['Filtered_Count']
fc1 = (fc['Split'] - fc['POR']) * 1.0 / fc['POR']
combined_stats.loc[:, ('Filtered_Count', 'diff')] = fc1

stats_pivot.columns = stats_pivot.columns.droplevel(level = 1)


# Sameness Analysis

# In[9]:

def sameness(x):
    por_mean = x.loc[('mean', 'POR')]
    por_std = x.loc[('std', 'POR')]
    split_mean = x.loc[('mean', 'Split')]
    split_std = x.loc[('std', 'Split')]
    
    
    lower_3sigma = por_mean - (3 * por_std)
    upper_3sigma = por_mean + (3 * por_std)
    
    norm_split = norm(split_mean, split_std)
    
    lower_cdf = norm_split.cdf(lower_3sigma)
    upper_cdf = norm_split.cdf(upper_3sigma)
    
    sameness = (upper_cdf - lower_cdf) / 0.997
    
    return sameness

combined_stats['Sameness'] = stats_pivot.apply(sameness, axis=1)


# In[10]:

# Filter for split lots that have failed sameness
combined_pivot = combined_stats.join(bad_pivot.unstack(level=1))
sameness_filter = combined_pivot.Sameness < 0.95
sameness_filter = sameness_filter.rename('FailSameness')

# Filter for split lots that failed for filtered data - add column that indicates 
split_fails_filter = ((combined_pivot['Filtered_Count', 'Split'] > 1) & (combined_pivot['Filtered_Count', 'POR'] == 0)) | (combined_pivot['Filtered_Count', 'diff'] > 1)
split_fails_filter = split_fails_filter.rename('FailFilter')

# Merge sameness and failed filter indicators to df with all parameters
df = df.merge(sameness_filter.reset_index(), on='PARAMETER')
df = df.merge(split_fails_filter.reset_index(), on='PARAMETER')


# In[32]:

# Append the fail filter and sameness data to the stats table
cs = combined_stats.reset_index().merge(sameness_filter.reset_index(), on='PARAMETER')
cs = cs.merge(split_fails_filter.reset_index(), on='PARAMETER')
cs[ ('Filtered_Count', 'diff')].fillna(0, inplace = True)


# In[ ]:




# In[31]:

cs


# In[35]:

cs.columns


# In[ ]:




# In[ ]:




# In[ ]:




# In[12]:

#plot_data = df[(df.PARAMETER == 'VIA_TV1R_Cx112_Res') & (df.FailFilter == True) & (df.value_cat == 'Good')]
plot_data = df[(df.FailSameness == True) & (df.value_cat == 'Good')]
g = plot_data.groupby('PARAMETER')


# In[77]:

def prob_plot(param, data):
    
    cols =  [('PARAMETER', ''), ('Filtered_Count', 'diff'), ('Sameness', ''), 'FailSameness', 'FailFilter']
    statable = cs[cs.PARAMETER == parameter][cols]
    statable.columns = ['Parameter', '%diff Filtered', 'Sameness', 'Fail Sameness?', 'Fail Filter?']
    statable = statable.round(decimals = 3)
    
    fg = (sns.FacetGrid(data=data, hue='split', size=8, aspect=1.25, table = statable) 
        .map(probscale.probplot, 'SITE_VALUE', probax='y')
        .set_ylabels('Normal Probability')
        .add_legend()
         )

    fg.set(xlabel = 'Site Value',  
            ylabel = 'Normalized Probability', 
            title = param)
    fg.ax.set_ylim(bottom=0.05, top=99.95)
    
       
    
#     plt.table(cellText=statable,
#           rowLabels=param.index,
#           colLabels=param.columns,
#           cellLoc = 'right', rowLoc = 'center',
#           loc='right', bbox=[.65,.05,.3,.5])

    return fg


# In[14]:




# In[ ]:




# In[78]:

testlist = list(plot_data.PARAMETER.unique()[:5])
testees = plot_data.loc[plot_data.PARAMETER.isin(testlist)]
pairoftestees = testees.groupby('PARAMETER')


# In[79]:

for parameter, data in pairoftestees:
    prob_plot(parameter,data)


# In[16]:

with PdfPages('testplots.pdf') as pdf:

    for parameter, data in pairoftestees:
        prob_plot(parameter,data)
        
        pdf.savefig()
        plt.close()






# In[75]:

for parameter, data in pairoftestees:
    cols =  [('PARAMETER', ''), ('Filtered_Count', 'diff'), ('Sameness', ''), 'FailSameness', 'FailFilter']
    xx = cs[cs.PARAMETER == parameter][cols]
    xx.columns = ['Parameter', '%diff Filtered', 'Sameness', 'Fail Sameness?', 'Fail Filter?']
    xx = xx.round(decimals = 3)
    
    
    print(xx)
#     for parameter, data in pairoftestees:
#         prob_plot(parameter,data)
    
    
#     print(xx)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[514]:

def prob_plot(param, data):
    fg = (sns.FacetGrid(data=data, hue='split', size=8, aspect=1.25) 
        .map(probscale.probplot, 'SITE_VALUE', probax='y')
        .set_ylabels('Normal Probability')
        .add_legend()
         )

    fg.set(xlabel = 'Site Value',  
            ylabel = 'Normalized Probability', 
            title = param)
    fg.ax.set_ylim(bottom=0.05, top=99.95)
    
    plt.table(cellText=data.Sameness,
          rowLabels=summary.index,x
          colLabels=summary.columns,
          cellLoc = 'right', rowLoc = 'center',
          loc='right', bbox=[.65,.05,.3,.5])
    
    return fg

# for parameter, data in g:
#     prob_plot(parameter,data)


# In[ ]:


    


# In[ ]:




# In[493]:

# This one works

param = '2PhiP5V_0.4'
data = plot_data.loc[plot_data.PARAMETER ==param]


fg = (sns.FacetGrid(data=data, hue='split', size=6, aspect=1.25) 
    .map(probscale.probplot, 'SITE_VALUE', probax='y')
    .set_ylabels('Normal Probability')
    .add_legend()
     )

fg.set(xlabel = 'Site Value',  
        ylabel = 'Normalized Probability', 
        title = param)
fg.ax.set_ylim(bottom=0.05, top=99.95)

    


# In[492]:

df.head()
    


# #Converted to markdown to supress
# plot_data = df[(df.FailFilter == True )& (df.value_cat == 'Good']
# g = plot_data.groupby('PARAMETER')
# #Plot Tail Fails
# for param, data in g:
#     means = combined_stats.loc[param, 'mean']
#     stds = combined_stats.loc[param, 'std']
#     title = '{0} \n'.format(param)
#     title = title + 'POR: Mean = {0:.3f}, Std= {1:.4f} \n'.format(means['POR'], stds['POR']
#     title = title + 'Mean = {0:.3f}, Std= {1:.4f}'.format(means['Split'], stds['Split'])
#                                                                   
#     filename = 'out\{0}_fail_filtered_prob.png'.format(param)
#     prob_plot(param, data, title, filename)

# In[480]:

# fig, ax = plt.figure(figsize = (10,4))
# ax2 = fig.add_subplot

param = '2PhiP5V_0.4'
data = plot_data.loc[plot_data.PARAMETER ==param]
# splitdata = data.loc[data.split == ' Split']
# pordata = data.loc[data.split == 'POR']

fig = sns.kdeplot(data.SITE_VALUE, cumulative = True)

fig.set(xlabel = 'Site Value',  
        ylabel = 'Normalized Probability', 
        title = param)
# fig.axvline( x = )


# In[474]:

tp


# In[425]:

cs = combined_stats.reset_index().merge(sameness_filter.reset_index(), on='PARAMETER')
cs = cs.merge(split_fails_filter.reset_index(), on='PARAMETER')


# ## Data summary output is here
#  Recomend to feed the failed parameters into ACE to generate the plots

# In[426]:

cs.head()
# cs.to_csv('Parameter_Summary.csv')


# In[ ]:



