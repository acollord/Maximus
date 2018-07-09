
# coding: utf-8

# In[31]:

import pandas as pd
import datetime as dt
import numpy as np
from IPython.display import display
import re

def strip_tool(s):
    s_out=""
    for tool in ''.join(s).split(','):
        if 'DUM' not in tool:
            s_out += (tool[:6] ) + " "
    return s_out



# # Scrap Data

# In[50]:

scrap = pd.read_csv(r'\\mfndata2\vision\datafeed\ScrapRTV.csv')
scrap = scrap.drop([0,0])
scrap = scrap[['TransDate', 'ScrapRTV', 'FabLotID', 'Process', 'Area', 'Oper', 'Product', 'Wafers', 'TotalMoves', 'Category', 'Comment', 
               'SubArea','QtyCategory', 'Route']]
scrap['TransDate'] = pd.to_datetime(scrap['TransDate'])
scrap = scrap[scrap['TransDate'] >= pd.to_datetime((dt.date.today()-dt.timedelta(1)))]
scrap = scrap[scrap['Area'].isin(['METALS', 'PLASMA', 'PECVD'])]

scrap


# # WIP Data

# In[46]:

# Using an export from the GNR website
xl = pd.ExcelFile(r'C:\Users\andrew.collord\Documents\Morning Report\WIP.xls')
wip = xl.parse('WIP Data', skiprows = 2)

wip = wip[['Fab Lot #', 'Cassette ID', 'Process Family', 'Process Flow',  'Part Name', 'Lot Priority',  'Wafer Count', 'Mfg Area', 'Lot State',
        'MES Call Procedure', 'MES Operation', 'Eqp Type', 'Promis Call Procedure Name', 'Capability', 'Available Tools',  'Unavailable Tools',  
        'Running Equipment', 'Lot Hold Code', 'Promis Lot Hold Reason', 'Non Hold Comment(Non Promis)',  'Lot Hold User', 
       'Time At Step (Hours)', 'Time On Hold (Hours)']]

wip = wip[wip['Lot State'] != 'RUN']

# Fill blank cells in the Avail/Unavail tools before combining to new Tools column
wip['Unavailable Tools'].fillna('', inplace = True)
wip['Available Tools'].fillna('', inplace = True)
wip['Tools'] = wip['Unavailable Tools'] + wip['Available Tools']
wip['Tools'] = wip['Tools'].apply(strip_tool)
wip['Tools'].fillna('', inplace = True)

# Strip out (DISALLOWED)
wip['Unavailable Tools'] = wip['Unavailable Tools'].apply(strip_tool)

# Add a column indicating if the lot is stagnant, and if so put the number of wafers
wip['Stags'] = wip.loc[((wip['Time At Step (Hours)'] >= 5) & (wip['Process Family'] == 'S18')) | (
    (wip['Time At Step (Hours)'] >= 10) & (wip['Process Family'] != 'S18'))]['Wafer Count']
wip['Stags'].fillna(0, inplace = True)

# Filter down to relevant areas
acmeareas =['METALS8C', 'METALS', 'CVD', 'CVD8C', 'ETCH', 'ETCH8C']
acmefilter = ((wip['Mfg Area'].isin(acmeareas)) | (wip['Tools'].str.contains('AST')) | (wip['Tools'].str.contains('RTA')))
wip = wip[acmefilter]
wip = wip[~wip.Tools.str.contains('SINK')]


nopath = wip['Available Tools'] == 'No Path'
nopathsummary = wip[nopath][['Tools', 'Unavailable Tools', 'Eqp Type', 'Capability', 'Wafer Count', 'MES Call Procedure', 'Time At Step (Hours)', 'Stags']]
nopathtable = nopathsummary.groupby(['Eqp Type', 'Capability', 'Unavailable Tools']).agg({'Wafer Count': np.sum, 
                                                                                                                 'Stags' : np.sum, 
                                                                                                                 'Time At Step (Hours)' : np.median})


stags = (wip['Available Tools'] != 'No Path') & (wip['Stags'] > 0)
stags = wip['Stags'] > 0
stagtable = wip[stags][['Available Tools', 'Unavailable Tools', 'Eqp Type', 'Capability', 'Wafer Count', 'MES Call Procedure', 'Time At Step (Hours)', 'Stags']]
stagtable = stagtable.groupby(['Eqp Type', 'Capability', 'Unavailable Tools']).agg({'Wafer Count': np.sum,  
                                                                                                                     'Stags' : np.sum, 
                                                                                                                     'Time At Step (Hours)' : np.median})

stagtable = stagtable[stagtable.Stags > 25]
worststags = stagtable[stagtable.Stags > 100]


# # Equipment Status

# In[47]:

# Equipment Data Pull
xl2 = pd.ExcelFile(r'C:\Users\andrew.collord\Documents\Morning Report\Equipment.xlsx')
equip = xl2.parse('EqpStatusSnapshot')
equip = equip.iloc[1:]

def toolgetter(x):
    x = x.reset_index()
    
    if any( x.columns == 'Available Tools'):
        z = x['Available Tools'] + ',' + x['Unavailable Tools']
    else:
        z = x['Unavailable Tools']
    
    z = ''.join(z)
    z =z.replace(',', '')
    z = z.replace(' ', '')
    z = re.sub(r'(.{6})(?!$)','\\1 ', z)
    z = sorted(list(set(list(z.split()))))
    return z

def equip_status(x):
    z = toolgetter(x)
    
    y = pd.DataFrame()
    for tool in z:
         y = y.append(equip[equip.EqpID.str.contains(tool) == True]) 
    
    y = y[['EqpID', 'Status', 'Comment']]
    y.loc[y.Status == 'AVAIL', 'Comment'] = ''
    return y

stagequip = equip_status(stagtable)
nopathequip = equip_status(nopathtable)


# In[ ]:




# ['Safety Slide of the Day'](https://intranet.maxim-ic.com/manufacturinginternal/MFN/mfnsafety/startsafety/Shared%20Documents/06-19-18.pdf)

# In[51]:

display(scrap, nopathtable, stagtable)


# In[28]:

exportname = r"C:\Users\andrew.collord\Documents\Morning Report\Morning Report.xlsx"
writer = pd.ExcelWriter(exportname)

scrap.to_excel(writer, "Scrap", index_label = 'Scrap')

nopathlen = 0
nopathtable.to_excel(writer, "No Path", startrow = nopathlen)
nopathlen += len(nopathtable) + 5
nopathequip.to_excel(writer, "No Path", startrow = nopathlen, index = False)


staglen = 0
stagtable.to_excel(writer, "Stags", startrow = staglen)
staglen += len(stagtable) + 5
stagequip.to_excel(writer, "Stags", startrow = staglen, index = False)

writer.save()
writer.close()


# In[ ]:




# # Sandbox

# In[450]:

boobs = wip[['Mfg Area', 'Eqp Type', 'Capability', 'Tools']]
boobs[boobs['Eqp Type'].str.contains('NOV')]


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[374]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

# Equipment Data Pull
xl2 = pd.ExcelFile(r'C:\Users\andrew.collord\Documents\Morning Report\Equipment.xlsx')
equip = xl2.parse('EqpStatusSnapshot')
equip = equip.iloc[1:]

def equip_status(x):
    z = ''.join(x.index.get_level_values(3).tolist()).split(' ')[:-1]
    z = [i for i in z if len(i) > 0]
    
    y = pd.DataFrame()
    for tool in z:
         y = y.append(equip[equip.EqpID.str.contains(tool) == True]) 
    
    y = y[['EqpID', 'Status', 'Comment']]
    y.loc[y.Status == 'AVAIL', 'Comment'] = ''
    return y


stagequip = equip_status(stagtable)
nopathequip = equip_status(nopathtable,)


# In[ ]:




# In[ ]:

# Using the hourly WIP file
xl = pd.ExcelFile(r'\\mfn-production.maxim-ic.com\share\Prod_Ctr\Morning_Meeting_Reports\Excel Master FilesWIP.xlsx')
wip = xl.parse('WIP')
wip = wip[wip['Area'].isin(['METAL', 'PECVD', 'PLASMA'])]
wip = wip[['Size', 'Area', 'Subarea', 'Fablotid', 'PcLotid', 'QTY', 'Oper', 'Stepremain', 'Process', 'Route', 'HF', 'Product', 'lotstate', 'Qtime', 
                  'HoldComments', 'NON_PROMIS_COMMENTS', 'COMMENT_ENTERED_BY','AVAILABLE_TOOLS', 'UNAVAILABLE_TOOLS', 
                  'DateTime', 'State']]
wip['Stags'] = wip.loc[wip['State'] == 'Wait (Long)'].QTY


# In[278]:

nopath = wip[wip['AVAILABLE_TOOLS'] == 'No Path']
nopathsummary = nopath[['UNAVAILABLE_TOOLS', 'Subarea', 'QTY', 'Oper', 'Qtime', 'Stags']]
nopathsummary.groupby(['UNAVAILABLE_TOOLS', 'Oper']).agg({'QTY': np.sum, 'Stags' : np.sum,  'Qtime' : np.max})


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[166]:




# In[ ]:




# In[ ]:




# In[ ]:



