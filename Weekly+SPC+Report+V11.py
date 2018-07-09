
# coding: utf-8

# In[1]:

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import cx_Oracle
from sqlalchemy import create_engine
import datetime as dt  

# Turn on Filter Parameters? True will filter values with no measurement in last 50 days, 
# parameters with Pts < 45, and infinite Cpk values
myfilters = False

direc = r'C:\Users\andrew.collord\Documents\Python Scripts\Weekly SPC\FY18'
WW = dt.date.isocalendar(dt.date.today())[1] + 26 - 52
FY = '18'

header = 'FY' + FY  + str(WW)
exportname = header + " SPC Report.xlsx"

WW


# In[2]:

def importSPCreport(WW):
    
    path = direc + '\\' + 'WW' + WW + ".xls"
    xl = pd.ExcelFile(path)
    
    CMP = xl.parse("CMP", skiprows = 26)
    Copper = xl.parse("Copper", skiprows = 26)
    Diffusion = xl.parse("Diffusion", skiprows = 26)
    Epi = xl.parse("Epi", skiprows = 26)
    Implant = xl.parse("Implant", skiprows = 26)
    Metals = xl.parse("Metals", skiprows = 26)
    PECVD = xl.parse("PECVD", skiprows = 26)
    Photo = xl.parse("Photo", skiprows = 26)
    Plating = xl.parse('Plating', skiprows = 26)
    Plasma = xl.parse("Plasma", skiprows = 26)
    Wet = xl.parse('Wet', skiprows = 26)
    
    data = pd.concat([CMP,Copper, Diffusion, Epi, Implant, Metals, PECVD,
                       Photo, Plating, Plasma, Wet], ignore_index = True)
    
    data = data[data.Parameter != 'Parameter']
    cols = ['Control', 'WS', 'MTT', '%OOC']
    data[cols] = data[cols].fillna(False)
    
    data = data[['Parameter', 'Area', 'Date Start', 'Pts', 'Mean', 'SD', 'Cp', 'Cpk', '% Fail', 
       'WS Ratio (UCL-LCL)/(6*SD)', '(MEAN -TGT)/SD', '% OOC (against fixed limits)', 'Control', 'WS',
       'MTT', '%OOC']]
    
    data.columns = [['Parameter', 'Area', 'Date Start', 'Pts', 'Mean', 'SD', 'Cp', 'Cpk',  '%Fail',  
                 'WS Ratio', 'MTT', '%OOC', 'calcCpk', 'calcWS', 'calcMTT', 'calcOOC']]
    
   
    
#     cols = ['calcCpk', 'calcWS', 'calcMTT', 'calcOOC']
#     data[cols] = data[cols].fillna(False)
#     data[cols] = data[cols].astype(bool)
    
    ints = ['Pts', 'Mean', 'SD', 'Cp', 'Cpk',  '%Fail', 'WS Ratio', 'MTT', '%OOC']
    data[ints] = data[ints].astype(float)
    
#     dates = ['Date Start']
#     data[dates] = data[dates]
    
    return(data)



def pullparameters():
    
    engine = create_engine('oracle+cx_oracle://ro:ro@muon')

    sql = '''
    select area_list.area, machine_list.machine,node.node, NODE_MACHINE.DAQ as MDAS
    from prime.node_machine 
    join prime.machine_list 
    on node_machine.machine_index = MACHINE_LIST.INDEXNO
        join prime.node on NODE_MACHINE.NODE_INDEX = NODE.INDEXNO 
        join prime.area_list on MACHINE_LIST.AREA_INDEX = AREA_LIST.INDEXNO
    where machine_list.active = 1 and node.active = 1
    order by area,machine,node,daq
    '''

    allparamlist = pd.read_sql(sql, engine)
    return(allparamlist)


 # merge the Cpk data with tool/parameter/owner data, then calc MTT, WSR, %OOC
def addowner(data):
    
    allparamlist = pullparameters()
    # List of areas I want to pull the SPC data from
    areas = [ 'CMP', 'COPPER','DIFFUSION 2', 'EPI', 'IMPLANT', 'METALS', 'PECVD', 'PHOTO', 'PLASMA', 'WET PROCESSES']

    # import the file which maps the tools to owners
    toolowners = pd.DataFrame.from_csv(r'C:\Users\andrew.collord\Documents\Python Scripts\Weekly SPC\Master Tool Owner List MAIN2.csv')

    # merge the tool-owner to the parameter-tool to get parameter-owner relationship
    paramlist = allparamlist[allparamlist.area.isin(areas)]
    toolparamowner = paramlist.merge(toolowners, on = 'machine', how = 'left')
    toolparamowner = toolparamowner.drop_duplicates(subset = ['node', 'area'])
    toolparamowner = toolparamowner.drop('mdas', axis = 1)

    # Apparently the listing in the Parameterlist doesn't match that of the Cpk report...
    #toolparamowner.area = toolparamowner.replace({'area' : {'DIFFUSION 2': 'DIFF2', 'WET PROCESSES': 'WET'}})    
    
    data = data.merge(toolparamowner, how= 'left', left_on = ['Parameter','Area'], right_on = ['node','area'])
    data = data.drop(['machine', 'node', 'area'], axis = 1)
    data.rename(columns={'owner':'Owner'}, inplace=True)
    data = data[~data.Parameter.isnull()]
    
    return(data)
    
    
# sort all the failing parameters by the parameter type they are failing
def failfunc(alldata):
    
    alldata['Fail Cpk?'] = ((alldata['Cpk'] < 1.33) & (alldata['calcCpk']))
    alldata['Fail WS?'] = (alldata['WS Ratio'] > 1.33) & alldata['calcWS']
    alldata['Fail MTT?'] = (abs(alldata['MTT']) > 1.5) & alldata['calcMTT']
    alldata['Fail %OOC?'] = (alldata['%OOC'] > 5) & alldata['calcOOC']
    alldata['Fails'] = alldata[['Fail Cpk?', 'Fail WS?', 'Fail MTT?', 'Fail %OOC?']].sum(axis=1)
    
    alldata = alldata[~alldata.Parameter.isnull()]
    alldata['Owner'] = alldata.Owner.fillna('Unknown')
    
    return(alldata)


def spcdelta(alldata, lastalldata):
    
    lastalldata.columns = lastalldata.columns.str.lower()
    
    combo = alldata.merge(lastalldata, left_on= ['Area', 'Parameter'], right_on = ['area', 'parameter'])
    combo = combo[combo.Cpk < 10000]
    combo['cpkdelta'] = combo['Cpk'] - combo['cpk']
    combo['oocdelta'] = combo['%OOC'] - combo['%ooc']
    combo = combo[['Parameter', 'Area', 'Date Start', 'Pts', 'Mean', 'SD', 'Cp', 'Cpk',
       '%Fail', 'WS Ratio', 'MTT', '%OOC', 'calcCpk', 'calcWS', 'calcMTT',
       'calcOOC', 'Owner',  'cpkdelta', 'oocdelta']]
    combo = combo[((combo['calcCpk'] == True) & (combo.cpkdelta < -0.1)) | ((combo['calcOOC'] == True) & (combo.oocdelta > 0.25))]

    return (combo)


def addsupercritical(alldata):
    # Import the list of "super-critical" parameters
    scloc = r'C:\Users\andrew.collord\Documents\Python Scripts\Weekly SPC\Parameter Mapping\Super Critical.csv'
    sclist = pd.read_csv(scloc)
    
    sclist = sclist.merge(alldata, on = ['Area', 'Parameter'])
    sclist = sclist[[ 'Parameter', 'Area', 'Date Start', 'Pts', 'Mean',
       'SD', 'Cp', 'Cpk', '%Fail', 'WS Ratio', 'MTT', '%OOC', 'calcCpk',
       'calcWS', 'calcMTT', 'calcOOC', 'Owner', 'Fail Cpk?', 'Fail WS?',
       'Fail MTT?', 'Fail %OOC?', 'Fails']]
    
    return(sclist)


# In[3]:

def cpkreportgen(WW):
    
#     lastcpkreport = importSPCreport(str(WW-1))
    
#     thiscpkreport = importSPCreport(str(WW))    
    lastcpkreport = importSPCreport('52')
    
    thiscpkreport = importSPCreport('01')    
    thiscpkreport = addowner(thiscpkreport) 
    thiscpkreport = failfunc(thiscpkreport)

    combo = spcdelta(thiscpkreport, lastcpkreport)
    sclist = addsupercritical(thiscpkreport)
    
    return(thiscpkreport, combo, sclist)

alldata, combo, sclist = cpkreportgen(WW)


# In[4]:

writer = pd.ExcelWriter(exportname)

failingdata = alldata[alldata.Fails > 0]

failingdata.to_excel(writer,'All Failing Parameters')

# Picks the parameters falling more than 2 metrics to put as highest priority
Worst = failingdata[(failingdata.Fails > 2)].sort_values('Fails', ascending = False).to_excel(writer, 'Worst Parameters')

# These are the parameters set forth in the SPC spec doc 10-0009F
bestdata = alldata[(alldata['Cp'] > 1.9) & (alldata['%OOC'] < 1)]

owners = sorted(alldata.Owner.unique())

for owner in owners:
    databyowner = alldata[alldata.Owner == owner]
    ownerdelta = combo[combo.Owner == owner]
    failsbyowner = failingdata[failingdata.Owner == owner]
    deltacpkdata = combo[combo.Owner == owner]
    scbyowner = sclist[sclist.Owner == owner]
    
    if owner == 'Unknown':
        failsbyowner.to_excel(writer, owner)
    
    elif owner == 'Decomissioned':
        pass

    else:
        OwnerWorst = failsbyowner[failsbyowner.Fails > 2].sort_values('Fails', ascending = False)
        scbyowner = scbyowner[scbyowner.Fails > 0].sort_values('Fails', ascending = False)
        cpkdelta = ownerdelta[ownerdelta.cpkdelta < -0.1].sort_values('cpkdelta',ascending = True).drop('oocdelta',1)
        oocdelta = ownerdelta[ownerdelta.oocdelta > 0.25].sort_values('oocdelta',ascending = False).drop('cpkdelta',1)
        highoochighws = failsbyowner[(failsbyowner['Fail %OOC?'] == True) & 
                                    (failsbyowner['WS Ratio'] < 1)].sort_values('WS Ratio',ascending = True)
        highoochighws = highoochighws[~highoochighws.Parameter.str.contains('PARTICLE')]
        WorstCpk = failsbyowner[failsbyowner['Fail Cpk?'] == True].sort_values('Cpk', ascending = True)
        WorstWS = failsbyowner[failsbyowner['Fail WS?'] == True].sort_values('WS Ratio', ascending = False)
        WorstMTT = failsbyowner[failsbyowner['Fail MTT?'] == True].sort_values('MTT', ascending = False)
        WorstOOC = failsbyowner[failsbyowner['Fail %OOC?'] == True].sort_values('%OOC', ascending = False)
        bestdata = bestdata[bestdata['Owner'] == owner].sort_values('Cp', ascending = False)
        
        
        OwnerWorst.to_excel(writer, owner, index_label = 'Highest Priority')
        long = len(OwnerWorst) + 5
        
        scbyowner.to_excel(writer, owner, startrow = long, index_label = 'Super Critical Fails')
        long += len(scbyowner) + 5
        
        cpkdelta.to_excel(writer, owner, startrow = long, index_label = 'Biggest Cpk Delta')
        long += len(cpkdelta) + 5
        
        oocdelta.to_excel(writer, owner, startrow = long, index_label = 'Biggest OOC Delta')
        long += len(oocdelta) + 5
        
        WorstCpk.to_excel(writer, owner, startrow = long, index_label = 'Worst Cpk')
        long += len(WorstCpk) + 5
        
        WorstWS.to_excel(writer, owner, startrow = long, index_label = 'Worst WS')
        long += len(WorstWS) + 5
        
        WorstMTT.to_excel(writer, owner, startrow = long, index_label = 'Worst MTT')
        long += len(WorstMTT) + 5
    
        WorstOOC.to_excel(writer, owner, startrow = long, index_label = 'Worst OOC')
        long += len(WorstOOC) + 5
        
        highoochighws.to_excel(writer, owner, startrow = long, index_label = 'Good WS, Bad OOC')
        long += len(highoochighws) + 5
        
        bestdata.to_excel(writer, owner, startrow = long, index_label = 'Stop Measuring?')
        
metrics = ['Fail Cpk?', 'Fail WS?', 'Fail MTT?', 'Fail %OOC?']
reported = ['calcCpk', 'calcWS', 'calcMTT', 'calcOOC']
rename = ['Cpk', 'WS', 'MTT', '%OOC']

failsbyeng = failingdata[ metrics + ['Owner']]
failsbyarea = failingdata[metrics + ['Area']]

# total number of fails by area/owner
areafails = failsbyarea.groupby('Area').sum()
ownerfails = failsbyeng.groupby('Owner').sum()

# total number of parameters - note they must have "True" under each category
totalareaparams = alldata.groupby('Area')[reported].sum()
totalownerparams = alldata.groupby('Owner')[reported].sum()


ownerfails.columns = [rename]
totalownerparams.columns = [rename]

areafails.columns = [rename]
totalareaparams.columns = [rename]

areapercentpass = (1- areafails/totalareaparams)
ownerpercentpass = (1- ownerfails/totalownerparams)

areapercentpass.to_excel(writer, 'Statistics')
totalareaparams.to_excel(writer, 'Statistics', startrow = 15)


ownerpercentpass.to_excel(writer, 'Statistics', startrow = 30)
totalownerparams.to_excel(writer, 'Statistics', startrow = 65)
        

writer.save()
writer.close()


# In[ ]:




# In[17]:

alldata[(alldata.Area == 'PLASMA') & (alldata.Cpk < 1.33)].sort_values(by = 'Cpk', ascending = True)


# In[14]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

plasmaooc = alldata[(alldata.Area == 'PLASMA') & (alldata['Fail %OOC?'] == True)]


# In[ ]:

temppath = r'C:\Users\andrew.collord\Documents\Python Scripts\Weekly SPC\Comparison to Ricks Report\Ricks Plasma OOC.csv'
ricksooc = pd.read_csv(temppath)


# In[ ]:

ricksooc[~ricksooc.Parameter.isin(plasmaooc.Parameter)]


# In[ ]:

rawcpkreport[(rawcpkreport.machine == 'ALL') & (rawcpkreport.parameter == '6IN ALLIANCE - POLY ER')]


# In[ ]:

sql3 = '''
SELECT * FROM PRIME.WEEKLY_CPK_DATA D
INNER JOIN PRIME.WEEKLY_CPK_CATEGORY_MAP F
ON D.CATEGORY_INDEX = F.INDEXNO
WHERE D.CATEGORY_INDEX > 99
AND (D.FY*100 + D.WW) >= 201725
'''

rawspchistory = pd.read_sql(sql3, engine)


# In[ ]:




# In[ ]:

rawspchistory[['cp', 'cpk']] = rawspchistory[['cp', 'cpk']].astype('float')
spchistory = cpkreportprep(rawspchistory)
#spchistory = spchistory[spchistory['Control?'] == True]
#spchistory = spchistory[['Area', 'Parameter', 'Cpk', 'MTT', 'WS Ratio', '% OOC', 'Owner', 'WW']]


#spchistory = spchistory[spchistory.Owner == 'Andrew Collord']
#mesa = pd.pivot_table(spchistory, index = ['Area', 'Owner', 'Parameter'], columns = 'WW', values = ['Cpk'])
#mesa['avg'] = mesa['Cpk'].mean(axis = 1)
#mesa


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

# The worst OOC charts from PLASMA
writer3 = pd.ExcelWriter('Worst Plasma OOCs.xlsx')
areyouforreal = failingdata[(failingdata.Area == 'PLASMA') & (failingdata['Fail %OOC?'] == True)].sort_values('% OOC', ascending = False)
areyouforreal.to_excel(writer3)
writer3.save()


# In[ ]:

# The worst Cpk charts from Plasma
writer4 = pd.ExcelWriter('Worst Plasma Cpk.xlsx')
metalswsr = failingdata[(failingdata.Area == 'PLASMA') & (failingdata['Fail Cpk?'] == True)].sort_values('Cpk', ascending = True)
metalswsr.to_excel(writer4)
writer4.save()
metalswsr


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

# Work with only the combined Cpk values, drop the 'ALL' machine column
cpkreport = thiscpkreport[thiscpkreport.machine == 'ALL']
cpkreport = cpkreport.drop('machine',1)

# merge the Cpk data with tool/parameter/owner data, then calc MTT, WSR, %OOC
cpkdata1 = cpkreport.merge(toolparamowner, how= 'left', left_on = ['parameter'], right_on = ['node'])
cpkdata2 = cpkreport.merge(toolparamowner, how= 'left', left_on = ['parameter','category'], right_on = ['node','area'])

print(len(cpkdata1))
print(len(cpkdata2))

cpkdata1[~cpkdata1.isin(cpkdata2)].dropna()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



