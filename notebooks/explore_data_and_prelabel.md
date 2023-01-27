---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
## Data Exploration
### insights
- charges on the bank account (dkb) appear in the paypal statements as pending and with type is bank deposit


### next steps:
1. [x] we dropped the wrong data from the paypal, get better match by looking for the positve entries in the paypal data, the one corresponding to the deposit are status=pending, that's why they have been dropped -> rewrite `get_paypal_index_for_single_row` to work with the raw paypal data and drop only later
1. [ ] drop paypal from dkb data without asignment to paypal and run
1. [ ] drop non relevant data from paypal without asignment to dkb and run
1. [ ] run models on paypal and dkb independently and run analysis
1. [ ] later try to match paypal and dkb data
<!-- #endregion -->

todos:
1. save data for each account -> add_data(filename, account)
2. to each account, find the paypal transactions and the don't consider them in the training, instead classify the remaining paypal data entries
3.  merge paypal and dkb to one dataset
<!--  merge with baysian model -->
 

```python
# ! python -m textblob.download_corpora
```

```python
# ! pip install pandas textblob colorama tabulate
```

```python
# WRKDIR = '/home/jovyan/work/'
# import sys

# sys.path.insert(0,WRKDIR)


%load_ext autoreload

%autoreload 2
```

```python
import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import re
from pathlib import Path
from collections import Counter
project_dir = str(Path(os.getcwd()).parent)
if not project_dir in sys.path:
    sys.path.insert(0,project_dir)
sys.path

from load_data import read_dkb_csv, read_paypal_csv
```

### define functions

```python
# def read_dkb_csv(filename, drop_duplicates=True, verbose=1)-> pd.DataFrame:
#     """Read a file in the CSV format that dkb provides downloads in.

#     Returns a pd.DataFrame with columns of 'date', 'desc', and 'amount'."""
#     account_nr = re.search('\d{10}', Path(filename).name)[0]
#     data=pd.read_csv(filename,sep=';', skiprows=9, encoding='latin-1',
#                      usecols=[0,2,3,4,7,9,10],
#                      names=['Buchungstag', 'Wertstellung', 'Buchungstext','Auftraggeber / Beguenstigter', 'Verwendungszweck', 'Kontonummer',
#                            'BLZ', 'Betrag (EUR)', 'Glaeubiger-ID', 'Mandatsreferenz','Kundenreferenz', 'Unnamed'] )

#     if drop_duplicates:
#         if verbose > 1:
#             duplicated_data = data[data.duplicated()]
#             print(f"dropping {len(duplicated_data)} duplicated entries")
#             if verbose > 2:
#                 if len(duplicated_data)>0:
#                     print(duplicated_data)
#                     print('\n')
#         data.drop_duplicates(inplace=True)
        
    
#     data = data[data['Verwendungszweck'] != 'Tagessaldo'] # drop all the Tagessaldo entries
    
#     print(f"number of data loaded: {len(data)}")
    
#     data['Buchungstag'] = pd.to_datetime(data['Buchungstag'], format = "%d.%m.%Y")
#     # data['Wertstellung'] = pd.to_datetime(data['Wertstellung'], format = "%d.%m.%Y")
#     data['account_nr'] = account_nr
#     if verbose > 1:
#         print(f"new data from {filename} with {len(data)} entries, years {data['Buchungstag'].dt.year.unique()}")


#     data.fillna("", inplace=True)
#     # if drop_duplicates:
#     #     if verbose > 1:
#     #         duplicated_data = data[data.duplicated()]
#     #         print(f"dropping {len(duplicated_data)} duplicated entries")
#     #         # if len(duplicated_data)>0:
#     #         #     print(duplicated_data)
#     #         #     print('\n')
#     #     data.drop_duplicates(inplace=True)
#     # print(len(data))

#     data['date'] = data['Buchungstag'] #.dt.strftime('%d/%m/%Y')
#     data['amount'] = data['Betrag (EUR)'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
#     data.drop(data[data['amount']==''].index, inplace=True)
#     data['amount'] = data['amount'].astype(float)

#     data = data[data['amount'].astype(float)!=0] # drop all zero values

#     desc_column_names = ['Buchungstext', 'Auftraggeber / Beguenstigter', 'Verwendungszweck']
#     data['desc'] = data[desc_column_names].agg(' '.join, axis=1)
#     df = data[['date', 'amount', 'desc', 'account_nr']]
#     # df = data


#     return df
```

```python


# def read_paypal_csv(filename,verbose=1)-> pd.DataFrame:
#     """Read a file in the CSV format that paypal provides downloads in.

#     Returns a pd.DataFrame with columns of 'datetime', 'description', and 'amount'."""

#     data = pd.read_csv(filename, decimal=",", dtype="string")

#     if verbose>1:
#         print(f"loaded {len(data)} paypal records")
        
#     data = data[data['Balance Impact']!='Memo']
#     if verbose>1:
#         print(f"{len(data)} paypal records after dropping `Balance Impact` = `Memo` entries")        
        
#     # data = data[data['Status']!='Pending']
#     # if verbose>1:
#     #     print(f"{len(data)} paypal records after dropping `Status` = `Pending` entries")     

#     type_list = [
#         'Bank Deposit to PP Account',
#         'Payment Refund`'
#         ]

#     #tmp comment out
#     # data = data[data['Type'].str.contains('|'.join(type_list))]
#     # # data = data[data['Status'] == 'Completed']
#     # if verbose>1:
#     #     print(f"{len(data)} status completed")


#     data['TimeZone'].replace('CEST', '+02:00', inplace=True)
#     data['TimeZone'].replace('CET', '+01:00', inplace=True)

#     data.insert(0, 'datetime', pd.to_datetime(data['Date'] + " " + data['Time']+ data['TimeZone'], format = "%d/%m/%Y %H:%M:%S%z", utc=True))
#     # data.drop(['Time', 'Date', 'TimeZone', 'Status'], axis=1, inplace=True) #tmp comment out


#     # to make the transaction ID unique we add the datetime
#     data['Transaction ID'] = data['Transaction ID'] + "-" +data['datetime'].dt.strftime("%d%m%Y %H:%M:%S%z")

#     data.set_index('Transaction ID', inplace=True)

#     # convert to floats
#     for k in ['Balance', 'Fee', 'Net', 'Gross']:
#         data[k] = data[k].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
#     # data = data[data['Balance'] != 0]

#     # insert new column description
#     data.insert(1, 'description', data[['Name', 'Country', 'Subject', 'Note']].fillna('').agg(' '.join, axis=1))

#     # data.drop(['Name', 'Country', 'Subject', 'Note'], axis=1, inplace=True) # drop the columns that are in the desciption now


#     data.rename(columns={"Gross": "amount"}, inplace=True)


#     # data = data[['datetime', 'description', 'Type',  'amount', 'From Email Address', 'To Email Address']]
#     # data = data[['datetime', 'Balance Impact', 'Status', 'description', 'Type',  'amount', 'From Email Address', 'To Email Address']]

#     return data

```

```python
def paypal_drop_reimbursed_positions(data, verbose=1):
    indecies_to_drop = []

    for i, r in data[data.duplicated()].iterrows():

        # print(data[data['amount']==-r['amount']])
        # print(data[(data['datetime']==r['datetime']) & (data['amount']==-r['amount'])])
        # if we find duplicated data we check if there was an immediate reimbursement    
        if verbose >=2:
            print(f"found duplicated data: \t{i}\t amount: {r['amount']} ")
        reimbursement = data[(data['datetime']==r['datetime']) & (data['amount']==-r['amount'])]
        if len(reimbursement)==1:
            reimbursement = reimbursement.iloc[0]
            if verbose >=2:
                print(f"\t => reimbursed \t{reimbursement.name}\t amount: {reimbursement['amount']} ")
            indecies_to_drop.append(i)
            indecies_to_drop.append(reimbursement.name)
    # print(indecies_to_drop)
    # print(data.loc[indecies_to_drop])
    assert data.loc[indecies_to_drop]['amount'].sum() == 0, f"sum ({data.loc[indecies_to_drop]['amount'].sum()}) of the reimbursed values should be zero!"
    if verbose >=2:
        print(f'dropping {len(indecies_to_drop)} reimbursement values')
    return data.drop(indecies_to_drop)
```

```python
def get_paypal_index_for_single_row(row, data_paypal, max_day_diff = 7, verbose=1):
    """
    row: datarow for which to find the paypal index, e.g. entry from dkb
    data_paypal: paypal data in which to look for the entry
    max_day_diff: max difference in days that the timestamp of the row may differ from the entry in the paypal data
    
    """

    if  ('PAYPAL' not in row['desc'].upper()):
        return np.nan, None
    paypal = data_paypal[data_paypal['amount'].abs() == abs(row['amount'])]

    if len(paypal) == 0:
        # if it is a deposit, then we have to invert the amount
        if verbose>0:
            print('suspect deposit looking for inverted amount')
        paypal = data_paypal[data_paypal['amount'] == -row['amount']]
        if len(paypal) == 0:

            if verbose>0:
                print(f"{row['date']} did not find entry with amount |{row['amount']}|")
            return np.nan, None

    record, time_diff =  get_closest_record(paypal, time_index = row['date'].tz_localize(timezone.utc))

    if abs(time_diff.days) > max_day_diff:
        if verbose>0:
            print(f"WARNING: large timediff {time_diff.days} for {len(paypal['datetime'])} records on {str(row['date']).split(' ')[0]} amout {row['amount']}")
            print('***** paypal record ******')
            print(record)
            print('***** bank row ******')
            print(row['date'], row['amount'])
            print(row['desc'])
            print('\n')
            return np.nan, None

    if verbose>2:
        print(f'\n entry (amount {row["amount"]}): {row["desc"]}')
        print('record:', record)
    return record.name, float(time_diff.days)


def get_paypal_index(data, data_paypal, max_day_diff = 7, verbose=1):
    """
    data: data for which to find the paypal index, e.g. entry from dkb
    data_paypal: paypal data in which to look for the entry
    max_day_diff: max difference in days that the timestamp of the row may differ from the entry in the paypal data
    
    returns list of tuples, first is the index in the data, second the  paypal index
    
    """
    indecies = []
    for i, row in data[data['desc'].str.lower().str.contains('paypal')].iterrows():
        index, days = get_paypal_index_for_single_row(row, data_paypal, max_day_diff = max_day_diff, verbose=verbose)
        indecies.append((i,index))
    return indecies
```

```python

def get_closest_record(records, time_index):
    """
    returns the record, where the datetime index is closest to time_index
    and the min time
    
    """
    min_time = (records['datetime'] - time_index).abs().min()
    
    closest_record = records[(records['datetime'] - time_index).abs() == min_time]
    return closest_record.iloc[0], min_time
        
```

### final

```python
input_data_path =  Path('/Users/jangie/Documents/Finanzen/Detailed_Data')
categories = {}
with open('../data/categories.txt') as f:
    for i, line in enumerate(f.readlines()):
        categories[i] = line.strip()
        
categories
```

```python
df_dkb = None
drop_duplicates = False

for f in (input_data_path/'dkb').glob('*.csv'):
    assert f.exists()
    # df_dkb = pd.read_csv(str(dkb_data_file))
    df_dkb_in = read_dkb_csv(f, verbose=3, drop_duplicates=drop_duplicates)
    if df_dkb is None:
        df_dkb = df_dkb_in
    else: 
        df_dkb = pd.concat([df_dkb, df_dkb_in], ignore_index=True)
        if drop_duplicates:
            print(f"dropping duplicates considering only ['date', 'amount', 'desc', 'account_nr'] (length {len(df_dkb)})")
            df_dkb.drop_duplicates(subset=['date', 'amount', 'desc', 'account_nr'], inplace=True)
    print(f"=> total length {len(df_dkb)}")
    
df_dkb['target account'] = np.nan
```

```python
# df_dkb
```

#### filter out the transaction to other accounts
this is just moving money between accounts and are not real expenses or income. Later we will try to find the corresponding entires to check the consistency. But this is actually not so trivial.

```python
# filter out the paypal entries
paypal_indecies = df_dkb['desc'].str.upper().str.contains('PAYPAL \(EUROPE\) S.A.R.L')
df_dkb.loc[paypal_indecies, 'target account'] = 'paypal'


# filter out the paypal entries
dkb_indecies = df_dkb['desc'].str.contains('Überweisung HERR ')
df_dkb.loc[dkb_indecies, 'target account'] = 'dkb'

credit_indecies = df_dkb['desc'].str.contains('Umbuchung KREDITKARTEN GELDANLAGE')
df_dkb.loc[credit_indecies, 'target account'] = 'credit card'



# df = df_dkb[~paypal_indecies]
```

```python
df_dkb['account_nr'].unique()
```

```python
df_dkb.groupby(['target account', 'account_nr'])['amount'].sum()
```

```python
df_dkb['target account'].unique()
```

```python
df_dkb[df_dkb['target account'].isna()]
```

```python
!ls ..
```

```python
df_dkb
```

```python
# df_dkb.to_csv('dkb.csv')
```

#### label by the most obvious keywords - obsolete, now use the regex function in the helper module

```python
categories
```

```python
label_dict = {
    'Supermarket & Everyday commodities': ['Kartenzahlung/-abrechnung REWE', 'Kartenzahlung/-abrechnung EDEKA', 'Kartenzahlung/-abrechnung DIRK ROSSMANN', 'Kartenzahlung/-abrechnung DANKE, IHR LIDL', 'Kartenzahlung/-abrechnung DM', 'Kartenzahlung/-abrechnung MUELLER SAGT DANKE'],
    'Bill - Communications': ['Lastschrift Vodafone Deutschland GmbH'],
    'Bill - Utilities, Rent & Fees': ['Lastschrift Aberdeen Standard Investments'],
    'Income - Salary & Benefits': ['Lohn, Gehalt, Rente IAV GmbH', 'Lohn, Gehalt, Rente Hertie School', 'Gutschrift SavingBuddies e. V. Honorar'],
    'Income - Reimbursement': ['Gutschrift IAV GmbH'],
    'Public Transport': ['SWAPFIETS']
}

for label_desc, desc_list in label_dict.items():
    # label = {v:k for k, v in categories.items()}[label_desc]
    
    label_index = pd.DataFrame([df_dkb['desc'].str.contains(desc) for desc in desc_list]).T.any(axis=1)
    
    
    df_dkb.loc[label_index, 'class'] = label_desc # label
# df_dkb[df_dkb['desc'].str.contains('ROSSMANN')]
```

```python
df_dkb[df_dkb['class'].isna() & df_dkb['target account'].isna()]
```

```python
df_dkb[df_dkb['desc'].str.contains('BVG')]
```

#### save prelabeled data

```python
df_dkb.to_csv('../data/dkb.csv')
```

```python
df_dkb.loc[~paypal_indecies, 'amount'].sum()
```

```python
'Überweisung HERR DR. JAN GIESELER'
```

```python
df_dkb[df_dkb['amount']< -2000]
```

## Playing around


### explore data


#### load the data

```python
input_data_path =  Path('/Users/jangie/Documents/Finanzen/Detailed_Data')
account = '1002423109'
```

```python
agg_data_file = Path('../all_data_dkb_1002423109.csv')

assert agg_data_file.exists()
df = pd.read_csv(str(agg_data_file))
```

```python
pp_raw

```

```python
dkb_data_file = input_data_path/(f'dkb/2022_{account}.csv')

assert dkb_data_file.exists()
# df_dkb = pd.read_csv(str(dkb_data_file))
df_dkb = read_dkb_csv(dkb_data_file, verbose=2, drop_duplicates=True)
df_dkb
```

```python
# pd.to_datetime
# min_t = pd.to_datetime(df_dkb['date'].min())
# max_t = pd.to_datetime(df_dkb['date'].max())
```

```python

# min_t = datetime.strftime(df_dkb['date'].min().to_pydatetime(), "%m/%d/%Y, %H:%M:%S")
# max_t = datetime.strftime(df_dkb['date'].max().to_pydatetime(), "%m/%d/%Y, %H:%M:%S")
```

```python
pp_data_file = input_data_path/(f'Paypal/Paypal_01 Jan 2019 - 04 Aug 2022.csv')
assert pp_data_file.exists()
# df_dkb = pd.read_csv(str(dkb_data_file))
df_pp = read_paypal_csv(pp_data_file, verbose=2)
df_pp['Type'].value_counts()
```

```python

min_t = datetime.strftime(df_dkb['date'].min().to_pydatetime(), "%m/%d/%Y, %H:%M:%S")
max_t = datetime.strftime(df_dkb['date'].max().to_pydatetime(), "%m/%d/%Y, %H:%M:%S")
min_t, max_t
```

```python
df_pp = df_pp[df_pp['datetime'].between(min_t,max_t, inclusive='both')]
len(df_pp), df_pp['datetime'].min(), df_pp['datetime'].max()
```

```python
df_pp
```

```python
for s in df_pp['Status'].unique():
    print(s, len(df_pp['Status'][df_pp['Status']==s]))
```

```python
for s in df_pp['Currency'].unique():
    print(s, len(df_pp['Currency'][df_pp['Currency']==s]))
```

```python
for s in ['BRL', 'AUD']:
    print(s)
    print(df_pp[['datetime', 'amount', 'description']][df_pp['Currency']==s])
```

```python
unique_datetimes = df_pp[df_pp['Currency']!='EUR']['datetime'].unique()
unique_datetimes[0]
```

```python
df_pp[df_pp['Currency']!='EUR']
```

```python
''.join(df_pp[df_pp['datetime']==unique_datetimes[1]]['description'].unique())
```

```python
def merge_foreign_transactions(df, verbose=1):
    """
    the description for foreign transactions is in the row with foreign currency
    here we find all the rows correcponding to the foreign transaction and fill in the description for the transaction in EUR
    
    """
    
    # the transactions are all recored at the same time, this is how we know which records belong together
    unique_datetimes = df[df['Currency']!='EUR']['datetime'].unique()
    if verbose>1:
        print(f'found {len(unique_datetimes)} foreign transactions')
        
    for unique_datetime in unique_datetimes:
        df_transaction = df[df['datetime']==unique_datetime]
        print(len(df_transaction))
        assert len(df_transaction[df_transaction['Currency']=='EUR']) == 1
        assert len(df_transaction[df_transaction['Currency']!='EUR']) == 2
        index = df_transaction[df_transaction['Currency']=='EUR'].index[0]
        print('index', index)
        # print(df_transaction[df_transaction['Currency']=='EUR'])[['Currency']]
        # replace the description in the EUR entry with the combined descriptions of all the entries
        ''.join(df_transaction['description'].unique())
        
    
    
    
merge_foreign_transactions(df_pp, verbose=2)
    
```

```python
df_pp[df_pp['datetime'] == df_pp['datetime'][df_pp['Currency']==s][0]]
```

```python
df_pp[df_pp['Status'] == 'Pending'][['amount', 'Balance', 'description']]
```

```python
df_pp[df_pp['Status'] == 'Denied']
```

```python
df_pp[df_pp['Balance Impact'] == 'Credit']
```

```python
# df_dkb[df_dkb['amount']==-7.9]
```

```python

```

```python
df_pp
```

```python
df_pp['datetime'].min(), df_pp['datetime'].max()
```

```python
df_dkb['date'].min(), df_dkb['date'].max()
```

```python
df['date'].min(), df['date'].max()
```

```python
df_dkb[df_dkb['desc'].str.lower().str.contains('Lastschrift PAYPAL'.lower())]
```

```python
df_dkb[df_dkb['desc'].str.lower().str.contains('DKB'.lower())]
```

#### find the paypal entries in the dkb data

```python
df_dkb
```

```python
paypal_index = get_paypal_index(df_dkb, df_pp, max_day_diff = 7, verbose=3)
paypal_index
```

```python
df_dkb
```

```python
df_pp.iloc[35]
```

```python
df_dkb.loc[[35]]
```

```python
df_dkb.iloc[35]['target_account']= 'xx'
```

```python
df_pp['Type'].value_counts()
```

## paypal explore

```python
paypal_file = '../data_extern/Paypal/Paypal_Activity_01 Jan 2019 - 17 Sep 2022.CSV'
```

```python
pp_raw = pd.read_csv(paypal_file, decimal=",", dtype="string")
pp_raw['Gross'] = pp_raw['Gross'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
pp_raw['Type'].value_counts()
```

```python
pp_raw[pp_raw['Gross'] == -0.01]
```

```python
pp_raw[['Type', 'Gross']].groupby('Type').agg('sum')
```

### explore with facets

```python

# Display the Dive visualization for the training data.
# from IPython.core.display import display, HTML
from IPython.display import display
from IPython.core.display import HTML

jsonstr = pp_raw.to_json(orient='records')
HTML_TEMPLATE = """
        
        
        
        """
html = HTML_TEMPLATE.format(jsonstr=jsonstr)
display(HTML(html))
```

## load all the data from several files

```python
[f for f in (input_data_path/'dkb').glob('*.csv')]
```

```python
# ?pd.concat
```

```python
df_dkb = None

for f in (input_data_path/'dkb').glob('*.csv'):
    assert f.exists()
    # df_dkb = pd.read_csv(str(dkb_data_file))
    df_dkb_in = read_dkb_csv(f, verbose=2, drop_duplicates=True)
    if df_dkb is None:
        df_dkb = df_dkb_in
    else: 
        df_dkb = pd.concat([df_dkb, df_dkb_in], ignore_index=True)
        print(f"dropping duplicates considering only ['date', 'amount', 'desc', 'account_nr']")
        df_dkb.drop_duplicates(subset=['date', 'amount', 'desc', 'account_nr'], inplace=True)
```

```python
input_data_path
```

```python
[f for f in (input_data_path/'dkb').glob('*.csv')]
```

```python
pp_data_file = input_data_path/(f'Paypal/Paypal_01 Jan 2019 - 29 Dec 2022.csv')
assert pp_data_file.exists()
# df_dkb = pd.read_csv(str(dkb_data_file))
df_pp = read_paypal_csv(pp_data_file, verbose=2)
df_pp['Type'].value_counts()
```

```python
# min_t = datetime.strftime(df_dkb['date'].min().to_pydatetime(), "%m/%d/%Y, %H:%M:%S")
# max_t = datetime.strftime(df_dkb['date'].max().to_pydatetime(), "%m/%d/%Y, %H:%M:%S")
min_t = datetime.strftime(df_dkb['date'].min().to_pydatetime(), "%d.%b.%Y")
max_t = datetime.strftime(df_dkb['date'].max().to_pydatetime(), "%d.%b.%Y")

print(f"dkb data exits from {min_t} to {max_t}")
```

```python
min_t = datetime.strftime(df_pp['datetime'].min().to_pydatetime(), "%d.%b.%Y")
max_t = datetime.strftime(df_pp['datetime'].max().to_pydatetime(), "%d.%b.%Y")
print(f"paypal data exits from {min_t} to {max_t}")
```

#### find the paypal entries

```python
def get_paypal_index_for_single_row(row, data_paypal, max_day_diff = 7, verbose=1):
    """
    row: datarow for which to find the paypal index, e.g. entry from dkb
    data_paypal: paypal data in which to look for the entry
    max_day_diff: max difference in days that the timestamp of the row may differ from the entry in the paypal data
    
    """

    # if  ('PAYPAL' not in row['desc'].upper()):
    #     return np.nan, None
    assert ('PAYPAL' in row['desc'].upper())
    
    # look for entries that have the same amount with opposite sign and have not been asigned yet
    paypal = data_paypal[(data_paypal['amount'] == -row['amount']) & (data_paypal['target'].isna())]
    # paypal = data_paypal[data_paypal['amount'] == -row['amount']]
    
    # data_paypal[data_paypal['Type'].str.strip().str.lower() == 'bank deposit to pp account']
    

    if len(paypal) == 0:
        if verbose>0:
            print(f"{row['date']} did not find entry with amount {row['amount']}")
        return np.nan, None

    record, time_diff =  get_closest_record(paypal, time_index = row['date'].tz_localize(timezone.utc))

    if abs(time_diff.days) > max_day_diff:
        if verbose>0:
            print(f"WARNING: large timediff {time_diff.days} for {len(paypal['datetime'])} records on {str(row['date']).split(' ')[0]} amount {row['amount']}")
            if verbose>1:
                print('***** paypal record ******')
                print(record)
                print('***** bank row ******')
                print(row['date'], row['amount'])
                print(row['desc'])
                print('\n')
        # return np.nan, None

    if verbose>2:
        print(f'\n entry (amount {row["amount"]}): {row["desc"]}')
        print('record:', record)
        
    data_paypal.loc[record.name, 'target'] = row.name
    return record.name, float(time_diff.days)


def get_paypal_index(data, data_paypal, max_day_diff = 7, verbose=1):
    """
    data: data for which to find the paypal index, e.g. entry from dkb
    data_paypal: paypal data in which to look for the entry
    max_day_diff: max difference in days that the timestamp of the row may differ from the entry in the paypal data
    
    returns list of tuples, first is the index in the data, second the  paypal index
    
    """
    indecies = []
    for i, row in data[data['desc'].str.lower().str.contains('paypal')].iterrows():
        index, days = get_paypal_index_for_single_row(row, data_paypal, max_day_diff = max_day_diff, verbose=verbose)
        indecies.append((i,index))
    return indecies


# get_paypal_index(df_dkb, df_pp, max_day_diff = 7, verbose=0)
# data = df_dkb
# max_day_diff = 7

data_paypal = df_pp
data_paypal['target']=np.nan
# verbose=1
c = 0
indecies = []
for i, row in data[data['desc'].str.lower().str.contains('lastschrift paypal')].iterrows():
    index, days = get_paypal_index_for_single_row(row, data_paypal, max_day_diff = max_day_diff, verbose=verbose)
    indecies.append((i,index))
#     c = c+1
#     if c>4:
        
#         break
    
# indecies
```

```python
data_paypal[~data_paypal['target'].isna()]
```

```python
data_paypal
```

```python
data_paypal.loc[indecies[0][1]]['target']
```

```python
df_dkb[df_dkb['desc'].str.upper().str.contains('PAYPAL')]
```

```python
data_paypal[~data_paypal['target'].isna()].count()
```

```python
data_paypal[~data_paypal['target'].isna()]
```

```python
data_paypal[~data_paypal['target'].isna()]['amount'].count()
```

```python
data_paypal[data_paypal['target'].isna()]['amount'].sum()
```

```python
data_paypal
```

```python
df_pp['target'] = np.nan
# df_pp.replace('<NA>', np.nan)

```

```python

```

```python
type(row)
```

```python
row.name
```

```python
print('paypal data - number of deposits to pp account:', len(df_pp[df_pp['Type'].str.strip().str.lower() == 'bank deposit to pp account']))

print('dkb data - transfers to pp account:', len(df_dkb[df_dkb['desc'].str.lower().str.contains('paypal')]))
```

```python
paypal_index = get_paypal_index(df_dkb, df_pp, max_day_diff = 100, verbose=1)
```

```python
counts = pd.Series(pp_index).value_counts(sort=False)
s_ambivalent = counts[counts>1]
print(f"found {len(s_ambivalent)} enties with ambivalent paypal entries")
s_ambivalent
```

```python
df_pp.loc[s_ambivalent.index[0]]
```

```python
pp_index  = [p for d, p in paypal_index]

s_unique = set(pp_index)
s_ambivalent = 

print(f"found {len(s_unique)} unique entries")
```

```python
.count('')
```

```python
i_dkb, i_pp = indecies[0]

print_infos_matched_entries(df_dkb, df_pp, *indecies[0])
```

```python
# df_pp.loc[i_pp]
```

```python

```

```python

```

```python

```

```python

```

```python
df_dkb[df_dkb['amount'] == -15]
```

```python
df_pp[df_pp['amount']== 15]
```

```python
i=4
def print_infos_matched_entries(df_dkb, df_pp, i_dkb, i_pp):
    assert df_dkb.loc[i_dkb]['amount'] == -df_pp.loc[i_pp]['amount']
    print(f"dkb ({df_dkb.loc[i_dkb]['date']}):\n\t{df_dkb.loc[i_dkb]['desc']}")
    print(f"paypal ({df_pp.loc[i_pp]['datetime']}):\n\t{df_pp.loc[i_pp]['description']}\n\t{df_pp.loc[i_pp]['Type']}")
# print_infos_matched_entries(df_dkb, df_pp, *paypal_index[i])
```

```python
for i, row in df_dkb.iterrows():
    'PAYPAL' in row['desc'].upper() 
```

```python
paypal_index[59]
```

### closer look at the entries that could not be found in paypal data

difference is due to different amounts in the dkb and paypal data, most likely because there is some credit in the paypal account left

```python
[ i for i in paypal_index if isinstance(i[1], float) ]
```

```python
i_dkb = 132
print(f"dkb ({df_dkb.loc[i_dkb]['date']}, {df_dkb.loc[i_dkb]['amount']}):\n\t{df_dkb.loc[i_dkb]['desc']}")
```

```python
df_dkb.loc[1289]
```

```python
for i, row in df_pp.iterrows():
    if ('IKEA' in row['description'].upper()):
        print(f"{row['datetime']}, {row['amount']}, {row['description']}")
        r2=row
```

```python
for i, row in pp_raw.fillna('').iterrows():
    if ('IKEA' in row['Item Title']):
        print(i, f"{row['Date']}, {row['Gross']}, {row['Item Title']}")
        r2=row
```

```python
pp_raw.loc[729]
```

```python
pp_raw.loc[730]
```

```python
pp_raw.loc[100]
```

```python
df_pp.loc[paypal_index[i][1]]
```

```python
df_dkb
```

```python

```

### expore and clean dkb data

```python
df_dkb['']
```

```python

```
