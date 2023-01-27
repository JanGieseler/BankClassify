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

todo:
- [ ] fix the regex assignment, somehow this doesn't work right
    

```python

```

```python
import sys, os


import logging
import pandas as pd
import numpy as np
from pathlib import Path
# sys.path.insert(0,'..')
project_dir = str(Path(os.getcwd()).parent)
if not project_dir in sys.path:
    sys.path.insert(0,project_dir)
sys.path
from BankClassify import BankClassify, assign_target_account, assign_label
from load_data import read_dkb_csv, read_paypal_csv
from classify_helper import merge_paypal_enties, check_balance
import yaml
```

```python
%load_ext autoreload

%autoreload 2
```

```python
logging.basicConfig(level=logging.DEBUG)
```

```python
input_data_path =  Path('/Users/jangie/Documents/Finanzen/Detailed_Data')
```

```python
# bc.data['dkb_3109'].index = len(bc.data['dkb_3109'])
```

```python
bc = BankClassify(Path('../data'))
```

<!-- #region tags=[] -->
#### label new data based on identical description in labeled data
<!-- #endregion -->

```python
bc.data['dkb_4217']['class'] = np.nan
bc.data['dkb_4217']['target account'] = np.nan
```

```python
labeled_dataset_name = 'dkb_4217_old'
unlabeled_dataset_name = 'dkb_4217'
```

```python
## consistency check that all identical descriptions give identical classes

for i, r in bc.data[labeled_dataset_name].iterrows():

    index = bc.data[labeled_dataset_name][bc.data[labeled_dataset_name]['desc'] == r['desc']].index
    if len(index)>1:
        assert len(bc.data[labeled_dataset_name].loc[index, 'target account'].unique()) ==1
        assert len(bc.data[labeled_dataset_name].loc[index, 'class'].unique()) ==1
        # print(index)
```

```python
for i, r in bc.data[labeled_dataset_name].iterrows():

    index = bc.data[unlabeled_dataset_name][bc.data[unlabeled_dataset_name]['desc'] == r['desc']].index
    # bc.data[unlabeled_dataset_name].loc[index, ['target account', 'class']] = r[['target account', 'class']]
    
    bc.data[unlabeled_dataset_name].loc[index, 'target account'] = r['target account']
    bc.data[unlabeled_dataset_name].loc[index, 'class'] = r['class']

```

```python
# bc.data[unlabeled_dataset_name].loc[index, ['target account', 'class']]
```

```python
bc.data[labeled_dataset_name].count()
```

```python
bc.data[unlabeled_dataset_name].count()
```

```python
# check that the two are identical
bc.data['dkb_3109_old'][~(bc.data['dkb_3109'].fillna('na').eq(bc.data['dkb_3109_old'].fillna('na'))['target account'])]
```

```python
# check that the two are identical
bc.data['dkb_3109_old'][~(bc.data['dkb_3109'].fillna('na').eq(bc.data['dkb_3109_old'].fillna('na'))['class'])]
```

```python
for f in Path('../data').glob('*.csv'):
    print(f.with_suffix('').name)
```

```python
# bc.data
```

```python
sorted([f for f in (input_data_path/'dkb').glob('*.csv')])
```

### load new raw data and add to existing dataset

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
#### paypal data
<!-- #endregion -->

```python
filename = Path('/Users/jangie/Documents/Finanzen/Detailed_Data/Paypal/Paypal_01 Jan 2019 - 29 Dec 2022.CSV')
df_pp = read_paypal_csv(filename)
print(f"paypal datasize is {len(df_pp)}")
merge_paypal_enties(df_pp)
print(f"after merge reduced to {len(df_pp)}")
check_balance(df_pp)

# consitency check, merge_paypal_enties, should eliminate all the foreig currency entries
assert (df_pp['Currency'].unique() == ['EUR']).all()

# enties with identical datetime, there should be one that is the transfer to the dkb bank
# if we eliminate these, the remaining datetimes should be unique
assert (df_pp[df_pp['target account'].isna()]['datetime'].value_counts().unique() == [1]).all()
```

```python
# df_pp[['datetime', 'amount', 'desc', 'target account', 'class']]
```

```python
bc.add_data(df_pp, "paypal")
```

```python
# bc.data.keys()
```

```python
# pd.concat(bc.data.values(), keys=bc.data.keys())[['datetime', 'amount', 'desc']]
```

```python
# bc.data['paypal']
```

```python
bc.data_unlabeled
```

<!-- #region tags=[] -->
#### dkb data
<!-- #endregion -->

```python
filename = Path('/Users/jangie/Documents/Finanzen/Detailed_Data/dkb/2023_1071944217.csv')
```

```python
df_new = read_dkb_csv(filename)
len(df_new)
```

```python
# df_new.loc[218:230]
```

```python
for k, v in bc.data.items():
    print(f"{k}: {len(v)} entires")
f"adding to dkb_{filename.with_suffix('').name[-4:]}"
```

```python
bc.add_data(df_new, f"dkb_{filename.with_suffix('').name[-4:]}")
```

```python
# df_new[df_new.duplicated()]
```

```python
bc.save_data()
```

```python
account_name = 'dkb_3109'
account_name = 'dkb_4217'
```

```python
bc.data[account_name]
```

```python
# all_df = pd.merge(df_new, bc.data[account_name][['datetime', 'amount', 'desc']], on=['datetime', 'amount', 'desc'], how='left', indicator='exists')
# # consistency check that the 
# assert len(all_df) == len(df_new)

# all_df['exists'] = np.where(all_df.exists == 'both', True, False)
# print(f"loaded dataset with {len(df_new)} entries, {all_df['exists'].sum()} are old, {(~all_df['exists']).sum()} are new")
# len(df_new[~all_df['exists']])


```

```python
all_df[~all_df['exists']]
```

```python
bc.data_unlabeled
```

<!-- #region tags=[] -->
### assign target account
<!-- #endregion -->

```python
df = bc.data_unlabeled
```

```python
# df[df['desc'].str.match(r'^Gutschrift J.*und C.*')]
```

```python
# bc.data_unlabeled[bc.data_unlabeled['amount']> 2000]
```

```python
# df_update.loc[('dkb_4217', 950)]
```

```python
len(bc.data_unlabeled)
```

```python
regex = r'Lastschrift PAYPAL .*'

indecies = bc.data_unlabeled['desc'].str.lower().str.match(regex.lower())
indecies.sum()
```

```python
indecies[indecies]
```

```python
bc.data_unlabeled[indecies]
```

```python
bc.data_unlabeled
```

```python
bc.data_unlabeled.loc[('dkb_4217',820)]
```

```python
df_update = assign_target_account(bc.data_unlabeled)
df_update
```

```python
len(df_update[~df_update['target account'].isna()])
```

```python
df_update[~df_update['target account'].isna()]
```

```python
# df_update.loc['dkb_3109']
```

```python
bc.data.keys()
```

```python
bc.update_data(df_update)
```

```python
len(bc.data_unlabeled)
```

```python
bc.save_data()
```

### label that data with regex

```python
bc.data_unlabeled['desc'].value_counts()
```

```python
# k = r'^REWE Markt GmbH Germany .* Bestellung .*'
# bc.data_unlabeled[bc.data_unlabeled['desc'].str.lower().str.match(k.lower())]
```

```python
# bc.data_unlabeled[bc.data_unlabeled['desc'].str.contains('REWE')]
```

```python
# bc.data_unlabeled
```

```python
# df_update.loc['paypal', 12]['desc']
```

```python
df_update = assign_label(bc.data_unlabeled)
print(len(df_update))
df_update
```

```python
df_update[~df_update['class'].isna()]
```

```python
bc.update_data(df_update)
```

```python
bc.save_data()
```

```python
bc.data_unlabeled
```

```python
bc.data_unlabeled[bc.data_unlabeled['amount']<0]['amount'].sum()
```

```python
bc.data_unlabeled['amount'].sum()
```

```python
bc.data_unlabeled.sort_values('amount')
```

### define the regex expressions and save to disk

```python
regex_dict = {
    r'^Lastschrift Vodafone .*': 'Bill - Communications',
    r'^Vodafone GmbH .*': 'Bill - Communications',
    r'^Lohn, Gehalt, Rente .*': 'Income - Salary & Benefits',
    r'^Lastschrift European Bank for Financial Services .*':'Investments',
    r'^Lastschrift Sauren SICAV SPARPLAN .*':'Investments',
    r'^Überweisung GISELA SILVA .*':'Education',
    r'^Dauerauftrag GIRASOLES-FOERDERVEREIN .*':'Education',
    r'^Lastschrift GIRASOLES - SONNENBLUMEN Beitrag .*':'Education',
    r'^buecher.de GmbH & Co. KG*': 'Education',
    r'^Lastschrift VATTENFALL EUROPE .*': 'Bill - Utilities, Rent & Fees',
    r'^Lastschrift Aberdeen Standard Investments .*':'Bill - Utilities, Rent & Fees',
    r'^Kartenzahlung/-abrechnung BARES GbR .*':'Restaurants & Bars',
    r'^app smart GmbH .*':'Restaurants & Bars',
    r'^Takeaway.com Payments B.V.':'Restaurants & Bars',
    r'^Kartenzahlung/-abrechnung EDEKA .*': 'Supermarket & Everyday commodities',
    r'^Kartenzahlung/-abrechnung .*ROSSMANN .*':'Supermarket & Everyday commodities',
    r'^Kartenzahlung/-abrechnung MUELLER.*': 'Supermarket & Everyday commodities',
    r'^Kartenzahlung/-abrechnung ALDI SAGT DANKE .*' : 'Supermarket & Everyday commodities',
    r'^Kartenzahlung/-abrechnung DENN?S BIOMARKT .*' : 'Supermarket & Everyday commodities',
    r'^Kartenzahlung/-abrechnung REWE SAGT DANKE.*' : 'Supermarket & Everyday commodities',
    r'^REWE Markt GmbH Germany .* Bestellung bei REWE .*': 'Supermarket & Everyday commodities',
    r'^DURSTEXPRESS GmbH Germany .*': 'Supermarket & Everyday commodities',
    r'^flaschenpost SE Germany .*': 'Supermarket & Everyday commodities',
    r'.*apotheke.*': 'Personal Care & Sport',
    r'.*Spotify.*': 'Entertainment, Hobbies & Memberships',
    r'^Berliner Bäder-Betriebe Germany.*': 'Entertainment, Hobbies & Memberships',
    r'^LogPay Financial Services GmbH  BVG Tickets .*':'Public Transport',
    r'nextbike GmbH .*':'Public Transport',
    r'DB Vertrieb GmbH  Ihre Buchung bei bahn.de .*':'Public Transport',
    r'^Hansemerkur Krankenversicherung AG.*':'Bill - Insurrance',
    r'^Lastschrift .*Getsafe Digital GmbH *': 'Bill - Insurrance',
    r'^Fashion Retail SA Germany.*': 'Clothes & Shopping',
    r'^H&M Hennes & Mauritz .*': 'Clothes & Shopping',
    r'^Airbnb .*': 'Holidays - Accomodation',
    r'^Ryanair Limited .*': 'Holidays - Transport',
    r'^IKEA Deutschland GmbH & Co. KG .*': 'Home Improvement & Stationary',
}

with open(Path('../data/labeling_regex.yaml'), 'w') as file:
    documents = yaml.dump(regex_dict, file)



```

```python

```

```python
with open(Path('../data/labeling_regex.yaml')) as file:
    documents = yaml.full_load(file)
```

```python
documents
```

```python
target_dict = {
    r'.*PAYPAL (?EUROPE)? S.A.R.L .*': 'paypal', # filter out the paypal entries
    r'^Überweisung HERR .*': 'dkb',
    r'Umbuchung KREDITKARTEN GELDANLAGE': 'dkb',
    r'^Gutschrift J.*und C.*': 'dkb_4217',
    r'^Überweisung HERR DR.*AN.*': 'dkb_3109'
}
with open(Path('../data/target_account_regex.yaml'), 'w') as file:
    documents = yaml.dump(target_dict, file)
```

```python
with open(Path('../data/target_account_regex.yaml')) as file:
    documents = yaml.full_load(file)
documents
```

### manually label data with regex

```python
regex = r'.*Lastschrift PAYPAL \(EUROPE\) .*'
regex = r'Lastschrift PAYPAL .*'

# regex =     r'^Lastschrift .*Getsafe Digital GmbH *'
# label = 'Income - Salary & Benefits'
# label = 'Bill - Insurrance'
label = 'Investments'
account_name = 'dkb_4217'
# account_name = 'dkb_3109'

bc.data[account_name][bc.data[account_name]['desc'].str.lower().str.match(regex.lower())]

```

```python
index = bc.data[account_name][bc.data[account_name]['desc'].str.lower().str.match(regex.lower())].index
```

```python
bc.data[account_name].loc[index, 'class'] = label
```

```python
bc.data[account_name].loc[index, 'class']
```

```python
bc.data['dkb_4217'].loc[979]
```

```python
bc.save_data()
```

### manually label


#### label the positive amount as reimbursement

```python
i = bc.data_unlabeled[bc.data_unlabeled['amount']>0].index
```

```python
bc.data['paypal'].loc[bc.data['paypal']['amount']>0, 'class']= 'Income - Reimbursement'
```

```python
bc.data['paypal'].loc[bc.data['paypal']['amount']>0, 'class']
```

```python
data['paypal'].loc[i, ['class']] = 'Income - Reimbursement'
```

```python
bc.data_unlabeled
```

```python
bc.save_data()
```

#### label by string

```python
bc.data_unlabeled[bc.data_unlabeled['desc'].str.contains('Uber Payments BV')]
```

```python
i = bc.data_unlabeled[bc.data_unlabeled['desc'].str.contains('Uber Payments BV')].index
```

```python
i = bc.data['paypal'][bc.data['paypal']['desc'].str.contains('Uber Payments BV')].index
```

```python
bc.data['paypal'].loc[i, ['class']] = 'Public Transport'
```

```python
bc.data['paypal'].loc[i, ['class']]
```

```python
bc.save_data()
```

```python

```

### rename and merge labels

```python
old_label = 'Education'
new_label = 'Education & Books'


# old_label = 'Home Improvement & Stationary'
# new_label = 'Home Improvement, Furniture & Stationary'


```

```python
bc = BankClassify(Path('../data'))
```

```python
for k, df in bc.data.items():
    print(k)
    print(f"number of old labels: {len(df[df['class'] == old_label])}")
    print(f"number of new labels: {len(df[df['class'] == new_label])}")
    # df[df['class'] == old_label] =new_label
    df.loc[df['class'] == old_label, 'class'] = new_label
    print(f"number of new labels: {len(df[df['class'] == new_label])}")
```

```python
bc.save_data()
```

```python
len(df[df['class'] == new_label])
```

```python

```
