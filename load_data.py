import logging
import pandas as pd
import numpy as np
import re
from pathlib import Path


from classify_helper  import merge_paypal_enties

def read_dkb_csv(filename, drop_duplicates=False)-> pd.DataFrame:
    """Read a file in the CSV format that dkb provides downloads in.

    Returns a pd.DataFrame with columns of 'datetime', 'desc', and 'amount'."""
    account_nr = re.search('\d{10}', Path(filename).name)[0]
    data=pd.read_csv(filename,sep=';', skiprows=9, encoding='latin-1',
                    usecols=[0,2,3,4,7,9,10,11],
                    names=['Buchungstag', 'Wertstellung', 'Buchungstext','Auftraggeber / Beguenstigter', 'Verwendungszweck', 'Kontonummer',
                        'BLZ', 'Betrag (EUR)', 'Glaeubiger-ID', 'Mandatsreferenz','Kundenreferenz', 'Unnamed'] )

    if drop_duplicates:
        duplicated_data = data[data.duplicated()]
        logging.debug(f"dropping {len(duplicated_data)} duplicated entries")
        
        # if verbose > 2:
        #     if len(duplicated_data)>0:
        #         print(duplicated_data)
        #         print('\n')
        data.drop_duplicates(inplace=True)
        
    
    data = data[data['Verwendungszweck'] != 'Tagessaldo'] # drop all the Tagessaldo entries
    
    logging.info(f"number of data loaded: {len(data)}")
    
    data['Buchungstag'] = pd.to_datetime(data['Buchungstag'], format = "%d.%m.%Y", utc=True)
    # data['Wertstellung'] = pd.to_datetime(data['Wertstellung'], format = "%d.%m.%Y")
    data['account_nr'] = account_nr

    logging.debug(f"new data from {filename} with {len(data)} entries, years {data['Buchungstag'].dt.year.unique()}")


    data.fillna("", inplace=True)

    data['datetime'] = data['Buchungstag'] #.dt.strftime('%d/%m/%Y')
    data['amount'] = data['Betrag (EUR)'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    data.drop(data[data['amount']==''].index, inplace=True)
    data['amount'] = data['amount'].astype(float)

    data = data[data['amount'].astype(float)!=0] # drop all zero values

    desc_column_names = ['Buchungstext', 'Auftraggeber / Beguenstigter', 'Verwendungszweck']
    data['desc'] = data[desc_column_names].agg(' '.join, axis=1)
    df = data[['datetime', 'amount', 'desc', 'account_nr','Kundenreferenz', 'Mandatsreferenz']]

    df.index = pd.Index(range(len(df)))

    return df


def read_paypal_csv(filename)-> pd.DataFrame:
    """Read a file in the CSV format that paypal provides downloads in.

    Returns a pd.DataFrame with columns of 'datetime', 'description', and 'amount'."""

    data = pd.read_csv(filename, decimal=",", dtype="string")

    logging.debug(f"loaded {len(data)} paypal records")
        
    data = data[data['Balance Impact']!='Memo']
    logging.debug(f"{len(data)} paypal records after dropping `Balance Impact` = `Memo` entries")

    # drop all columns that are completely empty
    data.dropna(axis=1, how='all', inplace=True)        
        
    # data = data[data['Status']!='Pending']
    # if verbose>1:
    #     print(f"{len(data)} paypal records after dropping `Status` = `Pending` entries")     

    type_list = [
        'Bank Deposit to PP Account ',
        'Payment Refund'
        ]

    #tmp comment out
    # data = data[data['Type'].str.contains('|'.join(type_list))]
    # # data = data[data['Status'] == 'Completed']
    # if verbose>1:
    #     print(f"{len(data)} status completed")


    data['TimeZone'].replace('CEST', '+02:00', inplace=True)
    data['TimeZone'].replace('CET', '+01:00', inplace=True)

    data.insert(0, 'datetime', pd.to_datetime(data['Date'] + " " + data['Time']+ data['TimeZone'], format = "%d/%m/%Y %H:%M:%S%z", utc=True))
    # data.drop(['Time', 'Date', 'TimeZone', 'Status'], axis=1, inplace=True) #tmp comment out


    # to make the transaction ID unique we add the datetime
    # data['Transaction ID'] = data['Transaction ID'] + "-" +data['datetime'].dt.strftime("%d%m%Y %H:%M:%S%z")

    # data.set_index('Transaction ID', inplace=True)

    # convert to floats
    for k in ['Balance', 'Fee', 'Net', 'Gross']:
        data[k] = data[k].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
    # data = data[data['Balance'] != 0]

    # insert new column description
    data.insert(1, 'desc', data[['Name', 'Country', 'Subject', 'Note', 'Item Title']].fillna('').agg(' '.join, axis=1))

    # data.drop(['Name', 'Country', 'Subject', 'Note'], axis=1, inplace=True) # drop the columns that are in the desciption now


    data.rename(columns={"Gross": "amount"}, inplace=True)
    data['target account'] = np.nan
    data['class'] = np.nan

    data = data[['datetime', 'amount', 'desc', 'target account', 'class', 'Type',  'From Email Address', 'To Email Address', 'Currency', 'Status', 'Balance','Transaction ID']]
    # data = data[['datetime', 'Balance Impact', 'Status', 'description', 'Type',  'amount', 'From Email Address', 'To Email Address']]


    data.loc[data['Type'].str.contains('Bank Deposit to PP Account'), 'target account'] = 'dkb'
    data.loc[data['Type'].str.contains('General Card Deposit'), 'target account'] = 'dkb'

    data.index = range(len(data))

    return data
