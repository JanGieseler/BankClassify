import logging
import yaml
from pathlib import Path

def assign_target_account(df):
    with open(Path('../data/target_account_regex.yaml')) as file:
        target_dict = yaml.full_load(file)
    
    for k, v in target_dict.items():
        
        indecies = df['desc'].str.match(k)
        if indecies.sum()>0:
            logging.debug(f'assigning {indecies.sum()} entries to {v}')
        df.loc[indecies, 'target account'] = v
    return df


def assign_label(df):

    """
    assign labels based on regular expression
    
    """
    with open(Path('../data/labeling_regex.yaml')) as file:
        regex_dict = yaml.full_load(file)
    
    for k, v in regex_dict.items():
        
        indecies = df['desc'].str.lower().str.match(k.lower())
        if indecies.sum()>0:
            logging.debug(f'labeling {indecies.sum()} entries as \'{v}\'')
        df.loc[indecies, 'class'] = v
    return df


def merge_paypal_enties(df_pp):
    """
    merge the rows in the paypal data that belong to the same transaction
    
    """
    
    value_counts = df_pp['datetime'].value_counts()

    expected_enties = value_counts.replace([3,4],2).sum() # this is the expected resulting length of the merged dataset
    
    
    for timestamp, count in value_counts.items():
        # print(timestamp, count)
        
        df_timestamp = df_pp[df_pp['datetime']==timestamp]
        # _, df_timestamp = pop_row_by_timestamp(df_pp, timestamp)
        
        assert count <= 4
        
        # if count == 1:
            # df_pp_merged.append(df_timestamp)
        if count == 2:
            # print(timestamp, df_timestamp['Type'].values)

            # assert {'Bank Deposit to PP Account ', 'Payment Refund}'.issubset in df_timestamp['Type'].values, timestamp
            # make sure that if we have 2 entries one of them is one of the following
            assert len({'Bank Deposit to PP Account ', 'Payment Refund', 'General Card Deposit', 'Reversal of General Account Hold', 'Reversal of ACH Deposit', 'Payment Reversal'}.intersection(set(df_timestamp['Type'].values))) == 1, timestamp
            
            # print(len({'Payment Refund', 'Reversal of ACH Deposit', 'Payment Reversal'}.intersection(df_timestamp['Type'].values)) == 1)
            # print('>>>')
            # remove entries which have been refunded
            if len({'Payment Refund', 'Reversal of ACH Deposit', 'Payment Reversal'}.intersection(df_timestamp['Type'].values)) == 1:
                assert df_timestamp['amount'].sum() == 0
                
                df_pp.drop(df_timestamp.index, inplace=True)
                expected_enties -=2
                
            if 'Reversal of General Account Hold' in df_timestamp['Type'].values:
                transaction_type_index = df_timestamp[df_timestamp['Type'] != 'Reversal of General Account Hold'].index

                transaction_reversal_index = df_timestamp[df_timestamp['Type'] == 'Reversal of General Account Hold'].index

                # sum up that transaction with the reversal
                df_pp.loc[transaction_type_index, 'amount'] +=  df_timestamp.loc[transaction_reversal_index, 'amount'].values[0]
                # drop the reversal transaction
                df_pp.drop(transaction_reversal_index, inplace=True)
                expected_enties -=len(transaction_reversal_index)

        elif count == 3:
            # ================================================================================
            # this treats the case with a 'Reversal of General Account Hold' transaction
            # ================================================================================
            # assert 'Bank Deposit to PP Account ' in df_timestamp['Type'].values
            # assert 'Reversal of General Account Hold' in df_timestamp['Type'].values
            transaction_types = {'Bank Deposit to PP Account ', 'Reversal of General Account Hold'}
            assert transaction_types.issubset(set(df_timestamp['Type'].values))

            # the transaction type is the one that is not in the set `transaction_types`
            transaction_type = list(set(df_timestamp['Type']).difference(transaction_types))[0]
            transaction_type_index = df_timestamp[df_timestamp['Type'] == transaction_type].index

            transaction_reversal_index = df_timestamp[df_timestamp['Type']=='Reversal of General Account Hold'].index

            # sum up that transaction with the reversal
            df_pp.loc[transaction_type_index, 'amount'] +=  df_timestamp.loc[transaction_reversal_index, 'amount'].values[0]
            # drop the reversal transaction
            df_pp.drop(transaction_reversal_index, inplace=True)
        elif count == 4:
            # ================================================================================
            # this treats the case with a 'General Currency Conversion' transaction
            # ================================================================================
            transaction_types = {'Bank Deposit to PP Account ', 'General Currency Conversion'}
            assert transaction_types.issubset(set(df_timestamp['Type'].values))
            # the transaction type is the one that is not in the set `transaction_types`
            transaction_type = list(set(df_timestamp['Type']).difference(transaction_types))[0]
            transaction_type_index = df_timestamp[df_timestamp['Type'] == transaction_type].index


            currency_conversion_index =  df_timestamp[(df_timestamp['Type']=='General Currency Conversion') & (df_timestamp['Currency']=='EUR')].index
            # print('currency_conversion_index', currency_conversion_index)
            # print('currency_conversion_index', currency_conversion_index)
            
            df_pp.loc[transaction_type_index, ['amount', 'Currency']] = df_timestamp.loc[currency_conversion_index, ['amount', 'Currency']].values[0]
            # not we drop both the currency conversion entries (the one in foreig currency and the one in eur)
            currency_conversion_index = df_timestamp[df_timestamp['Type']=='General Currency Conversion'].index
            df_pp.drop(currency_conversion_index, inplace=True)
        # df_pp_merged.append(df_timestamp)
        
    # df_pp_merged = pd.concat(df_pp_merged)
    
    assert expected_enties == len(df_pp), f"expected {expected_enties}, got {len(df_pp)}"



def check_balance(df_pp):
    """
    check wether the payment of the transaction is matched by the depoit to the account,
    taking into account a potential balance that was still in the paypal account

    if test doesnt pass an assertion error is raised
    """
    for timestamp in df_pp[
        (df_pp['Type'] == 'Bank Deposit to PP Account ') | (df_pp['Type'] == 'General Card Deposit')]['datetime']:
        indecies = df_pp[df_pp['datetime'] == timestamp].index
        # the deposit and the transaction should sum to zero
        if df_pp.loc[indecies, 'amount'].sum() != 0:
            # unless there is a remaining balance on the paypal account
            diff = df_pp.loc[indecies, 'amount'].sum() + df_pp.loc[indecies[0]-1, 'Balance']
            assert diff < 1e-12, diff
    print('check passed!!')