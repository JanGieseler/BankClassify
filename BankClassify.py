import numpy as np
import pandas as pd
from datetime import datetime, timezone
import re
from pathlib import Path
from collections import Counter
import os
import colorama

import logging

from classify_helper import assign_target_account, assign_label

from textblob.classifiers import NaiveBayesClassifier

# logging.addLevelName(5, 'DEBUG_DETAILS')
class BankClassify():

    def __init__(self, datapath=Path("../data"), prob_threshold=1):
        """
        Load in the previous data (by default from `data`) and initialise the classifier
        agg_data_file: filename where the aggregated data is stored

        
        """

        logging.debug(f'FFFFF {Path(__file__).absolute()}')

        assert prob_threshold <=1
        assert prob_threshold >=0
        self.prob_threshold = prob_threshold
        self._datapath = datapath

        
        self.data = {}

        if self._datapath.exists():
            for f in self._datapath.glob('*.csv'):
                account_name = f.with_suffix('').name
                self.data[account_name] = pd.read_csv(f , index_col=0)
                self.data[account_name]['date'] = pd.to_datetime(self.data[account_name]['date'], format = '%Y-%m-%d')
                self.data[account_name]['account_nr'] = self.data[account_name]['account_nr'].astype(int)
                logging.info(f"loaded previous {account_name} data {len(self.data[account_name])} entries")

        # data_train = self._get_training(self.prev_data)
        
        data_train = []
        for df in self.data.values():
            data_train += self._get_training(df)

        logging.info(f'train dataset size {len(data_train)}')
            
            
        categories = self._read_categories()
        # self.print_categories(categories)
        self.data_train = data_train
        self.classifier = NaiveBayesClassifier(data_train, self._extractor)

    @property
    def data_all(self):
        df = pd.concat(self.data.values(), keys=self.data.keys())
        # df = pd.concat([d for d in self.data.values()], ignore_index=True)
        return df[df['target account'].isna()]

    @property
    def data_labeled(self):
        df = self.data_all
        # return df[((df['class'] == '')  | (df['class'].isna())) & df['target account'].isna()]
        return df[~((df['class'] == '')  | (df['class'].isna()))]

    @property
    def data_unlabeled(self):
        df = self.data_all
        # return df[((df['class'] == '')  | (df['class'].isna())) & df['target account'].isna()]
        return df[((df['class'] == '')  | (df['class'].isna()))]


    def update_data(self, df_new, dataset_name=None):
        """
        update the dataset with the new data, the new dataset should contain the updated gound truth class
        we ignore the columns 'class_guess' and 'class_prob', since they are filled in by the model
        """

        # if data_set is not provided, the dataframe should have a multiindex, where the first index contains the dataset_name
        if dataset_name is None:
            assert isinstance(df_new.index, pd.MultiIndex)
            for dataset_name in df_new.index.get_level_values(0).unique():
                
                df_tmp = df_new.loc[dataset_name]
                logging.debug(f'updating dataset_name: {dataset_name}')

                self.update_data(df_tmp, dataset_name)
        else:
            assert dataset_name in self.data.keys()

            column_names = df_new.columns[~df_new.columns.isin(['class_guess', 'class_prob'])]
            print('ffff', column_names)
            assert list(column_names) == list(self.data[dataset_name].columns)
            # True: overwrite original DataFrame's values with values from `other`.
            # False: only update values that are NA in the original DataFrame.
            self.data[dataset_name].update(df_new[column_names], overwrite = True)


    def save_data(self):

        for k, v in self.data.items():
            v.to_csv(self._datapath/f'{k}.csv')

    def add_data(self, df_new, account_name):
        """Add new data and interactively classify it.
        """

        logging.debug(f"adding {account_name} data!")
        logging.debug(f"previous dataset {len(self.data[account_name])} entries")
        # logging.debug(f"\t {self.data[account_name]['class'].isna().sum()} without category")
        
        
        is_new = ~df_new['date'].isin(self.data[account_name]['date'])

        logging.debug(f"new dataset {len(df_new)} entries with {len(df_new[is_new])} new ones")
        if len(df_new[is_new]) > 0:

            # consistency check 1, all the new data should be in a single chunk
            assert np.array_equal(np.diff(df_new[is_new].index), np.ones(len(df_new[is_new])-1))
            # consistency check 2, all the old data should be in a single chunk
            assert np.array_equal(np.diff(df_new[~is_new].index), np.ones(len(df_new[~is_new])-1))


            df_new = df_new[is_new]


            self.data[account_name] = pd.concat([self.data[account_name], df_new], ignore_index=True)
            # logging.debug(f"dropping duplicates considering only ['date', 'amount', 'desc']")
            # self.prev_data.drop_duplicates(subset=['date', 'amount', 'desc'], inplace=True)
            
            # logging.debug(f"total dataset after dropping duplicates {len(self.prev_data)} entries")
            # logging.debug(f"\t {self.data[account_name]['class'].isna().sum()} without category")
                
            # self.prev_data.to_csv(self._datapath, index=False)

            # logging.debug(f"saved dataset {len(self.prev_data)} entries")
            # logging.debug(f"\t {self.prev_data['class'].replace('', np.nan, inplace=False).isna().sum()} without category")
            
                
            
    def check_data(self):
        # self._ask_with_guess(self.new_data)


        #todo: generalize to more than one account
        # prev_data = self.data['dkb'][self.data['dkb']['class'].isna() & self.data['dkb']['target account'].isna()]
        data_unlabeled = self.data_unlabeled
        predictions = self._make_predictions(data_unlabeled)
        data_unlabeled_with_guess = pd.concat([data_unlabeled, predictions], ignore_index=False, axis=1, copy=False, join='inner')

        data_unlabeled_with_guess.sort_values('class_prob', inplace=True) # sort such that the ones with the highest uncertainty come first
     
        df = self._ask_with_guess(prev_data)
        prev_data = pd.concat([df, self.prev_data])
        # self.prev_data.drop_duplicates(subset=self.prev_data.columns.difference(['cat', '']), inplace=True)
        self.prev_data.drop_duplicates(subset=['date', 'amount', 'desc'], inplace=True)
        
        logging.debug(f"total dataset after dropping duplicates {len(self.prev_data)} entries")
        logging.debug(f"\t {self.prev_data['class'].isna().sum()} without category") 

        self.prev_data.sort_values('date', inplace=True)
        
        self.prev_data.dropna(axis = 0, how = 'all', inplace = True)
        
        # self.prev_data = pd.concat([self.prev_data, self.new_data])
        # save data to the same file we loaded earlier
        self.prev_data.to_csv(self._datapath, index=False)
        logging.debug(f"saved dataset {len(self.prev_data)} entries")
        logging.debug(f"\t {self.prev_data['class'].replace('', np.nan, inplace=False).isna().sum()} without category")
        

#     def _prep_for_analysis(self):
#         """Prepare data for analysis in pandas, setting index types and subsetting"""
#         self.prev_data = self._make_date_index(self.prev_data)

#         self.prev_data['cat'] = self.prev_data['cat'].str.strip()

#         self.inc = self.prev_data[self.prev_data.amount > 0]
#         self.out = self.prev_data[self.prev_data.amount < 0]
#         self.out.amount = self.out.amount.abs()

#         self.inc_noignore = self.inc[self.inc.cat != 'Ignore']
#         self.inc_noexpignore = self.inc[(self.inc.cat != 'Ignore') & (self.inc.cat != 'Expenses')]

#         self.out_noignore = self.out[self.out.cat != 'Ignore']
#         self.out_noexpignore = self.out[(self.out.cat != 'Ignore') & (self.out.cat != 'Expenses')]

    def _read_categories(self):
        """Read list of categories from categories.txt"""
        categories = {}
        with open(self._datapath/'categories.txt') as f:
            for i, line in enumerate(f.readlines()):
                categories[i] = line.strip()

        return categories

    def _add_new_category(self, category):
        """Add a new category to categories.txt"""
        with open(self._workdir + 'categories.txt', 'a') as f:
            f.write('\n' + category)
            
    def print_categories(self, categories):
        
        # Generate the category numbers table from the list of categories
        cats_list = [[idnum, cat] for idnum, cat in categories.items()]
        cats_table = tabulate(cats_list)
        
        # print(chr(27) + "[2J")
        print(cats_table)
        print("\n\n")
        print("(q) to quit and (enter) to accept guess")
        print("\n\n")

    def _make_predictions(self, df):

        df_pred = pd.DataFrame(data = np.ones([len(df),2])*np.nan, columns=['class_guess', 'class_prob'])
        # df.loc['class_guess'] = np.nan
        categories = self._read_categories()

        def guess(row):
            assert isinstance(row['desc'], str), f"{row} is not a string"
            stripped_text = self._strip_numbers(row['desc'])

            # Guess a category using the classifier (only if there is data in the classifier)
            if len(self.classifier.train_set) > 1:
                guess = self.classifier.classify(stripped_text)
                prob = self.classifier.prob_classify(stripped_text).prob(guess)
            else:
                guess = np.nan
                prob = np.nan
                
            return guess, prob
        if len(df_pred)>0:
            df_pred = df.apply(lambda row: guess(row), axis='columns', result_type='expand')
            df_pred.columns = ['class_guess', 'class_prob']
            return df_pred
        else:
            logging.info('data set is empty')
            return None
            
        
    def _ask_with_guess(self, df):
        """Interactively guess categories for each transaction in df, asking each time if the guess
        is correct"""
        
        logging.info(f"asking with guess, total {len(df)}")
        print('============================================================')
        print('==== to exit interactive mode enter "q" and hit enter ======')
        print('============================================================')
        # Initialise colorama
        colorama.init()

        df['class'] = np.nan
        categories = self._read_categories()
        
        for index, row in df.iterrows():

            stripped_text = self._strip_numbers(row['desc'])
            
            logging.debug('>> tokens', list(self._extractor(stripped_text).keys()))

            # Guess a category using the classifier (only if there is data in the classifier)
            if len(self.classifier.train_set) > 1:
                guess = self.classifier.classify(stripped_text)
                prob = self.classifier.prob_classify(stripped_text).prob(guess)
            else:
                guess = np.nan
                prob = 0

            if prob < self.prob_threshold:
                # Print list of categories
                # self.print_categories(categories)
                # print(chr(27) + "[2J")
                # print(cats_table)
                # print("\n\n")
                # print("(q) to quit and (enter) to accept guess")
                # print("\n\n")
                
                # Print transaction
                print("On: %s\t %.2f\n%s" % (row['date'], row['amount'], row['desc']))
                print(colorama.Fore.RED  + colorama.Style.BRIGHT + f"My guess is: {guess} {100*prob:0.2f}%" + colorama.Fore.RESET)

                input_value = input("> ")
            else:
                print("On: %s\t %.2f\n%s" % (row['date'], row['amount'], row['desc']))
                print(colorama.Fore.BLUE  + colorama.Style.BRIGHT + f"My guess is: {guess} {100*prob:0.2f}%" + colorama.Fore.RESET)
                input_value = ""

            if input_value.lower() == 'q':
                print('exiting ...')
                return df
            if input_value == "":
                # If the input was blank then our guess was right!
                df.at[index, 'class'] = guess
                self.classifier.update([(stripped_text, guess)])
            else:
                # Otherwise, our guess was wrong
                try:
                    # Try converting the input to an integer category number
                    # If it works then we've entered a category
                    category_number = int(input_value)
                    category = categories[category_number]
                except ValueError:
                    # Otherwise, we've entered a new category, so add it to the list of
                    # categories
                    category = input_value
                    self._add_new_category(category)
                    categories = self._read_categories()

                # Write correct answer
                df.at[index, 'class'] = category
                # Update classifier
                self.classifier.update([(stripped_text, category)])
                
            print(f"current dataset, not categorized {(df['class'].isna()).sum()}/{len(df)}")

        return df

    def _make_date_index(self, df):
        """Make the index of df a Datetime index"""
        df.index = pd.DatetimeIndex(df.date.apply(dateutil.parser.parse,dayfirst=True))

        return df


    def _get_training(self, df):
        """Get training data for the classifier, consisting of tuples of
        (text, category)"""
        train = []
        # subset = df[df['cat'] != '']
        subset = df[~((df['class'] == '')  | (df['class'].isna()))]
        
        for i in subset.index:
            row = subset.loc[i]
            new_desc = self._strip_numbers(row['desc'])
            train.append( (new_desc, row['class']) )

        return train

    def _extractor(self, doc, min_length=3):
        """Extract tokens from a given string"""
        # TODO: Extend to extract words within words
        # For example, MUSICROOM should give MUSIC and ROOM
        
        delims = [' ', '/', ';', '<', '>', '//', ':', '-']
        tokens = self._split_by_multiple_delims(doc, delims)

        tokens = set([t.strip(''.join(delims)) for t in tokens if len(t)>= min_length])
        
        tokens = [t.lower() for t in tokens] # make lower case
        features = {}

        for token in tokens:
            if len(token) < min_length:
                continue
            # if token == "":
            #     continue
            features[token] = True

        return features

    def _strip_numbers(self, s):
        """Strip numbers from the given string"""
        stripped_text = re.sub(r'[0-9]', '', s)
        # stripped_text = re.sub("[^A-Z ]", "", s)
        return stripped_text

    def _split_by_multiple_delims(self, string, delims):
        """Split the given string by the list of delimiters given"""
        regexp = "|".join(delims)
        return re.split(regexp, string)

