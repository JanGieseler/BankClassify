import numpy as np
import pandas as pd
from datetime import datetime, timezone
import re
from pathlib import Path
from collections import Counter
import os
import colorama

from textblob.classifiers import NaiveBayesClassifier


class BankClassify():

    def __init__(self, datapath=Path(".."), verbose=0, prob_threshold=0.9):
        """
        Load in the previous data (by default from `data`) and initialise the classifier
        agg_data_file: filename where the aggregated data is stored

        
        """
        assert prob_threshold <=1
        assert prob_threshold >=0
        self.prob_threshold = prob_threshold
        self._datapath = datapath
        self.verbose = verbose
        
        self.data = {}

        if self._datapath.exists():
            if (self._datapath/'dkb.csv').exists():
                self.data['dkb'] = pd.read_csv(self._datapath/'dkb.csv' , index_col=0)
                self.data['dkb']['date'] = pd.to_datetime(self.data['dkb']['date'], format = '%Y-%m-%d')
                print(f"loaded previous dkb data {len(self.data['dkb'])} entries")

        # data_train = self._get_training(self.prev_data)
        
        data_train = []
        for df in self.data.values():
            data_train += self._get_training(df)
        if self.verbose >= 2:
            print(f'train dataset size {len(data_train)}')
            
            
        categories = self._read_categories()
        # self.print_categories(categories)
        self.data_train = data_train
        self.classifier = NaiveBayesClassifier(data_train, self._extractor)

#     def add_paypal_data(self, filename):

#         new_data = self._read_paypal_csv(filename)

#         if self._data_paypal is None:
#             self._data_paypal = new_data
#         else:
#             if self.verbose >=2:
#                 print(f"paypal data before with {len(self._data_paypal)} entries")
#             self._data_paypal =  pd.concat([self._data_paypal, new_data])

#         self._data_paypal.drop_duplicates(inplace=True)


#         self._data_paypal.to_csv(self._datapath_paypal, index=False)
#         if self.verbose >=2:
#             print(f"saved paypal dataset {len(self._data_paypal)} entries")
            

    def add_data(self, filename, bank="santander"):
        """Add new data and interactively classify it.

        Arguments:
         - filename: filename of Santander-format file
        """

        if bank == "dkb":
            print("adding DKB Bank data!")
            self.new_data = self._read_dkb_csv(filename)
        else:
            raise ValueError('new_data appears empty! probably tried an unknown bank: ' + bank)

        if self.verbose >=2:
            print(f"previous dataset {len(self.prev_data)} entries")
            print(f"\t {self.prev_data['cat'].isna().sum()} without category")
            print(f"new dataset {len(self.new_data)} entries")
        
        self.prev_data = pd.concat([self.prev_data, self.new_data])
        print(f"dropping duplicates considering only ['date', 'amount', 'desc']")
        self.prev_data.drop_duplicates(subset=['date', 'amount', 'desc'], inplace=True)
        
        if self.verbose >=2:
            print(f"total dataset after dropping duplicates {len(self.prev_data)} entries")
            print(f"\t {self.prev_data['cat'].isna().sum()} without category")
            
        self.prev_data.to_csv(self._datapath, index=False)
        if self.verbose >=2:
            print(f"saved dataset {len(self.prev_data)} entries")
            print(f"\t {self.prev_data['cat'].replace('', np.nan, inplace=False).isna().sum()} without category")
        
            
            
    def check_data(self):
        # self._ask_with_guess(self.new_data)


        #todo: generalize to more than one account
        prev_data = self.data['dkb'][self.data['dkb']['class'].isna() & self.data['dkb']['target account'].isna()]
        print(f">>>> {len(prev_data)}")
        prev_data = self._make_predictions(prev_data)
        prev_data.sort_values('cat_prob', inplace=True) # sort such that the ones with the highest uncertainty come first
     
        print('check all nan')
        df = self._ask_with_guess(prev_data)
        print(f">>3 df>> {len(df)}")
        self.prev_data = pd.concat([df, self.prev_data])
        print(f">>3>> {len(self.prev_data)}")
        # self.prev_data.drop_duplicates(subset=self.prev_data.columns.difference(['cat', '']), inplace=True)
        self.prev_data.drop_duplicates(subset=['date', 'amount', 'desc'], inplace=True)
        if self.verbose >=2:
            print(f"total dataset after dropping duplicates {len(self.prev_data)} entries")
            print(f"\t {self.prev_data['cat'].isna().sum()} without category") 

        self.prev_data.sort_values('date', inplace=True)
        
        print(f">>4>> {len(self.prev_data)}")
        
        self.prev_data.dropna(axis = 0, how = 'all', inplace = True)
        print(f">>5>> {len(self.prev_data)}")
        
        # self.prev_data = pd.concat([self.prev_data, self.new_data])
        # save data to the same file we loaded earlier
        self.prev_data.to_csv(self._datapath, index=False)
        if self.verbose >=2:
            print(f"saved dataset {len(self.prev_data)} entries")
            print(f"\t {self.prev_data['cat'].replace('', np.nan, inplace=False).isna().sum()} without category")
        

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
        df['cat_guess'] = ""
        categories = self._read_categories()

        def guess(row):
            assert isinstance(row['desc'], str), f"{row} is not a string"
            stripped_text = self._strip_numbers(row['desc'])

            # Guess a category using the classifier (only if there is data in the classifier)
            if len(self.classifier.train_set) > 1:
                guess = self.classifier.classify(stripped_text)
                prob = self.classifier.prob_classify(stripped_text).prob(guess)
            else:
                guess = ""
                prob = 0
                
            return guess, prob
        
        df[['class_guess', 'class_prob']] = self.prev_data.apply(lambda row: guess(row), axis='columns', result_type='expand')
        return df
            
        
    def _ask_with_guess(self, df):
        """Interactively guess categories for each transaction in df, asking each time if the guess
        is correct"""
        
        if self.verbose >=2:
            print(f"asking with guess, total {len(df)}")
        
        # Initialise colorama
        colorama.init()

        df['class'] = np.nan
        categories = self._read_categories()
        
        for index, row in df.iterrows():

            stripped_text = self._strip_numbers(row['desc'])
            
            # print('>> stripped_text', stripped_text)
            print('>> tokens', list(self._extractor(stripped_text).keys()))

            # Guess a category using the classifier (only if there is data in the classifier)
            if len(self.classifier.train_set) > 1:
                guess = self.classifier.classify(stripped_text)
                prob = self.classifier.prob_classify(stripped_text).prob(guess)
            else:
                guess = ""
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
                print(Fore.RED  + Style.BRIGHT + f"My guess is: {guess} {100*prob:0.2f}%" + Fore.RESET)

                input_value = input("> ")
            else:
                print("On: %s\t %.2f\n%s" % (row['date'], row['amount'], row['desc']))
                print(Fore.BLUE  + Style.BRIGHT + f"My guess is: {guess} {100*prob:0.2f}%" + Fore.RESET)
                input_value = ""

            if input_value.lower() == 'q':
                # If the input was 'q' then quit
                if self.verbose >=2:
                    print('exiting ...')
                return df
            if input_value == "":
                # If the input was blank then our guess was right!
                df.at[index, 'cat'] = guess
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
                df.at[index, 'cat'] = category
                # Update classifier
                self.classifier.update([(stripped_text, category)])
                
            if self.verbose >=2:
                print(f"current dataset, not categorized {(df['cat']=='').sum()}/{len(df)}")

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
        # print(doc)
        # print('>> tokens', tokens)
        tokens = set([t.strip(''.join(delims)) for t in tokens if len(t)>= min_length])
        
        tokens = [t.lower() for t in tokens] # make lower case
        # print('>> tokens', tokens)
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
