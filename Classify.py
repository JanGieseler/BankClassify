from BankClassify import BankClassify

# bc = BankClassify()


account = "1002423109"
account = "1071944217"
#bc = BankClassify(data_path=f"all_data_dkb_{account}.csv", verbose=2, prob_threshold=1)
bc = BankClassify(verbose=2, prob_threshold=1)

filename = f'../data/dkb/2020_{account}.csv' # "Statement_Example.txt"
filename = f'../data/dkb/2021_{account}.csv' # "Statement_Example.txt"
filename = f'../data/dkb/2022_{account}.csv' # "Statement_Example.txt"

filename_paypal = f'../data/Paypal/Paypal_01 Jan 2019 - 04 Aug 2022.csv' # "Statement_Example.txt"

# filename = "Statement_Example.txt"

# bc.add_data(filename, "dkb")

bc.add_paypal_data(filename_paypal)

# bc.add_data("85561768_20205411_0903.csv", "lloyds")