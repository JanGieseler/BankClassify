from BankClassify import BankClassify

bc = BankClassify()
bc = BankClassify(data_path="AllData.csv", verbose=2, check_all_new=False)

filename = '../data/dkb/2020_1002423109.csv' # "Statement_Example.txt"
# filename = "Statement_Example.txt"

bc.add_data(filename, "dkb")

# bc.add_data("85561768_20205411_0903.csv", "lloyds")