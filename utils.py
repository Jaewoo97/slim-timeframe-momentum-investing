def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key

def refineSymbols(sumWeights):
    for key in sumWeights:
        foo = sumWeights[key].iloc[1]
        if str(foo) == 'nan':
            sumWeights.at[1, key] = '.'
