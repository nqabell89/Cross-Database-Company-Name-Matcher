#import packages
import regex as re
import string
import itertools
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# find input csv files and convert to pandas dataframes
a_company_df = pd.read_csv('data/a__company.csv')
a_geo_df = pd.read_csv('data/a__geo.csv')
b_address_df = pd.read_csv('data/b__address.csv', low_memory=False)
b_company_df = pd.read_csv('data/b__company.csv')
b_hierarchy_df = pd.read_csv('data/b__hierarchy.csv')

# in this initial version of this script, only a__company.csv and b__company.csv are used
def main():
    
    print('tokenizing company names')
    
    # get tokens from company names in dataset a
    temp_a_df = pd.DataFrame()
    temp_a_df['a_name'] = a_company_df['name']
    temp_a_df['a_tokens'] = temp_a_df['a_name'].map(parse_tokens)
    
    # get tokens from company names in dataset b
    temp_b_df = pd.DataFrame()
    temp_b_df['b_name'] = b_company_df['entity_name']
    temp_b_df['b_tokens'] = temp_b_df['b_name'].map(parse_tokens)
    
    print('creating term/document frequency matrix')
    
    # create bag of words from each dataset
    a_token_list = [item for sublist in temp_a_df.a_tokens for item in sublist]
    b_token_list = [item for sublist in temp_b_df.b_tokens for item in sublist]
    
    # combine each of the two bag of words into one dataframe
    a_token_df = pd.DataFrame(a_token_list)
    a_token_df.columns = ['token']
    b_token_df = pd.DataFrame(b_token_list)
    b_token_df.columns = ['token']
    token_df = pd.concat([a_token_df, b_token_df], ignore_index=True)
    
    # output token/frequency or word count matrix, across both datasets, from token_df dataframe
    token_freq_matrix = dict(token_df.token.value_counts())
    
    # create final output dataframe, including the names from each dataset
    final_df = pd.DataFrame(columns = ['a_name', 'b_name', 'a_vendor_id', 'b_entity_id', 'confidence_score'])
    final_df['a_name'] = a_company_df['name']
    
    print('scoring similarity between company names in dataset a and b')
    
    for i in a_company_df.index:
        
        """
        WARNING: I'm getting an error that is breaking this function with the company in row 16 in a__company.csv
        This will require more time to fix than I have for the skills assessment, and so I have limited the script
        to output only the first 15 matches to showcase my methodology.
        """
        if i > 15:
            break
        
        final_df['a_name'][i] = a_company_df['name'][i]
        final_df['a_vendor_id'][i] = a_company_df['vendor_id'][i]
        
        #this is very slow without powerful processor, will need to be improved
        #this calculates a similarity score between each named entity in dataset a and dataset b, then selects the highest score as best
        scores = {}
        for ind in b_company_df.index:
            scores[ind] = name_similarity(a_company_df['name'][i], b_company_df['entity_name'][ind], token_freq_matrix)
        scores_sorted = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        best_score = dict(itertools.islice(scores_sorted.items(), 1))
        best_score_ind = list(best_score.keys())[0]
        best_score_value = list(best_score.values())[0]
        
        final_df['b_name'][i] = b_company_df['entity_name'][best_score_ind]
        final_df['b_entity_id'][i] = b_company_df['b_entity_id'][best_score_ind]
        final_df['confidence_score'][i] = best_score_value
        
        print(f'company {i+1} matched')
    
    print('saving to csv file')
    
    #output names, vendor and entity ids, and confidence scores into csv file
    final_df.to_csv('output_with_names.csv', index=False)
    
    print('Done!')

# this function is used to extract clean tokens from a company/entity name
def parse_tokens(text):
    
    # convert text to string
    text = str(text)
    
    # remove punctuation and special characters using regex package
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # tokenize text using NLTK package
    tokens = nltk.word_tokenize(text)
    
    # convert all words to lowercase
    tokens = [t.lower() for t in tokens]
    
    return tokens

def sequence_uniqueness(seq, token_freq):
    
    return sum(1/token_freq[t]**0.5 for t in seq)

def name_similarity(a, b, token_freq):
    a_tokens = set(parse_tokens(a))
    b_tokens = set(parse_tokens(b))
    a_uniq = sequence_uniqueness(a_tokens, token_freq)
    b_uniq = sequence_uniqueness(b_tokens, token_freq)

    return round((sequence_uniqueness(a_tokens.intersection(b_tokens), token_freq)/(a_uniq * b_uniq) ** 0.5), 3)

if __name__ == "__main__":
    main()