# Cross-Database-Company-Name-Matcher
A script and classification algorithm to accurately marry company name fields from two different databases.

TO-DO:
- Diagnose errors with dividing by zero in distance/similarity functions
- Increase accuracy by adding additional distance measurements
- Increase accuracy by incorporating similarity score threshold and geo meta data as filter
- Manually label several hundred correct matches
- Build SVM classification algorithm to predict matches on labeled data


1. Problem Statement & End Goal
 
Given two sample data sets as inputs, a and b, devise an algorithm that outputs a file that maps a__company.vendor_id to b__company.b_entity_id, including a confidence_of_match metric.

2. Technology & Packages

This script uses Python 3.8, and the following Python packages;
- Pandas
- RegEx
- string
- itertools
- NLTK

3. Methodology

    A. Data Exploration & Findings
    B. Parsing & Tokenization
    C. Matching: Measuring Similarity Between Company Names
    D. Scoring, Filtering, & Output
    E. Next Steps: Manual Labeling and ML Classification


A. Data Exploration & Findings

Based on exploration of the 5 csv files; I found that the best data for entity matching would be the "name" feature for vendor_id in dataset a, and the "entity_name" feature for b_entity_id in dataset b. 

Using NLP techniques developed specifically for the "fuzzy matching" of two sequences of text, I generated the majority of the accuracy needed to match these entities. 

In future versions of this script, I would further increase the accuracy by incorporating the geo/address/location meta data as a secondary "filter" or "check", which would help in the case of two companies with extremely similar or identical names that were HQed in two different regions.


B. Parsing & Tokenization

The first function in the script takes in the "name" and "entity_name" features from each dataset as raw text, and outputs cleaned and standardized lists of words or "tokens" for each company name. Punctiation and special characters were removed, and each word was converted to lower case. 

I also originally removed company legal control terms (like LLC, INC, PLC, SA, etc), but upon manual inspection of company names, later decided to keep these terms. There are many companies with the same name that are only distinguishable from their legal control terms.



C. Matching: Measuring Similarity Between Company Names

There are a few well known methodologies for measuring and scoring the similarity (or 'edit' distance) between two sequences of text. Levenshtein distance and Jaro-Winkler distance are the most common, but neither would be ideal for this project.

Levenshtein is best used for comparing text with typos, for example, and is simply the number of individual character changes needed to transform one string into the a matching version of the original.

The variation in company names has nothing to do with typos though, as there are multiple ways to refer to the same company;

"Ford Motor Company" is the same as "Ford". "Tesco" is the same as "Tesco PLC". 

Even "Abell Consulting Company" is probably related to "Abell Limited", given how unique the word "Abell" is, as well as how "Limited", "Company" and "Consulting" are extremely common. A simple distance calculation would not catch this, because it doesn’t take into account how unique each word is in terms of their overall frequency in the larger corpus of words.

In this case, a much better distance measure would look at the words the two company names have in common, rather than the characters. It would also discount the words according to their uniqueness. For example; the word "Limited" occurs in many company names, "Consulting" is more important but still very common, and a unique name like "Abell" would be very rare and weighted higher in importance.

This is achieved by calculating a simple term frequency matrix or "word count" by combining all tokens from all names from both datasets into one, and counting the number of occurences of each term. This is commonly performed by sklearn's tfidf vectorizer function, but I manually created my own simple token count version. The reason why will be explained in the next step.



D. Scoring, Filtering, & Output

The final two functions in this script are sequence_uniqueness and name_similarity.

sequence_uniqueness calculates a fractional value of "uniqeness" for each token in a list of tokens, using the bag-of-word inverse term frequency matrix (word count) generated in the previous step, and sums them for a combined value.

name_similarity uses both the parsing/tokenization function and the sequence_uniqueness function to iterate through company name in dataset a and dataset b, and within a nested for loop, calculate and compare the scalar product of each set of names in the bag-of-word inverse term frequency matrix (word count) space. I used the square root (instead of log, like in sklearn's tfidf vectorizer) because it produces more intuitive "scores" between 0 and 1. 

I then matched the dataset a entity with the highest available similarity/confidence score from dataset b. 

In a future version of this script, I would use a certain score threshold, say below 0.9 but above 0.5, to run a secondary check to compare geo/address/location meta data. This would further increase accuracy.



E. Next Steps: Manual Labeling and ML Classification

In a future version of this script, I will take several scores (this uniqueness score, Jaccard distance, and Jaro-Winkler distance) and train a binary classification model using some manually labeled data, which will, given a number of scores, output if the candidate pair is a match or not.

This would allow me to build a trained classification algorithm that would work on new/unseen entity name entries, or between new datasets with new entity name lists.
