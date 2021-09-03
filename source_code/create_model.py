import unicodedata as unic
import re
import string
import os
import textwrap
from urllib.parse import quote_plus

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import editdistance as ed
import dill
import pyodbc
import pandas as pd
import sqlalchemy


SECRET_VARIABLES = ["SERVER_NAME", "DATABASE", "LOGIN_INPUT", "PASSWORD_INPUT"]
DRIVER = "{ODBC Driver 17 for SQL Server}"


def fetch_azure_connection_string():
    secrets = {}
    for secret in SECRET_VARIABLES:
        secrets[secret] = os.getenv(secret, False)
        if not secrets[secret]:
            raise NameError(f"{secret} is not set as an environment variable!")
    server = f"{secrets['SERVER_NAME']}.database.windows.net,1433"

    con_str = textwrap.dedent(
        f"""
        Driver={DRIVER};
        Server={server};
        Database={secrets["DATABASE"]};
        Uid={secrets["LOGIN_INPUT"]};
        Pwd={secrets["PASSWORD_INPUT"]};
        Encrypt=yes;
        TrustServerCertificate=no;
        Connection Timeout=30;
    """
    )
    return con_str


class DB_Connection:
    def __init__(self):
        connection_string = self.secrets = fetch_azure_connection_string()
        self.cnxn: pyodbc.Connection = pyodbc.connect(connection_string)
        self.crsr: pyodbc.Cursor = self.cnxn.cursor()

    def get_data(self):
        sql_code = "SELECT material_id, material_description FROM proc_db.zmm001 WHERE material_type = 'ZMAT'"
        return pd.read_sql(sql_code, self.cnxn)

    def run_query(self, sql_str=None, sql_file=None):
        if (sql_str is None) & (sql_file is None):
            raise Exception("A query or file need to be provided to run this method")
        elif sql_file is not None:
            with open(sql_file, "r") as sql:
                sql_str = sql.read()
        self.crsr.execute(sql_str)
        self.cnxn.commit()
        return True
        

class MaterialDistanceCalculator:
    def __init__(self, dataframe):
        self.df = dataframe
        self.process_dataframe()
    
    def get_materials(self, query):
        import numpy as np

        search_items = self.df.description_clean
        distances = np.empty((len(search_items),))

        for i, item in enumerate(search_items):
            distances[i] = self.dist_sentence(query, item)
        
        return self.df.iloc[np.argsort(distances)[0:10],:2].reset_index(drop=True).to_dict()


    def process_dataframe(self):
        nltk.download("stopwords")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("wordnet")

        self.df = self.df[self.df.MaterialType=="ZMAT"].copy()
        # clean text data
        self.df["description_clean"] = self.df["MaterialDescription"].apply(lambda x: self.clean_text(x) if x is not None else "")

    def clean_text(self, text):
        # multiple functions to clean the t
        # clean any unicode formatting left
        text = unic.normalize("NFKD", text)
        # lower text
        text = text.lower()
        # tokenize text and remove puncutation
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        # replace the rest of the punctuation by space
        text = ' '.join(text)
        text = [w for w in re.split('\.|\-|\s|\,|\(|\_|\d', text)]
        # remove words that contain numbers
        text = [word for word in text if not any(c.isdigit() for c in word)]
        # remove stop words | Not really needed
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        # remove empty
        text = [t for t in text if len(t) > 1]
        # pos tag text
        pos_tags = pos_tag(text)
        # lemmatize text
        text = [WordNetLemmatizer().lemmatize(t[0],  self.get_wordnet_pos(t[1])) for t in pos_tags]
        # remove words with only one letter
        text = [t for t in text if len(t) > 2]
        # join all
        text = " ".join(text)
        return(text)

    @staticmethod
    def get_wordnet_pos(pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    @staticmethod
    def dist_sentence(sentence_1, sentence_2):
        """Distance for sentence adapted from levenshtein but with no fixed order."""
        import editdistance as ed
        import numpy as np

        initial = sentence_1 + "|" + sentence_2
        
        sentence_1_set = set(sentence_1.split(' '))
        sentence_2_set = set(sentence_2.split(' '))
        
        # Get the maximum length of the sentences to use in the end
        # when we have to calculate the percentage of difference between 
        # both sentences.
        ln_max = max(len(sentence_1_set), len(sentence_2_set))
        
        # Remove common words
        sentence_1_list = list(sentence_1_set - sentence_2_set)
        sentence_2_list = list(sentence_2_set - sentence_1_set)
        
        # Get the current lengths
        m = len(sentence_1_list)
        n = len(sentence_2_list)
        ln = max(m, n)

        # Initialize distance variable
        distance = 0
        
        # Find similar words
        for i, s1 in enumerate(sentence_1_list):
            # Continue if there are still values in sentence_2 list
            if sentence_2_list:
                values = np.zeros((1, n))
                for j, s2 in enumerate(sentence_2_list):
                    l_word = max(len(s1),len(s2))
                    values[0,j] = ed.eval(s1, s2) / l_word
                try:
                    value = np.amin(values)
                except:
                    print(initial)
                if value < 0.33:
                    index = np.argmin(values)
                    # Add relative leven distance to distance
                    distance += value
                    # Remove the tokens
                    sentence_2_list.remove(sentence_2_list[index])
                    ln -= 1
                    n -= 1
        
        # Add the missing values that weren't matched
        distance += ln
        
        return int((distance / ln_max) * 100)

if __name__ == "__main__":

    dbcon = DB_Connection()
    df = dbcon.get_data()

    print("Got data")
    mdc = MaterialDistanceCalculator(df)

    print("Instantiated")
    # Dumping into pickle
    with open("models/ms-distance-calculator.pkl", "wb") as f:
        dill.dump(mdc, f)
