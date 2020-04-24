import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """

    :param messages_filepath: path to the messages csv file
    :param categories_filepath: path to the categories csv file
    :return: messages and categories combined data_frame
    """
    message_df = pd.read_csv(messages_filepath)
    categorie_df = pd.read_csv(categories_filepath)
    final_df = pd.merge(message_df, categorie_df, on='id')
    return final_df


def clean_data(df):
    """

    :param df: combined dataframe made after merging messages and the categories.
    :return: cleaned df
    """
    categories_df = df.categories.str.split(';', expand=True)  # creating a dataframe of the 36 individual category columns.
    row = categories_df.loc[0, :] # Selecting first row
    category_col_header = row.apply(lambda x: x.split('-')[0]).values # getting different column headers
    categories_df.columns = category_col_header # renaming the columns of `categories`

    for column in categories_df:
        # set each value to be the last character of the string
        categories_df[column] = categories_df[column].str.split('-').str[1]

    categories_df.apply(pd.to_numeric)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories_df], axis=1)
    return df

def save_data(df, database_filename):
    """

    :param df: combined and cleaned dataframe
    :param database_filename: database filepath
    """
    db_engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql("disaster_response", db_engine, if_exists='replace', index=False)




def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()