import pandas as pd

def split_csv(input_file, train_output_file, test_output_file):
    # Read the CSV file into a DataFrame with explicit encoding
    df = pd.read_csv(input_file, encoding='latin1')  # Change 'utf-8' to the appropriate encoding

    # Split the DataFrame based on the "type" column
    train_df = df[df['type'] == 'train'][['review', 'label']]
    test_df = df[df['type'] == 'test'][['review', 'label']]

    # Save the split DataFrames to new CSV files
    train_df.to_csv(train_output_file, index=False)
    test_df.to_csv(test_output_file, index=False)

if __name__ == "__main__":
    # Replace 'input.csv', 'train.csv', and 'test.csv' with your actual file names
    input_file = 'imdb_master.csv'
    train_output_file = 'train.csv'
    test_output_file = 'test.csv'

    split_csv(input_file, train_output_file, test_output_file)
