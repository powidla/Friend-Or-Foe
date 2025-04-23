import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split
import os


def read_csv(path):
    return pd.read_csv(path)

def rearrange(df):
  '''
  Description: For concat environments matrix to set the columns.

  Arguments: df - pandas dataframe with all the environments;

  Outputs: df - pandas dataframe with new column names corresponding to the specific compound.

  '''

  binary_row = (df.iloc[0, :] >= 0.5).astype(int)
  df.loc[len(df)] = binary_row
  df.columns = [f"C_{i}" for i in range(1, len(df.columns) + 1)]

  return df


def reset(df):
  '''
  Description: For M_iM_j matrix to reset the columns.

  Arguments: df - pandas dataframe with indices of environmental microbes;

  Outputs: df - pandas dataframe with new column names.

  '''

  df = df.reset_index(drop=True)  # Ensure existing index is reset
  df.loc[-1] = list(df.columns)  # Add column names as a new row
  df.index = df.index + 1        # Shift index down by 1
  df = df.sort_index()

  df.columns = ['M1', 'M2']

  return df


def split_dfs(df):
  '''
  Description: For M_iM_j matrix to split into several csvs with unique M_i M_j pairs.

  Arguments: df - pandas dataframe with indices of environmental microbes;

  Outputs: creates multiple dfs for each unique pairing.

   '''

  grouped_dfs = [group.reset_index(drop=True) for _, group in df.groupby(['M1', 'M2'])]


  output_folder = 'split_dfs'
  os.makedirs(output_folder, exist_ok=True)

  for i, sub_df in enumerate(grouped_dfs):
      filename = f"{output_folder}/group_{i + 1}.csv"
      sub_df.to_csv(filename, index=False)
      # print(f"Saved: {filename}")


def read_subset(df, M_i, M_j):
  '''

  Description: Counts number of rows for the specific pair.

  Arguments: df - pandas dataframe with indices of environmental microbes;
            M_i (int) - index for a microbe agent;
            M_j (int) - index for a microbe agent;

  Outputs: parse the number of row for a specific pair.

  '''

  num_rows = df[(df['M1'] == M_i) & (df['M2'] == M_j)].shape[0]

  print(f"The file with M1={M_i} and M2={M_j} has {num_rows} rows.")


def assign_labels(df, type_int):
  '''
  Description: Creates a y vector as a label for each row given an env matrix.
                  0 - competition;
                  1 - facultative cooperation;
                  2 - obligate_plusx: are such that the first listed bacteria could grow by itself in that environment but the second could not (and was obligate on the first);
                  3 - obligate_xplus: are such that the first listed bacteria could NOT grow by itself in that environment and was obligate on the second which could alone;
                  4 - obligate_xx: are such that neither bacteria could grow alone and both depended on each other;
  Arguments: df - pandas dataframe with all the environments;
             type (str) - 'fcoop', 'comp', 'obligate_plusx', 'obligate_xplus', 'obliagete_xx';

  Outputs: new df with additional label vector with shape (df.shape[0], df.shape[1]+1).

  '''

  if type_int == 'comp':
    df['label'] = 0
  elif type_int == 'fcoop':
    df['label'] = 1
  elif type_int == 'obligate_plusx':
    df['label'] = 2
  elif type_int == 'obligate_xplus':
    df['label'] = 3
  elif type_int == 'obligate_xx':
    df['label'] = 4

  return df


def concat_dfs(df_list, axis=0, ignore_index=True):
  '''
  Description: Creates a combined dataset for supervised learning.

  Arguments: df_list of pandas dataframes;
             axis to concat;
             ignore_index - flag to index;

  Outputs: new df with additional label vector with shape (df1.shape[0]+df2.shape[0]+.., df1.shape[1]).

  '''
  if not all(isinstance(df, pd.DataFrame) for df in df_list):
      raise ValueError("All elements in dataframe_list must be pandas DataFrames.")
  return pd.concat(df_list, axis=axis, ignore_index=ignore_index)

def create_X_y(df, label_column='label'):
    '''
    Description: Splits the data into features (X) and labels (y).

    Arguments: data (pd.DataFrame): Input DataFrame;
               label_column (str): Name of the column to use as the label (target).

    Outputs: X (pd.DataFrame): Feature data;
             y (pd.Series): Target labels.
    '''
    X = df.drop(columns=[label_column])
    y = df[label_column]
    return X, y


def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=4421):
    """
    Description: Splits the data into training, validation, and test sets.

    Arguments: X (pd.DataFrame or np.array): Feature data;
               y (pd.Series or np.array): Target labels;
               val_size (float): Proportion of the dataset to include in the validation set;
               test_size (float): Proportion of the dataset to include in the test set;
               random_state (int): Random seed for reproducibility.

    Outputs: X_train, X_val, X_test (pd.DataFrame): Feature splits;
             y_train, y_val, y_test (pd.Series): Label splits.
    """
    # First split into train + val_test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state
    )

    # Calculate the tmp set ratio
    val_ratio = val_size / (val_size + test_size)

    # Split then into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test