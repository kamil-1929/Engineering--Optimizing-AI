import pandas as pd

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Encode labels
    label_map = {
        'Type2': {'Others': 0, 'Problem/Fault': 1, 'Suggestion': 2},
        'Type3': {'Install/Upgrade': 0, 'Use': 1, 'Third Party APPs': 2, pd.NA: -1},
        'Type4': {"Can't update Apps": 0, 'Others': 1, 'Refund': 2, pd.NA: -1}
    }
    
    for col, mapping in label_map.items():
        df[col] = df[col].map(mapping)
    
    return df
