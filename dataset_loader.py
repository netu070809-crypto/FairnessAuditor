import pandas as pd

def load_clean_data():
    # Read the raw whitespace-separated UCI dataset
    df = pd.read_csv("credit_data.csv", sep=r'\s+', header=None)
    
    # Column 8 in 0-indexed python is the 9th column (Personal status and sex)
    # Column 20 in 0-indexed python is the 21st column (The credit risk label: 1=Good, 2=Bad)
    sensitive_attribute = df[8]
    target_col = df[20]
    
    # Map 1 (Good) to 1, and 2 (Bad) to 0
    df['target_mapped'] = target_col.apply(lambda x: 1 if int(x) == 1 else 0)
    
    return df, sensitive_attribute, df['target_mapped']

if __name__ == "__main__":
    df, _, _ = load_clean_data()
    print("✨ SUCCESS: dataset_loader.py is fully functional!")
    print(f"Total Rows Loaded: {len(df)} rows found.")
