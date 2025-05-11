import pandas as pd
import re

def clean_dish_names(input_file, output_file):
    # Load the dataset from CSV
    df = pd.read_csv(input_file)
    
    # Ensure that the 'Dish Name' column is treated as string
    df['Dish Name'] = df['Dish Name'].astype(str)
    
    # Remove numbers from the dish names
    df['Dish Name'] = df['Dish Name'].apply(lambda x: re.sub(r'\d+', '', x))
    
    # Remove the word "ingredient" or "ingredients" (case insensitive)
    df['Dish Name'] = df['Dish Name'].apply(lambda x: re.sub(r'\bingredients?\b', '', x, flags=re.IGNORECASE))
    
    # Remove any extra whitespace from the start and end of the dish names
    df['Dish Name'] = df['Dish Name'].str.strip()
    
    # Save the cleaned dataset to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to {output_file}")
    return df

if __name__ == "__main__":
    input_file = 'data.csv'       # Your input CSV file
    output_file = 'cleaned_data.csv'  # The output CSV file for the cleaned dataset
    
    # Clean the dish names and save the result
    clean_dish_names(input_file, output_file)
