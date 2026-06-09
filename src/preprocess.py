import pandas as pd
import sys
import yaml
import os


##Load parameters from yaml file
params = yaml.safe_load(open("params.yaml")) 

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    
if __name__ == "__main__":
    preprocess(params["preprocess"]["input"], params["preprocess"]["output"])