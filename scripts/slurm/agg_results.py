#!/usr/bin/env python3

import os
import re
import ast
import csv
import torch

def parse_namespace_line(line: str) -> dict:
    """
    Given a line that looks like:
        Namespace(data='/path', batch=64, randomize=False, ...)
    Convert it into a Python dictionary of the arguments:
        {
          'data': '/path',
          'batch': 64,
          'randomize': False,
          ...
        }
    """
    # Strip "Namespace(" at the start and the closing ")" at the end
    if not line.startswith("Namespace(") or not line.endswith(")"):
        raise ValueError("Line does not match 'Namespace(...)' format.")
    contents = line[len("Namespace("):-1].strip()
    
    # Turn it into a valid Python expression by prefixing with "dict(" and suffixing with ")"
    # i.e., "dict(data='/path', batch=64, randomize=False, ...)"
    dict_expression = "dict(" + contents + ")"
    
    # Safely evaluate the string as a Python literal
    try:
        parsed_dict = ast.literal_eval(dict_expression)
    except Exception as e:
        raise ValueError(f"Unable to parse Namespace line:\n{line}\nError: {e}")

    return parsed_dict


def aggregate_results(root_dir: str, output_csv: str = None):
    """
    Given a root directory, find all subfolders that contain:
      - args.log (required)
      - avg_metrics.pt (optional)
    
    Parse the hyperparameters from args.log (first line),
    load the metrics from avg_metrics.pt if present,
    and collect them all into a single CSV.
    
    If no output_csv is provided, this will automatically use
    the 'out' parameter from the first log it finds as the output location
    and name it 'aggregated_results.csv'.
    """
    # We'll accumulate rows of data here
    all_rows = []
    # We'll keep track of all possible columns for hyperparameters and metrics
    hparam_keys = set()
    metric_keys = set()
    
    # Walk through each item in the root directory
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue  # skip non-directories
        
        args_log = os.path.join(folder_path, 'args.log')
        if not os.path.isfile(args_log):
            # Folder does not have args.log -> do not include
            continue
        
        # Parse the first line of args.log as a Namespace
        with open(args_log, 'r') as f:
            first_line = f.readline().strip()
        try:
            hparams = parse_namespace_line(first_line)
        except ValueError as e:
            print(f"Skipping folder {folder} due to parse error: {e}")
            continue
        
        # Mark that we've seen these hparam keys
        for k in hparams.keys():
            hparam_keys.add(k)
        
        # Attempt to load avg_metrics.pt
        metrics_file = os.path.join(folder_path, 'avg_metrics.pt')
        if os.path.isfile(metrics_file):
            try:
                metrics_data = torch.load(metrics_file, map_location='cpu')
                # If the loaded object is not a dict, wrap it or handle differently
                if not isinstance(metrics_data, dict):
                    print(f"Warning: {metrics_file} did not load a dict. Wrapping in dict.")
                    metrics_data = {'value': metrics_data}
            except Exception as e:
                print(f"Could not load metrics from {metrics_file}: {e}")
                metrics_data = {}
        else:
            metrics_data = {}
        
        for mkey in metrics_data.keys():
            metric_keys.add(mkey)
        
        # Prepare the row that combines hyperparams + metrics
        row_data = {
            **hparams,
            **metrics_data
        }
        
        # We'll include a special 'uuid' or 'folder' column
        # in case we want to identify the directory in the CSV
        row_data['uuid'] = folder
        
        all_rows.append(row_data)

    # Now determine final CSV field names in a stable order:
    # 1. sort hyperparam keys
    # 2. sort metric keys
    # 3. ensure 'uuid' is included at the front for clarity
    # (or any order you prefer)
    fieldnames = ['uuid'] + sorted(hparam_keys) + sorted(metric_keys)
    
    # If output_csv was not given, attempt to guess from the first row's 'out'
    if not output_csv:
        # Get the 'out' from the first row, if present
        if len(all_rows) == 0:
            # No data to write
            print("No valid folders found with args.log, nothing to aggregate.")
            return
        example_row = all_rows[0]
        out_dir = example_row.get('out', '.')  # default to current dir if 'out' not in hparams
        # Create the output path
        output_csv = os.path.join(out_dir, 'aggregated_results.csv')
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Write out the CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            # Because hparam_keys and metric_keys might not exist in every row,
            # we fill missing columns with None
            row_to_write = {k: row.get(k, None) for k in fieldnames}
            writer.writerow(row_to_write)
    
    print(f"Aggregated results written to: {output_csv}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate hparams and metrics from UUID folders.")
    parser.add_argument("root", help="Path to the root directory containing subfolders.")
    args = parser.parse_args()
    
    aggregate_results(args.root, args.root)
