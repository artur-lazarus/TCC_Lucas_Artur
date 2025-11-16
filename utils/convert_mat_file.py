#!/usr/bin/env python3
"""
Utility script to convert .mat files to different formats

Supported output formats:
- json: JSON format
- npz: NumPy compressed archive (recommended for multiple arrays)
- pickle: Python pickle format
- hdf5: HDF5 format
- csv: CSV format (for 2D arrays only)
- txt: Plain text format (class names, one per line)

Usage:
    python convert_mat_file.py <input.mat> <output_format> [options]
    
Examples:
    python convert_mat_file.py data.mat json
    python convert_mat_file.py data.mat npz
    python convert_mat_file.py data.mat hdf5 -o output.h5
    python convert_mat_file.py data.mat csv -v variable_name
    python convert_mat_file.py data.mat txt -v class_names
"""

import sys
import os
import argparse
import numpy as np
import json


def load_mat_file(filepath):
    """
    Load a .mat file using the appropriate method
    
    Returns:
        dict: Dictionary of variables from the .mat file
    """
    try:
        import scipy.io
        mat_data = scipy.io.loadmat(filepath)
        # Remove MATLAB metadata variables
        mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        print(f"✓ Loaded {len(mat_data)} variables from {filepath}")
        return mat_data
    except NotImplementedError:
        # Try h5py for MATLAB v7.3 files
        try:
            import h5py
            mat_data = {}
            
            with h5py.File(filepath, 'r') as f:
                def load_h5_item(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        mat_data[name.replace('/', '_')] = obj[()]
                
                f.visititems(load_h5_item)
            
            print(f"✓ Loaded {len(mat_data)} variables from {filepath} (HDF5 format)")
            return mat_data
        except ImportError:
            print("ERROR: h5py not installed. Install with: pip install h5py")
            sys.exit(1)
    except ImportError:
        print("ERROR: scipy not installed. Install with: pip install scipy")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading file: {e}")
        sys.exit(1)


def convert_to_json(mat_data, output_path):
    """Convert .mat data to JSON format"""
    
    def convert_numpy_to_json(obj):
        """Recursively convert numpy objects to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            # Handle different array types
            if np.issubdtype(obj.dtype, np.complexfloating):
                # Complex numbers: save as dict with real and imag parts
                return {
                    'real': obj.real.tolist(),
                    'imag': obj.imag.tolist(),
                    'dtype': 'complex'
                }
            elif obj.dtype == np.object_:
                # Object arrays: recursively convert each element
                return [convert_numpy_to_json(item) for item in obj.flat]
            else:
                return obj.tolist()
        elif isinstance(obj, np.generic):
            # Handle numpy scalar types
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_to_json(item) for item in obj]
        else:
            return obj
    
    json_data = {}
    
    for key, value in mat_data.items():
        json_data[key] = convert_numpy_to_json(value)
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✓ Saved to {output_path} (JSON format)")
    return output_path


def convert_to_npz(mat_data, output_path):
    """Convert .mat data to NumPy NPZ format"""
    np.savez_compressed(output_path, **mat_data)
    print(f"✓ Saved to {output_path} (NPZ format)")
    print(f"  Load with: data = np.load('{output_path}')")
    return output_path


def convert_to_pickle(mat_data, output_path):
    """Convert .mat data to Python pickle format"""
    import pickle
    
    with open(output_path, 'wb') as f:
        pickle.dump(mat_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Saved to {output_path} (Pickle format)")
    print(f"  Load with: data = pickle.load(open('{output_path}', 'rb'))")
    return output_path


def convert_to_hdf5(mat_data, output_path):
    """Convert .mat data to HDF5 format"""
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed. Install with: pip install h5py")
        sys.exit(1)
    
    with h5py.File(output_path, 'w') as f:
        for key, value in mat_data.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value, compression='gzip')
            else:
                f.create_dataset(key, data=value)
    
    print(f"✓ Saved to {output_path} (HDF5 format)")
    print(f"  Load with: f = h5py.File('{output_path}', 'r')")
    return output_path


def convert_to_csv(mat_data, output_path, variable_name=None):
    """Convert .mat data to CSV format (for 2D arrays)"""
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Install with: pip install pandas")
        sys.exit(1)
    
    if variable_name:
        # Convert specific variable
        if variable_name not in mat_data:
            print(f"ERROR: Variable '{variable_name}' not found in .mat file")
            print(f"Available variables: {list(mat_data.keys())}")
            sys.exit(1)
        
        value = mat_data[variable_name]
        if not isinstance(value, np.ndarray):
            print(f"ERROR: Variable '{variable_name}' is not an array")
            sys.exit(1)
        
        if value.ndim == 1:
            df = pd.DataFrame({variable_name: value})
        elif value.ndim == 2:
            df = pd.DataFrame(value)
        else:
            print(f"ERROR: Variable '{variable_name}' has {value.ndim} dimensions")
            print("CSV format only supports 1D or 2D arrays")
            sys.exit(1)
        
        df.to_csv(output_path, index=False)
        print(f"✓ Saved variable '{variable_name}' to {output_path} (CSV format)")
    else:
        # Save all 2D arrays to separate CSV files
        base_path = output_path.rsplit('.', 1)[0]
        saved_files = []
        
        for key, value in mat_data.items():
            if isinstance(value, np.ndarray) and value.ndim <= 2:
                csv_path = f"{base_path}_{key}.csv"
                
                if value.ndim == 1:
                    df = pd.DataFrame({key: value})
                else:
                    df = pd.DataFrame(value)
                
                df.to_csv(csv_path, index=False)
                saved_files.append(csv_path)
        
        if saved_files:
            print(f"✓ Saved {len(saved_files)} variables to CSV files:")
            for f in saved_files:
                print(f"  - {f}")
        else:
            print("ERROR: No 2D arrays found to save as CSV")
            sys.exit(1)
    
    return output_path


def convert_to_txt(mat_data, output_path, variable_name=None):
    """Convert .mat data to TXT format (class names, one per line)"""
    
    if variable_name:
        # Convert specific variable
        if variable_name not in mat_data:
            print(f"ERROR: Variable '{variable_name}' not found in .mat file")
            print(f"Available variables: {list(mat_data.keys())}")
            sys.exit(1)
        
        value = mat_data[variable_name]
        
        # Handle different data types
        class_names = []
        
        if isinstance(value, np.ndarray):
            if value.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object arrays
                # Flatten the array and convert to strings
                flat_array = value.flatten()
                for item in flat_array:
                    if isinstance(item, (bytes, np.bytes_)):
                        class_names.append(item.decode('utf-8'))
                    elif isinstance(item, str):
                        class_names.append(item)
                    elif hasattr(item, 'item'):  # numpy scalar
                        class_names.append(str(item.item()))
                    else:
                        class_names.append(str(item))
            else:
                print(f"ERROR: Variable '{variable_name}' does not contain string data")
                print(f"Data type: {value.dtype}")
                sys.exit(1)
        elif isinstance(value, (list, tuple)):
            # Handle list/tuple of strings
            for item in value:
                if isinstance(item, (bytes, np.bytes_)):
                    class_names.append(item.decode('utf-8'))
                else:
                    class_names.append(str(item))
        else:
            print(f"ERROR: Variable '{variable_name}' is not a supported type for txt conversion")
            print(f"Type: {type(value)}")
            sys.exit(1)
        
        # Write class names to file, one per line
        with open(output_path, 'w', encoding='utf-8') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        
        print(f"✓ Saved {len(class_names)} class names to {output_path} (TXT format)")
        
    else:
        # Save all string variables to separate TXT files
        base_path = output_path.rsplit('.', 1)[0]
        saved_files = []
        
        for key, value in mat_data.items():
            class_names = []
            
            if isinstance(value, np.ndarray) and value.dtype.kind in ['U', 'S', 'O']:
                flat_array = value.flatten()
                for item in flat_array:
                    if isinstance(item, (bytes, np.bytes_)):
                        class_names.append(item.decode('utf-8'))
                    elif isinstance(item, str):
                        class_names.append(item)
                    elif hasattr(item, 'item'):
                        class_names.append(str(item.item()))
                    else:
                        class_names.append(str(item))
                
                if class_names:
                    txt_path = f"{base_path}_{key}.txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        for class_name in class_names:
                            f.write(f"{class_name}\n")
                    saved_files.append(txt_path)
            
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, (bytes, np.bytes_)):
                        class_names.append(item.decode('utf-8'))
                    else:
                        class_names.append(str(item))
                
                if class_names:
                    txt_path = f"{base_path}_{key}.txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        for class_name in class_names:
                            f.write(f"{class_name}\n")
                    saved_files.append(txt_path)
        
        if saved_files:
            print(f"✓ Saved {len(saved_files)} variables to TXT files:")
            for f in saved_files:
                print(f"  - {f}")
        else:
            print("ERROR: No string arrays found to save as TXT")
            sys.exit(1)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert .mat files to different formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_mat_file.py data.mat json
  python convert_mat_file.py data.mat npz -o output_data.npz
  python convert_mat_file.py data.mat hdf5
  python convert_mat_file.py data.mat csv -v my_variable
  python convert_mat_file.py data.mat pickle
  python convert_mat_file.py data.mat txt -v class_names
        """
    )
    
    parser.add_argument('input_file', help='Input .mat file path')
    parser.add_argument('format', choices=['json', 'npz', 'pickle', 'hdf5', 'csv', 'txt'],
                       help='Output format')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-v', '--variable', help='Specific variable name (for CSV and TXT formats)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Load the .mat file
    print(f"\nLoading {args.input_file}...")
    mat_data = load_mat_file(args.input_file)
    
    if not mat_data:
        print("ERROR: No variables found in .mat file")
        sys.exit(1)
    
    print(f"\nVariables found:")
    for key, value in mat_data.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  - {key}: {type(value).__name__}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input_file)[0]
        ext_map = {
            'json': '.json',
            'npz': '.npz',
            'pickle': '.pkl',
            'hdf5': '.h5',
            'csv': '.csv',
            'txt': '.txt'
        }
        output_path = base_name + ext_map[args.format]
    
    # Convert to the specified format
    print(f"\nConverting to {args.format.upper()} format...")
    
    if args.format == 'json':
        convert_to_json(mat_data, output_path)
    elif args.format == 'npz':
        convert_to_npz(mat_data, output_path)
    elif args.format == 'pickle':
        convert_to_pickle(mat_data, output_path)
    elif args.format == 'hdf5':
        convert_to_hdf5(mat_data, output_path)
    elif args.format == 'csv':
        convert_to_csv(mat_data, output_path, args.variable)
    elif args.format == 'txt':
        convert_to_txt(mat_data, output_path, args.variable)
    
    print("\n✓ Conversion complete!")


if __name__ == "__main__":
    main()
