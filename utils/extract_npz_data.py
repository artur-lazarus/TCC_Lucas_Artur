#!/usr/bin/env python3
"""
NPZ Data Extraction Utility

This script extracts and manipulates data from NumPy NPZ (compressed archive) files.
NPZ files can contain multiple named arrays in a single compressed file.

Supported operations:
- List contents and inspect arrays
- Extract specific arrays or all arrays
- Convert to different formats (JSON, CSV, HDF5, pickle, individual .npy files)
- Show detailed statistics and information
- Save extracted data with various options

Usage:
    python extract_npz_data.py <input.npz> [options]
    
Examples:
    # List contents of NPZ file
    python extract_npz_data.py data.npz --list
    
    # Extract specific array to CSV
    python extract_npz_data.py data.npz --extract array_name --format csv
    
    # Extract all arrays to individual .npy files
    python extract_npz_data.py data.npz --extract-all --format npy
    
    # Convert entire NPZ to JSON with statistics
    python extract_npz_data.py data.npz --format json --stats
    
    # Extract array with custom output name
    python extract_npz_data.py data.npz --extract weights --output model_weights.npy
"""

import sys
import os
import argparse
import numpy as np
import json
from pathlib import Path


def load_npz_file(filepath):
    """
    Load an NPZ file and return the data dictionary
    
    Args:
        filepath (str): Path to the NPZ file
        
    Returns:
        dict: Dictionary containing arrays from the NPZ file
    """
    try:
        npz_data = np.load(filepath)
        # Convert to regular dict for easier manipulation
        data_dict = {key: npz_data[key] for key in npz_data.files}
        npz_data.close()
        
        print(f"✓ Loaded NPZ file: {filepath}")
        print(f"  Contains {len(data_dict)} arrays")
        return data_dict
        
    except Exception as e:
        print(f"ERROR loading NPZ file: {e}")
        sys.exit(1)


def list_npz_contents(data_dict, show_stats=False):
    """
    List the contents of the NPZ file with detailed information
    
    Args:
        data_dict (dict): Dictionary of arrays from NPZ file
        show_stats (bool): Whether to show detailed statistics
    """
    print(f"\n{'='*60}")
    print("NPZ FILE CONTENTS")
    print(f"{'='*60}")
    
    total_size = 0
    
    for i, (key, array) in enumerate(data_dict.items(), 1):
        print(f"\n{i}. Array: '{key}'")
        print(f"   Shape: {array.shape}")
        print(f"   Dtype: {array.dtype}")
        print(f"   Size: {array.size:,} elements")
        
        # Calculate memory usage
        memory_mb = array.nbytes / (1024 * 1024)
        print(f"   Memory: {memory_mb:.2f} MB")
        total_size += memory_mb
        
        # Show data preview
        if array.size <= 20:
            print(f"   Data:\n{array}")
        else:
            print(f"   Data preview:")
            if array.ndim == 1:
                print(f"     First 5: {array[:5]}")
                print(f"     Last 5:  {array[-5:]}")
            elif array.ndim == 2:
                print(f"     Shape: {array.shape}")
                print(f"     First few rows/cols:\n{array[:3, :min(5, array.shape[1])]}")
                if array.shape[0] > 3:
                    print(f"     ...")
            else:
                print(f"     First 10 elements (flattened): {array.flat[:10]}")
        
        # Show statistics for numeric arrays
        if show_stats and np.issubdtype(array.dtype, np.number) and array.size > 0:
            print(f"   Statistics:")
            print(f"     Min: {np.min(array)}")
            print(f"     Max: {np.max(array)}")
            print(f"     Mean: {np.mean(array):.6f}")
            print(f"     Std: {np.std(array):.6f}")
            
            # Additional stats for larger arrays
            if array.size > 1:
                print(f"     Median: {np.median(array):.6f}")
                print(f"     25th percentile: {np.percentile(array, 25):.6f}")
                print(f"     75th percentile: {np.percentile(array, 75):.6f}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total arrays: {len(data_dict)}")
    print(f"Total memory: {total_size:.2f} MB")
    print(f"{'='*60}")


def extract_array(data_dict, array_name, output_path=None, format_type='npy'):
    """
    Extract a specific array from the NPZ file
    
    Args:
        data_dict (dict): Dictionary of arrays from NPZ file
        array_name (str): Name of the array to extract
        output_path (str): Output file path (optional)
        format_type (str): Output format ('npy', 'csv', 'json', 'txt')
    
    Returns:
        str: Path to the saved file
    """
    if array_name not in data_dict:
        print(f"ERROR: Array '{array_name}' not found in NPZ file")
        print(f"Available arrays: {list(data_dict.keys())}")
        sys.exit(1)
    
    array = data_dict[array_name]
    
    # Generate output path if not provided
    if output_path is None:
        ext_map = {
            'npy': '.npy',
            'csv': '.csv',
            'json': '.json',
            'txt': '.txt'
        }
        output_path = f"{array_name}{ext_map.get(format_type, '.npy')}"
    
    print(f"\nExtracting array '{array_name}'...")
    print(f"  Shape: {array.shape}")
    print(f"  Dtype: {array.dtype}")
    print(f"  Output format: {format_type.upper()}")
    
    try:
        if format_type == 'npy':
            np.save(output_path, array)
            print(f"✓ Saved as NumPy array: {output_path}")
            
        elif format_type == 'csv':
            if array.ndim > 2:
                print(f"WARNING: Array has {array.ndim} dimensions, flattening for CSV")
                array = array.reshape(-1, array.shape[-1]) if array.ndim > 2 else array
            
            try:
                import pandas as pd
                if array.ndim == 1:
                    df = pd.DataFrame({array_name: array})
                else:
                    df = pd.DataFrame(array)
                df.to_csv(output_path, index=False)
                print(f"✓ Saved as CSV: {output_path}")
            except ImportError:
                # Fallback to numpy savetxt
                np.savetxt(output_path, array, delimiter=',')
                print(f"✓ Saved as CSV (using numpy): {output_path}")
                
        elif format_type == 'json':
            # Convert numpy array to JSON-serializable format
            if np.issubdtype(array.dtype, np.complexfloating):
                json_data = {
                    'real': array.real.tolist(),
                    'imag': array.imag.tolist(),
                    'dtype': str(array.dtype),
                    'shape': array.shape
                }
            else:
                json_data = {
                    'data': array.tolist(),
                    'dtype': str(array.dtype),
                    'shape': array.shape
                }
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"✓ Saved as JSON: {output_path}")
            
        elif format_type == 'txt':
            if array.ndim > 2:
                array = array.flatten()
            np.savetxt(output_path, array, fmt='%.6f')
            print(f"✓ Saved as text: {output_path}")
            
        return output_path
        
    except Exception as e:
        print(f"ERROR saving array: {e}")
        sys.exit(1)


def extract_all_arrays(data_dict, output_dir=None, format_type='npy'):
    """
    Extract all arrays from the NPZ file
    
    Args:
        data_dict (dict): Dictionary of arrays from NPZ file
        output_dir (str): Output directory (optional)
        format_type (str): Output format for all arrays
    
    Returns:
        list: List of saved file paths
    """
    if output_dir is None:
        output_dir = "extracted_arrays"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExtracting all {len(data_dict)} arrays to '{output_dir}'...")
    print(f"Output format: {format_type.upper()}")
    
    saved_files = []
    
    for array_name, array in data_dict.items():
        ext_map = {
            'npy': '.npy',
            'csv': '.csv',
            'json': '.json',
            'txt': '.txt'
        }
        
        output_path = os.path.join(output_dir, f"{array_name}{ext_map.get(format_type, '.npy')}")
        
        try:
            extract_array({array_name: array}, array_name, output_path, format_type)
            saved_files.append(output_path)
        except Exception as e:
            print(f"WARNING: Failed to save '{array_name}': {e}")
    
    print(f"\n✓ Successfully extracted {len(saved_files)} arrays")
    return saved_files


def convert_npz_to_format(data_dict, output_path, format_type):
    """
    Convert entire NPZ file to a different format
    
    Args:
        data_dict (dict): Dictionary of arrays from NPZ file
        output_path (str): Output file path
        format_type (str): Target format ('json', 'hdf5', 'pickle')
    """
    print(f"\nConverting NPZ to {format_type.upper()} format...")
    
    try:
        if format_type == 'json':
            json_data = {}
            for key, array in data_dict.items():
                if np.issubdtype(array.dtype, np.complexfloating):
                    json_data[key] = {
                        'real': array.real.tolist(),
                        'imag': array.imag.tolist(),
                        'dtype': str(array.dtype),
                        'shape': array.shape
                    }
                else:
                    json_data[key] = {
                        'data': array.tolist(),
                        'dtype': str(array.dtype),
                        'shape': array.shape
                    }
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"✓ Saved as JSON: {output_path}")
            
        elif format_type == 'hdf5':
            try:
                import h5py
                with h5py.File(output_path, 'w') as f:
                    for key, array in data_dict.items():
                        f.create_dataset(key, data=array, compression='gzip')
                print(f"✓ Saved as HDF5: {output_path}")
            except ImportError:
                print("ERROR: h5py not installed. Install with: pip install h5py")
                sys.exit(1)
                
        elif format_type == 'pickle':
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✓ Saved as pickle: {output_path}")
            
        elif format_type == 'npz':
            # Re-save as NPZ (useful for compression or reorganization)
            np.savez_compressed(output_path, **data_dict)
            print(f"✓ Saved as compressed NPZ: {output_path}")
            
    except Exception as e:
        print(f"ERROR converting to {format_type}: {e}")
        sys.exit(1)


def save_info_report(data_dict, npz_path, output_path=None):
    """
    Save a detailed information report about the NPZ file
    
    Args:
        data_dict (dict): Dictionary of arrays from NPZ file
        npz_path (str): Original NPZ file path
        output_path (str): Output report file path
    """
    if output_path is None:
        base_name = Path(npz_path).stem
        output_path = f"{base_name}_info_report.txt"
    
    try:
        with open(output_path, 'w') as f:
            f.write("NPZ FILE INFORMATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Source file: {npz_path}\n")
            f.write(f"Generated on: {np.datetime64('now')}\n\n")
            
            f.write(f"Total arrays: {len(data_dict)}\n")
            
            total_size = 0
            for key, array in data_dict.items():
                memory_mb = array.nbytes / (1024 * 1024)
                total_size += memory_mb
            
            f.write(f"Total memory: {total_size:.2f} MB\n\n")
            
            f.write("ARRAY DETAILS:\n")
            f.write("-" * 40 + "\n")
            
            for i, (key, array) in enumerate(data_dict.items(), 1):
                f.write(f"\n{i}. Array: '{key}'\n")
                f.write(f"   Shape: {array.shape}\n")
                f.write(f"   Dtype: {array.dtype}\n")
                f.write(f"   Size: {array.size:,} elements\n")
                f.write(f"   Memory: {array.nbytes / (1024 * 1024):.2f} MB\n")
                
                if np.issubdtype(array.dtype, np.number) and array.size > 0:
                    f.write(f"   Statistics:\n")
                    f.write(f"     Min: {np.min(array)}\n")
                    f.write(f"     Max: {np.max(array)}\n")
                    f.write(f"     Mean: {np.mean(array):.6f}\n")
                    f.write(f"     Std: {np.std(array):.6f}\n")
        
        print(f"✓ Information report saved: {output_path}")
        
    except Exception as e:
        print(f"ERROR saving report: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract and manipulate data from NumPy NPZ files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List contents
  python extract_npz_data.py data.npz --list
  
  # List with detailed statistics
  python extract_npz_data.py data.npz --list --stats
  
  # Extract specific array
  python extract_npz_data.py data.npz --extract weights
  
  # Extract array to CSV
  python extract_npz_data.py data.npz --extract features --format csv
  
  # Extract all arrays as individual .npy files
  python extract_npz_data.py data.npz --extract-all
  
  # Convert entire NPZ to JSON
  python extract_npz_data.py data.npz --format json --output data.json
  
  # Generate information report
  python extract_npz_data.py data.npz --info-report
        """
    )
    
    parser.add_argument('input_file', help='Input NPZ file path')
    
    # Main operations (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--list', '-l', action='store_true',
                      help='List contents of the NPZ file')
    group.add_argument('--extract', '-e', type=str,
                      help='Extract specific array by name')
    group.add_argument('--extract-all', '-a', action='store_true',
                      help='Extract all arrays to separate files')
    group.add_argument('--info-report', '-r', action='store_true',
                      help='Generate detailed information report')
    
    # Options
    parser.add_argument('--format', '-f', 
                       choices=['npy', 'csv', 'json', 'txt', 'hdf5', 'pickle', 'npz'],
                       default='npy',
                       help='Output format (default: npy)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file/directory path')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='Show detailed statistics (with --list)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Load NPZ file
    print(f"Loading NPZ file: {args.input_file}")
    data_dict = load_npz_file(args.input_file)
    
    if not data_dict:
        print("ERROR: No arrays found in NPZ file")
        sys.exit(1)
    
    # Execute requested operation
    if args.list:
        list_npz_contents(data_dict, show_stats=args.stats)
        
    elif args.extract:
        extract_array(data_dict, args.extract, args.output, args.format)
        
    elif args.extract_all:
        extract_all_arrays(data_dict, args.output, args.format)
        
    elif args.info_report:
        save_info_report(data_dict, args.input_file, args.output)
        
    else:
        # Default: convert entire file to specified format
        if args.output is None:
            base_name = Path(args.input_file).stem
            ext_map = {
                'json': '.json',
                'hdf5': '.h5',
                'pickle': '.pkl',
                'npz': '.npz'
            }
            args.output = f"{base_name}_converted{ext_map.get(args.format, '.npz')}"
        
        if args.format in ['json', 'hdf5', 'pickle', 'npz']:
            convert_npz_to_format(data_dict, args.output, args.format)
        else:
            print("ERROR: For full file conversion, use formats: json, hdf5, pickle, npz")
            print("Use --extract-all for individual array extraction in other formats")
            sys.exit(1)
    
    print("\n✓ Operation completed successfully!")


if __name__ == "__main__":
    main()
