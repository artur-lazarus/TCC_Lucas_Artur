#!/usr/bin/env python3
"""
Utility script to view and inspect .mat files (MATLAB data files)

Usage:
    python view_mat_file.py <path_to_mat_file>
"""

import sys
import numpy as np

def view_mat_file(filepath):
    """
    Load and display the contents of a .mat file
    
    Args:
        filepath: Path to the .mat file
    """
    print(f"\n{'='*60}")
    print(f"Viewing .mat file: {filepath}")
    print(f"{'='*60}\n")
    
    try:
        # Try loading with scipy.io (for MATLAB v7 and earlier)
        import scipy.io
        mat_contents = scipy.io.loadmat(filepath)
        
        print("✓ Successfully loaded with scipy.io")
        print(f"\nFile contains {len(mat_contents)} variables:\n")
        
        # Display each variable
        for key, value in mat_contents.items():
            # Skip metadata variables
            if key.startswith('__'):
                continue
            
            print(f"Variable: {key}")
            print(f"  Type: {type(value).__name__}")
            
            if isinstance(value, np.ndarray):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                print(f"  Size: {value.size} elements")
                
                # Show a preview of the data
                if value.size <= 20:
                    print(f"  Data:\n{value}")
                else:
                    print(f"  Data preview (first few elements):")
                    if value.ndim == 1:
                        print(f"    {value[:5]}...")
                    elif value.ndim == 2:
                        print(f"    {value[:3, :min(5, value.shape[1])]}")
                        if value.shape[0] > 3:
                            print(f"    ...")
                    else:
                        print(f"    {value.flat[:5]}...")
                
                # Show statistics for numeric arrays
                if np.issubdtype(value.dtype, np.number):
                    print(f"  Statistics:")
                    print(f"    Min: {np.min(value)}")
                    print(f"    Max: {np.max(value)}")
                    print(f"    Mean: {np.mean(value):.4f}")
                    print(f"    Std: {np.std(value):.4f}")
            else:
                print(f"  Value: {value}")
            
            print()
        
    except NotImplementedError:
        # If scipy.io fails, try h5py for MATLAB v7.3 files
        print("scipy.io couldn't load the file. Trying h5py (MATLAB v7.3 format)...\n")
        
        try:
            import h5py
            
            with h5py.File(filepath, 'r') as f:
                print("✓ Successfully loaded with h5py")
                print(f"\nFile contains {len(f.keys())} variables:\n")
                
                def print_h5_item(name, obj):
                    """Helper function to print h5py objects"""
                    if isinstance(obj, h5py.Dataset):
                        print(f"Dataset: {name}")
                        print(f"  Shape: {obj.shape}")
                        print(f"  Dtype: {obj.dtype}")
                        
                        # Load and show preview
                        data = obj[()]
                        if data.size <= 20:
                            print(f"  Data:\n{data}")
                        else:
                            print(f"  Data preview: {data.flat[:5]}...")
                        print()
                    elif isinstance(obj, h5py.Group):
                        print(f"Group: {name}")
                        print(f"  Contains {len(obj)} items")
                        print()
                
                f.visititems(print_h5_item)
                
        except ImportError:
            print("ERROR: h5py not installed. Install with: pip install h5py")
            return
        except Exception as e:
            print(f"ERROR loading with h5py: {e}")
            return
            
    except ImportError:
        print("ERROR: scipy not installed. Install with: pip install scipy")
        return
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        return


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python view_mat_file.py <path_to_mat_file>")
        print("\nExample:")
        print("  python view_mat_file.py data/my_data.mat")
        sys.exit(1)
    
    filepath = sys.argv[1]
    view_mat_file(filepath)


if __name__ == "__main__":
    main()
