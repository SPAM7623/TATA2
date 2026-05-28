# Part 1: File Upload & Handling Guide

## Overview

Part 1 now includes robust file upload and validation functionality. You can upload training and test data files in multiple ways.

---

## How to Upload Files

### **Option 1: Local Files (Recommended)**

Place `train.csv` and `test.csv` in the same directory as the script.

```bash
cd /home/user/TATA2/

# Verify files are present
ls -lh train.csv test.csv

# Run the workflow
python WORKFLOW_ORCHESTRATOR.py
```

**Expected Output**:
```
FILE UPLOAD & VALIDATION
═════════════════════════
✓ train.csv exists: True
✓ test.csv exists: True
✓ Using local files
```

---

### **Option 2: Custom File Paths**

```bash
# Train and test in different locations
python part1_industrial_eda.py --train /path/to/train.csv --test /path/to/test.csv

# Example with relative paths
python part1_industrial_eda.py --train ../data/train.csv --test ../data/test.csv

# Example with full paths
python part1_industrial_eda.py --train /home/user/data/train.csv --test /home/user/data/test.csv
```

**Expected Output**:
```
✓ Using custom train path: /path/to/train.csv
✓ Using custom test path: /path/to/test.csv
✓ Data validation successful!
✓ Files copied to workflow directory
```

---

### **Option 3: File Dialog (Interactive)**

Uncomment the file dialog code in `part1_industrial_eda.py`:

```python
from tkinter import filedialog
import tkinter as tk

root = tk.Tk()
root.withdraw()

print("Select train.csv file:")
train_path = filedialog.askopenfilename(
    title="Select Training Data",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

print("Select test.csv file:")
test_path = filedialog.askopenfilename(
    title="Select Test Data",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

root.destroy()
```

Then run:
```bash
python part1_industrial_eda.py --upload
```

This opens a file browser where you can select your files.

---

### **Option 4: Upload During Script Execution**

The script will guide you through the upload process:

```bash
python part1_industrial_eda.py
```

If files aren't found, you'll see:

```
FILE UPLOAD & VALIDATION
═════════════════════════
Local file check:
  ✓ train.csv exists: False
  ✓ test.csv exists: False

⚠ Files not found. Follow these steps:

OPTION 1: Upload Files to This Directory
  1. Place 'train.csv' in current directory
  2. Place 'test.csv' in current directory
  3. Run this script again

OPTION 2: Specify Custom File Paths
  python part1_industrial_eda.py --train <path> --test <path>

OPTION 3: Use File Selection Dialog
  Uncomment the file dialog code below:
  [shows code snippet]
```

---

## File Requirements

### **Training Data Format (train.csv)**

```
CoilID,X1,X2,X3,...,X49,Y
1,45.2,32.1,67.8,...,21.3,0
2,48.5,35.2,69.1,...,23.4,1
3,46.1,33.8,68.2,...,22.1,0
...
```

**Requirements**:
- Dimensions: At least 1000 rows × 51 columns
- Columns: `CoilID` (unique identifier), `X1-X49` (features), `Y` (target: 0 or 1)
- Format: CSV (comma-separated values)
- No missing values recommended
- Target values: 0 (No Defect) or 1 (Defect)

### **Test Data Format (test.csv)**

```
CoilID,X1,X2,X3,...,X49
1001,45.2,32.1,67.8,...,21.3
1002,48.5,35.2,69.1,...,23.4
1003,46.1,33.8,68.2,...,22.1
...
```

**Requirements**:
- Dimensions: At least 100 rows × 50 columns
- Columns: `CoilID` (unique identifier), `X1-X49` (features)
- **Note**: No `Y` column (predictions will be made)
- Format: CSV (comma-separated values)
- No missing values recommended

---

## Data Validation

The script automatically validates uploaded files for:

### **Validation Checks**

```
DATA VALIDATION
═══════════════════════

Reading training data...
  ✓ Shape: (1352, 51)
  ✓ Columns: ['CoilID', 'X1', 'X2', 'X3', 'X4', ...]
  ✓ All expected columns present
  ✓ Target values: [0, 1]
  ✓ Class distribution: {0: 1230, 1: 122}

Reading test data...
  ✓ Shape: (339, 50)

✓ Data validation successful!
```

### **What Gets Validated**

1. **File Existence**: Are both CSV files present?
2. **File Format**: Can files be read as CSV?
3. **Column Names**: Are expected columns present (CoilID, X1-X49, Y)?
4. **Data Types**: Are features numeric? Is target binary?
5. **Target Values**: Does Y contain only 0 and 1?
6. **Shape**: Do files have reasonable dimensions?
7. **Missing Values**: Warning if NaN values detected

---

## Python API

### **Direct Import**

```python
from part1_industrial_eda import (
    upload_and_validate_files,
    validate_data_files,
    copy_files_to_workflow,
    IndustrialEDA
)

# Method 1: Let IndustrialEDA handle everything
eda = IndustrialEDA(
    train_path='train.csv',
    test_path='test.csv',
    auto_upload=True  # Enables file handling
)

# Method 2: Manual handling
train_path, test_path = upload_and_validate_files('train.csv', 'test.csv')
train_df, test_df = validate_data_files(train_path, test_path)
train_path, test_path = copy_files_to_workflow(train_path, test_path)
```

### **Custom File Paths**

```python
from part1_industrial_eda import IndustrialEDA

# Use files from different locations
eda = IndustrialEDA(
    train_path='/data/my_train.csv',
    test_path='/data/my_test.csv',
    auto_upload=True
)

eda.run_complete_eda()
```

### **Conditional File Handling**

```python
from part1_industrial_eda import IndustrialEDA
import os

# Check if files exist
if os.path.exists('train.csv') and os.path.exists('test.csv'):
    eda = IndustrialEDA(auto_upload=False)  # Skip validation
else:
    eda = IndustrialEDA(auto_upload=True)   # Enable upload handling
```

---

## Error Handling

### **Common Errors**

**Error**: `FileNotFoundError: train.csv not found`

**Solution**: 
```bash
# Option 1: Upload file to current directory
cp /path/to/train.csv .

# Option 2: Use custom path
python part1_industrial_eda.py --train /path/to/train.csv --test /path/to/test.csv

# Option 3: Use interactive dialog
python part1_industrial_eda.py --upload
```

**Error**: `ValueError: Training data missing 'Y' column`

**Solution**:
- Training CSV must have a `Y` column with target values (0 or 1)
- Test CSV should NOT have `Y` column

**Error**: `Data validation failed: unexpected columns`

**Solution**:
- Ensure training file has columns: `CoilID`, `X1` through `X49`, `Y`
- Ensure test file has columns: `CoilID`, `X1` through `X49`

---

## Complete Workflow with Upload

### **Step-by-Step**

```bash
# Step 1: Navigate to project directory
cd /home/user/TATA2/

# Step 2: Upload your CSV files to this directory
# (copy train.csv and test.csv here)

# Step 3: Run the complete workflow
python WORKFLOW_ORCHESTRATOR.py

# The workflow will:
# 1. Check for train.csv and test.csv
# 2. Validate file structure
# 3. Proceed with analysis if valid
# 4. Generate 24+ visualizations and 8 insights reports
```

### **With Custom Paths**

```bash
# If your files are in different locations:
python WORKFLOW_ORCHESTRATOR.py
# (WORKFLOW_ORCHESTRATOR will use default paths from Part 1)

# OR run Part 1 directly with custom paths:
python part1_industrial_eda.py \
    --train /data/training.csv \
    --test /data/testing.csv
```

---

## Automating Upload in Deployment

### **Production Setup**

```python
import os
import shutil
from part1_industrial_eda import IndustrialEDA

# Function to auto-upload files from a source directory
def setup_workflow(source_dir='/incoming/data'):
    """Automatically copy files from upload directory"""

    print("Setting up workflow...")

    # Copy files from upload location
    if os.path.exists(f'{source_dir}/train.csv'):
        shutil.copy(f'{source_dir}/train.csv', './train.csv')
        print("✓ Training file copied")

    if os.path.exists(f'{source_dir}/test.csv'):
        shutil.copy(f'{source_dir}/test.csv', './test.csv')
        print("✓ Test file copied")

    # Initialize EDA with auto validation
    eda = IndustrialEDA(auto_upload=True)
    return eda

# Usage
if __name__ == "__main__":
    eda = setup_workflow()
    eda.run_complete_eda()
```

---

## Monitoring Upload Status

### **Verbose Output**

```python
class IndustrialEDA:
    def __init__(self, train_path='train.csv', test_path='test.csv', auto_upload=True):
        if auto_upload:
            print("\n[UPLOAD] Starting file upload process...")
            self.train_path, self.test_path = upload_and_validate_files(...)
            print("[UPLOAD] File validation in progress...")
            self.train_df, self.test_df = validate_data_files(...)
            print("[UPLOAD] Copying files to workflow directory...")
            self.train_path, self.test_path = copy_files_to_workflow(...)
            print("[UPLOAD] ✓ All files ready")
```

---

## Summary

| Method | Complexity | Use Case |
|--------|-----------|----------|
| **Local Files** | Simplest | Files in project directory |
| **Custom Paths** | Low | Files in different location |
| **File Dialog** | Low | Interactive selection |
| **Programmatic** | Medium | Custom validation logic |
| **Automated** | Medium | Production deployments |

---

## Next Steps

1. **Upload your files** (using any method above)
2. **Run the workflow**:
   ```bash
   python WORKFLOW_ORCHESTRATOR.py
   ```
3. **Review outputs**:
   - 8 insights reports (PART1-8_INSIGHTS.txt)
   - 24+ visualizations (PNG files)
   - Engineered features (X_engineered.csv)
4. **Deploy the model** (see production guide)

✅ **File upload handling is now integrated into Part 1 and the entire workflow!**
