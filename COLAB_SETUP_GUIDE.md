# Google Colab Setup Guide

## Complete Instructions for Running in Google Colab

---

## 📋 Prerequisites

- Google Account
- Access to Google Colab (colab.research.google.com)
- Your train.csv and test.csv files ready to upload

---

## 🚀 Quick Start (5 Steps)

### **Step 1: Create New Colab Notebook**

1. Go to https://colab.research.google.com
2. Click "New notebook"
3. Rename it to "Alpha Defect Prediction"

### **Step 2: Clone the Repository**

```python
# In first cell, run:
!git clone https://github.com/spam7623/TATA2.git
%cd TATA2
```

### **Step 3: Install Dependencies**

```python
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap imbalanced-learn scipy
```

### **Step 4: Upload Your Data Files**

```python
from google.colab import files

print("📁 Upload train.csv:")
train_files = files.upload()

print("\n📁 Upload test.csv:")
test_files = files.upload()

print("\n✓ Files uploaded successfully")
print(f"  Train: {list(train_files.keys())}")
print(f"  Test: {list(test_files.keys())}")
```

### **Step 5: Run the Workflow**

```python
# Run complete workflow
exec(open('WORKFLOW_ORCHESTRATOR.py').read())
```

---

## 📝 Complete Colab Code (Copy-Paste Ready)

```python
# ============================================================
# STEP 1: SETUP & DEPENDENCIES
# ============================================================

# Clone repository
!git clone https://github.com/spam7623/TATA2.git
%cd TATA2

# Install dependencies
!pip install -q pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap imbalanced-learn scipy

print("✓ Setup complete!")

# ============================================================
# STEP 2: UPLOAD FILES
# ============================================================

from google.colab import files

print("\n" + "="*70)
print("FILE UPLOAD")
print("="*70)

print("\n📁 UPLOAD TRAIN.CSV")
print("Click 'Choose Files' button when prompted")
train_upload = files.upload()

if train_upload:
    train_file = list(train_upload.keys())[0]
    print(f"✓ Uploaded: {train_file}")
else:
    raise FileNotFoundError("train.csv not uploaded")

print("\n📁 UPLOAD TEST.CSV")
print("Click 'Choose Files' button when prompted")
test_upload = files.upload()

if test_upload:
    test_file = list(test_upload.keys())[0]
    print(f"✓ Uploaded: {test_file}")
else:
    raise FileNotFoundError("test.csv not uploaded")

# ============================================================
# STEP 3: RUN WORKFLOW
# ============================================================

print("\n" + "="*70)
print("RUNNING WORKFLOW")
print("="*70)

# Run the complete workflow
exec(open('WORKFLOW_ORCHESTRATOR.py').read())

print("\n✓ Workflow complete!")
print("\nOutputs generated:")
print("  - 8 insights reports (PART1-8_INSIGHTS.txt)")
print("  - 24+ visualizations (PNG files)")
print("  - Engineered features (X_engineered.csv)")
print("  - Deployment config (workflow_insights.json)")

# ============================================================
# STEP 4: DOWNLOAD RESULTS (Optional)
# ============================================================

print("\n" + "="*70)
print("DOWNLOAD RESULTS")
print("="*70)

from google.colab import files
import os

# Create a folder for downloads
os.makedirs('results', exist_ok=True)

# Copy all outputs
!cp *.txt results/ 2>/dev/null || true
!cp *.png results/ 2>/dev/null || true
!cp *.csv results/ 2>/dev/null || true
!cp *.json results/ 2>/dev/null || true

print("\nDownloading results...")
files.download('results/results.zip')

print("✓ Results downloaded!")
```

---

## 🎯 Alternative: Run Individual Parts

If you want more control, run parts individually:

```python
# Part 1: EDA
exec(open('part1_industrial_eda.py').read())

# Part 2: Baseline
exec(open('part2_baseline_model.py').read())

# Part 3: Error Analysis
exec(open('part3_shap_error_analysis.py').read())

# ... continue with other parts
```

---

## 📊 File Upload Methods in Colab

### **Method 1: Google Colab files.upload() (Easiest)**

```python
from google.colab import files

# Upload train.csv
print("Upload train.csv:")
uploaded = files.upload()
train_file = list(uploaded.keys())[0]

# Upload test.csv
print("Upload test.csv:")
uploaded = files.upload()
test_file = list(uploaded.keys())[0]

print(f"✓ train.csv: {train_file}")
print(f"✓ test.csv: {test_file}")
```

**Output**: Button appears to browse and upload files

### **Method 2: Mount Google Drive**

```python
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Files in your Drive: /content/drive/My Drive/your_files/
train_file = '/content/drive/My Drive/data/train.csv'
test_file = '/content/drive/My Drive/data/test.csv'

print(f"✓ train.csv: {train_file}")
print(f"✓ test.csv: {test_file}")
```

**Advantage**: Access files from Google Drive

### **Method 3: Download from URL**

```python
import urllib.request

# If files are hosted online
url_train = 'https://example.com/train.csv'
url_test = 'https://example.com/test.csv'

urllib.request.urlretrieve(url_train, 'train.csv')
urllib.request.urlretrieve(url_test, 'test.csv')

print("✓ Files downloaded")
```

**Advantage**: No manual upload needed

---

## 💾 Saving Results from Colab

### **Method 1: Download Zip File**

```python
import shutil
from google.colab import files

# Create output folder
os.makedirs('output', exist_ok=True)

# Copy all important files
!cp *.txt output/ 2>/dev/null || true
!cp *.png output/ 2>/dev/null || true
!cp *.csv output/ 2>/dev/null || true
!cp *.json output/ 2>/dev/null || true

# Create zip
shutil.make_archive('workflow_results', 'zip', 'output')

# Download
files.download('workflow_results.zip')
print("✓ Downloaded workflow_results.zip")
```

### **Method 2: Save to Google Drive**

```python
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Copy results to Drive
!mkdir -p '/content/drive/My Drive/workflow_results'
!cp *.txt '/content/drive/My Drive/workflow_results/' 2>/dev/null || true
!cp *.png '/content/drive/My Drive/workflow_results/' 2>/dev/null || true
!cp *.csv '/content/drive/My Drive/workflow_results/' 2>/dev/null || true
!cp *.json '/content/drive/My Drive/workflow_results/' 2>/dev/null || true

print("✓ Results saved to Google Drive")
```

### **Method 3: Display Individual Files**

```python
# View insights
!head -50 PART1_INSIGHTS.txt

# View visualizations
from IPython.display import Image, display

display(Image('01_class_distribution.png'))
display(Image('13_probability_analysis.png'))
```

---

## 🐛 Troubleshooting in Colab

### **Issue: "ModuleNotFoundError: No module named 'X'"**

**Solution**: Install missing package
```python
!pip install package_name
```

### **Issue: "FileNotFoundError: train.csv not found"**

**Solution 1**: Re-upload the file
```python
from google.colab import files
files.upload()
```

**Solution 2**: Check current directory
```python
import os
print(os.listdir())  # List files in current directory
```

### **Issue: "Memory error" or "Runtime crashed"**

**Solution**: Restart runtime and use smaller parts
```python
# Restart runtime
!kill -9 -1  # Or use Runtime > Restart runtime

# Or split workflow
exec(open('part1_industrial_eda.py').read())
# ... wait for results
exec(open('part2_baseline_model.py').read())
```

### **Issue: "SHAP is slow"**

**Solution**: This is normal - SHAP computation is intensive. Let it run.
- Part 3 typically takes 5-10 minutes
- Be patient, don't interrupt

### **Issue: "Git clone failed"**

**Solution**: Use direct file upload instead
```python
from google.colab import files
print("Upload all .py files")
files.upload()
```

---

## 📋 Complete Colab Workflow Checklist

- [ ] 1. Create new Colab notebook
- [ ] 2. Clone repository (or upload files directly)
- [ ] 3. Install dependencies
- [ ] 4. Upload train.csv
- [ ] 5. Upload test.csv
- [ ] 6. Run complete workflow (or run parts sequentially)
- [ ] 7. Review insights reports
- [ ] 8. Download/save results
- [ ] 9. Deploy model (use workflow_insights.json)

---

## ⚡ Performance Tips for Colab

1. **Use GPU/TPU** (optional but faster):
   - Runtime > Change runtime type > GPU

2. **Close other tabs** to free memory

3. **Run parts sequentially** if memory is limited
   ```python
   exec(open('part1_industrial_eda.py').read())
   # Let it finish, then:
   exec(open('part2_baseline_model.py').read())
   ```

4. **Increase session timeout**:
   ```python
   from google.colab import _message
   _message.disable_colab_checks()  # Prevents timeout
   ```

5. **Save progress frequently**:
   ```python
   !cp *.csv /content/drive/MyDrive/backup/ 2>/dev/null || true
   ```

---

## 🎓 Recommended Colab Cell Order

### **Cell 1: Setup**
```python
!git clone https://github.com/spam7623/TATA2.git
%cd TATA2
!pip install -q pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap imbalanced-learn scipy
```

### **Cell 2: Upload Files**
```python
from google.colab import files
print("Upload train.csv"); train_upload = files.upload()
print("Upload test.csv"); test_upload = files.upload()
```

### **Cell 3: Run Workflow**
```python
exec(open('WORKFLOW_ORCHESTRATOR.py').read())
```

### **Cell 4: View Results** (Optional)
```python
!head -100 PART1_INSIGHTS.txt
from IPython.display import Image, display
display(Image('01_class_distribution.png'))
```

### **Cell 5: Download Results** (Optional)
```python
from google.colab import files
!zip -r results.zip *.txt *.png *.csv *.json
files.download('results.zip')
```

---

## 📱 Common Colab Commands

```python
# List files
!ls -lh

# Check directory
!pwd

# Check available disk space
!df -h

# List GPU/CPU
!nvidia-smi  # GPU info
!cat /proc/cpuinfo | grep processor | wc -l  # CPU count

# Install package
!pip install package_name

# Run Python script
!python script.py

# Run Python code from file
exec(open('script.py').read())

# Create directory
!mkdir directory_name

# Copy files
!cp source.txt destination.txt

# Create zip
!zip -r archive.zip folder/

# Download
from google.colab import files
files.download('filename')

# Upload
files.upload()

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

---

## ✨ Summary

**Quickest Colab Setup**:

1. Open https://colab.research.google.com
2. Paste this code:

```python
!git clone https://github.com/spam7623/TATA2.git
%cd TATA2
!pip install -q pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap imbalanced-learn scipy

from google.colab import files
print("Upload train.csv:"); files.upload()
print("Upload test.csv:"); files.upload()

exec(open('WORKFLOW_ORCHESTRATOR.py').read())
```

3. Run and wait 30-45 minutes
4. Download results

**Done!** ✅

---

## 🆘 Need Help?

- Check FILE_UPLOAD_GUIDE.md for general upload instructions
- Check README.md for workflow overview
- Check EXECUTION_CHECKLIST.md for detailed steps

**Pro Tip**: Save this notebook to Google Drive for future use:
```python
# In last cell:
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/TATA2 '/content/drive/My Drive/TATA2_backup'
print("✓ Backup saved to Drive")
```
