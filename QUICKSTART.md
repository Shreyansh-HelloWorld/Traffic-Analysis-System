# 🚀 Quick Start Guide

## Complete Step-by-Step Deployment from Google Colab to Local Streamlit

### 📥 Step 1: Download Your Model Files from Colab

In your Google Colab notebook, run:

```python
from google.colab import files

# Download ANPR model (number plate detection)
files.download('/content/best2.pt')

# Download Vehicle Classification model
files.download('/content/best.pt')
```

Save these files - you'll need them!

---

### 💻 Step 2: Setup on Your Local PC (Windows)

#### 2.1: Create Project Folder
1. Create a new folder: `C:\traffic-analysis-system`
2. Download all project files into this folder:
   - `app.py`
   - `anpr_module.py`
   - `vehicle_classifier.py`
   - `utils.py`
   - `requirements.txt`
   - `README.md`

#### 2.2: Create Models Folder
1. Inside your project folder, create a new folder called `models`
2. Move your downloaded model files into this folder:
   ```
   C:\traffic-analysis-system\
   └── models\
       ├── best2.pt
       └── best.pt
   ```

#### 2.3: Open VS Code
1. Open VS Code
2. File → Open Folder → Select `C:\traffic-analysis-system`

#### 2.4: Open Terminal in VS Code
- Press `` Ctrl + ` `` (backtick) to open terminal
- Or: Terminal → New Terminal

---

### 🐍 Step 3: Setup Python Environment

#### 3.1: Check Python Installation
In VS Code terminal:
```bash
python --version
```

If not installed, download from: https://www.python.org/downloads/
- Install Python 3.9 or higher
- ✅ Check "Add Python to PATH" during installation

#### 3.2: Create Virtual Environment
```bash
python -m venv venv
```

#### 3.3: Activate Virtual Environment
```bash
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal line.

---

### 📦 Step 4: Install Dependencies

With virtual environment activated:
```bash
pip install -r requirements.txt
```

⏳ This takes 5-10 minutes. Wait for it to complete!

**If you get errors**, try:
```bash
pip uninstall paddleocr paddlepaddle
pip install paddleocr==2.7.3 paddlepaddle==2.6.2 "numpy<2.0.0"
pip install streamlit ultralytics opencv-python pandas pillow
```

---

### 🎯 Step 5: Run the Application

In VS Code terminal (with venv activated):
```bash
streamlit run app.py
```

✨ Your default browser should open automatically to: `http://localhost:8501`

If not, manually open your browser and go to: `http://localhost:8501`

---

### 🖼️ Step 6: Use the Application

1. **Upload Image**: Click "Browse files" button
2. **Select Image**: Choose a traffic image with vehicles
3. **Configure** (Optional): Adjust confidence thresholds in sidebar
4. **Analyze**: Click "🔍 Analyze Traffic" button
5. **View Results**: 
   - See annotated image
   - Browse gallery view
   - Check table view
   - Download results

---

### 🛑 Step 7: Stop the Application

In VS Code terminal, press: `Ctrl + C`

---

## 🔧 Troubleshooting Common Issues

### Issue 1: "Python not found"
**Solution:**
- Install Python from python.org
- Restart VS Code
- Make sure "Add to PATH" was checked

### Issue 2: "streamlit: command not found"
**Solution:**
```bash
pip install streamlit
```

### Issue 3: Models Not Loading
**Solution:**
- Check models are in `models/` folder
- Check filenames: `best2.pt` and `best.pt`
- Check file sizes (should be >1MB each)

### Issue 4: Virtual Environment Not Activating
**Solution:**
```bash
# Delete old venv
rmdir /s venv

# Create new venv
python -m venv venv

# Activate
venv\Scripts\activate
```

### Issue 5: Import Errors
**Solution:**
```bash
pip uninstall -y ultralytics opencv-python paddleocr paddlepaddle
pip install -r requirements.txt
```

---

## 📁 Final Folder Structure

Your folder should look like this:

```
C:\traffic-analysis-system\
│
├── venv\                    (created by python -m venv venv)
│
├── models\
│   ├── best2.pt            (ANPR model)
│   └── best.pt             (Vehicle classification model)
│
├── app.py
├── anpr_module.py
├── vehicle_classifier.py
├── utils.py
├── batch_process.py
├── requirements.txt
├── README.md
└── QUICKSTART.md           (this file)
```

---

## 🎓 Alternative: macOS/Linux Setup

### Terminal Commands for Mac/Linux:

```bash
# Navigate to project folder
cd ~/traffic-analysis-system

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

---

## 🚀 Batch Processing (Optional)

To process multiple images at once:

```bash
python batch_process.py --input "path/to/images" --output "path/to/results"
```

Example:
```bash
python batch_process.py --input "C:\images" --output "C:\results"
```

---

## 💡 Tips for Best Results

1. **Image Quality**: Use clear, high-resolution images
2. **Lighting**: Better lighting = better detection
3. **Angle**: Front/rear views work best
4. **Distance**: Plates should be clearly visible
5. **Test First**: Try with 1-2 images before batch processing

---

## 📞 Quick Reference Commands

```bash
# Activate environment
venv\Scripts\activate           # Windows
source venv/bin/activate        # Mac/Linux

# Run app
streamlit run app.py

# Stop app
Ctrl + C

# Deactivate environment
deactivate
```

---

## ✅ Checklist Before Running

- [ ] Python 3.9+ installed
- [ ] VS Code installed
- [ ] Project folder created
- [ ] All .py files downloaded
- [ ] Models in `models/` folder
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Terminal in project folder

---

**Ready to go!** 🎉

Run: `streamlit run app.py`

Your traffic analysis system should now be running locally!

---

**Need Help?**
- Check README.md for detailed documentation
- Review error messages carefully
- Ensure all files are in correct locations
- Verify model files are not corrupted
