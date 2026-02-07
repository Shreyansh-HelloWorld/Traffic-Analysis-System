# 🚦 Intelligent Traffic Analysis System

Complete ANPR (Automatic Number Plate Recognition) + Vehicle Classification system with Streamlit deployment. Supports both **image and video** analysis.

## 📋 Features

- **Vehicle Detection & Classification**: Detects and classifies vehicles into 2W, 3W, 4W, and HMV
- **Number Plate Detection**: Locates vehicle number plates in images and videos
- **OCR with Smart Post-Processing**: Reads and validates Indian vehicle number plates with research-backed corrections
- **Multi-Method Preprocessing**: Uses 5 different preprocessing techniques for better accuracy
- **🎬 Video Analysis**: Frame-by-frame processing with automatic plate deduplication
- **Interactive Web Interface**: User-friendly Streamlit interface with Image / Video mode toggle
- **Detailed Results**: View results in gallery, table, or export formats
- **Real-time Statistics**: Get instant insights about detected vehicles

## 📁 Project Structure

```
traffic-analysis-system/
├── app.py                    # Main Streamlit application
├── anpr_module.py           # ANPR detection and OCR module
├── vehicle_classifier.py    # Vehicle classification module
├── utils.py                 # Utility functions (visualization, data handling)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── models/                 # Model weights directory
│   ├── best2.pt           # YOLO weights for plate detection
│   └── best.pt            # YOLO weights for vehicle classification
└── sample_images/         # Sample test images (optional)
```

## 🚀 Installation Guide

### Step 1: Prerequisites

Ensure you have Python 3.9 or higher installed:
```bash
python --version
```

### Step 2: Clone/Download the Project

Download all the files to a folder on your PC:
- `app.py`
- `anpr_module.py`
- `vehicle_classifier.py`
- `utils.py`
- `requirements.txt`
- `README.md`

### Step 3: Create Models Directory

Create a folder named `models` in your project directory and place your YOLO model weights:
```
traffic-analysis-system/
└── models/
    ├── best2.pt    # Your ANPR model
    └── best.pt     # Your vehicle classification model
```

### Step 4: Create Virtual Environment (Recommended)

Open a terminal/command prompt in your project directory:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 5: Install Dependencies

With your virtual environment activated:
```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (web framework)
- Ultralytics (YOLO models)
- OpenCV (image processing)
- PaddleOCR (OCR engine)
- PaddlePaddle (OCR backend)
- NumPy (numerical operations)
- Pandas (data handling)
- Pillow (image handling)

**Note:** Installation may take 5-10 minutes depending on your internet speed.

## 🎯 Running the Application

### Start the Streamlit Server

In your project directory with the virtual environment activated:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at:
```
http://localhost:8501
```

If it doesn't open automatically, manually navigate to the URL shown in the terminal.

### Stop the Application

Press `Ctrl+C` in the terminal to stop the server.

## 📖 How to Use

### 📷 Image Analysis
1. **Select Mode**: Choose "📷 Image Analysis" at the top
2. **Upload Image**: Click "Browse files" and select a traffic image
3. **Adjust Settings** (Optional): Use the sidebar to adjust detection confidence thresholds
4. **Analyze**: Click "🔍 Analyze Traffic" button
5. **View Results**: 
   - See annotated image with detected vehicles and plates
   - Browse gallery view for detailed information
   - Check table view for structured data
   - Export results as CSV or JSON

### 🎬 Video Analysis
1. **Select Mode**: Choose "🎬 Video Analysis" at the top
2. **Upload Video**: Click "Browse files" and select a traffic video (MP4, AVI, MOV, MKV)
3. **Frame Skip**: Adjust the "Process every Nth frame" slider (lower = more thorough but slower)
4. **Analyze**: Click "🎬 Analyze Video" button
5. **View Results**:
   - Video metadata (duration, FPS, resolution)
   - Sample annotated frames from the video
   - Deduplicated list of all unique plates found
   - Frame number and timestamp for each detection
   - Gallery, table, and export views

## 🛠️ Troubleshooting

### Issue: Models Not Loading

**Solution:** 
- Ensure your model files (`best2.pt` and `best.pt`) are in the `models/` directory
- Check file names match exactly
- Verify models are not corrupted

### Issue: PaddleOCR Installation Fails

**Solution:**
```bash
pip uninstall paddleocr paddlepaddle
pip install paddleocr==2.7.3 paddlepaddle==2.6.2 "numpy<2.0.0"
```

### Issue: OpenCV Import Error

**Solution:**
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.9.0.80
```

### Issue: Streamlit Not Found

**Solution:**
```bash
pip install --upgrade streamlit
```

### Issue: Memory Error

**Solution:**
- Process smaller images
- Reduce confidence thresholds
- Close other applications

## 🔧 Configuration

### Adjust Detection Thresholds

In the Streamlit sidebar, you can adjust:
- **Vehicle Detection Confidence**: Lower values detect more vehicles but may have false positives
- **Plate Detection Confidence**: Lower values detect more plates but may have false positives

### Modify Model Paths

If your models are in a different location, edit `app.py`:

```python
anpr_detector = ANPRDetector("path/to/your/anpr_model.pt")
vehicle_classifier = VehicleClassifier("path/to/your/vehicle_model.pt")
```

## 📊 Output Formats

### Gallery View
- Visual representation with plate crops
- Vehicle information
- OCR results with status indicators

### Table View
- Structured data in tabular format
- Statistics summary
- Sortable columns

### Export Options
- **CSV**: Comma-separated values for Excel/spreadsheet applications
- **JSON**: Structured data for programmatic access

## 🎨 Supported Vehicle Types

- **2W**: Two-wheelers (bikes, motorcycles)
- **3W**: Three-wheelers (auto-rickshaws)
- **4W**: Four-wheelers (cars, SUVs)
- **HMV**: Heavy Motor Vehicles (buses, trucks)

## 🇮🇳 Supported Plate Formats

- **Standard Format**: `SS00X0000` to `SS00XXX0000`
  - SS = State code (2 letters — 37 current + 4 legacy codes)
  - 00 = RTO code (1-2 digits; Delhi allows single-digit)
  - X = Series (0-3 letters; O and I are prohibited by regulation)
  - 0000 = Registration number (1-4 digits)

- **Bharat Series**: `YYBHXXXXAA`
  - YY = Year of registration (2 digits)
  - BH = Bharat indicator (national portability)
  - XXXX = Registration number (1-4 digits)
  - AA = Series letters (1-2 letters)

- **Legacy State Codes**: TS (Telangana → TG), OR (Orissa → OD), DN (→ DD), UA (→ UK) — all still accepted

### OCR Post-Processing Highlights

- **Smart state-code recovery**: Automatically fixes common OCR misreads (e.g., XA→KA, DI→DL, 6J→GJ, 8R→BR)
- **Digit/letter confusion maps**: Handles O↔0, I↔1, B↔8, S↔5, G↔6, Z↔2, A↔4, T↔7, L↔1, and more
- **O/I prohibition enforcement**: India bans O and I in series letters — forces correction
- **Score-based candidate selection**: When multiple preprocessing methods produce different valid plates, picks the most typical one
- **Flexible parsing**: Handles variable RTO lengths, 0-3 series letters, and leading junk characters

## 🔐 Privacy & Security

- All processing is done locally on your machine
- No data is sent to external servers
- Images are processed in memory and not stored unless you explicitly save them

## 📝 Development Notes

### Adding New Vehicle Types

Edit `vehicle_classifier.py`:
```python
VEHICLE_TYPE_MAP = {
    "bike": "2W",
    "car": "4W",
    "your_class": "YOUR_TYPE"  # Add your mapping
}
```

### Customizing OCR Post-Processing

Edit the `smart_post_process` method in `anpr_module.py` to add custom validation rules.

### Changing UI Theme

Streamlit supports themes. Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## 🐛 Known Limitations

- Works best with clear, well-lit images and videos
- Requires visible number plates
- Performance depends on model quality
- Large images/videos may take longer to process
- OCR accuracy varies with plate condition (faded, hand-painted, non-HSRP)
- Video processing speed depends on frame-skip setting and video length
- Military, diplomatic, and temporary plates are not validated (returned as INVALID)

## 📈 Performance Tips

1. **Image Quality**: Use high-resolution, clear images
2. **Lighting**: Better lighting = better detection
3. **Angle**: Front/rear views work best
4. **Distance**: Plates should be clearly visible
5. **Batch Processing**: Process multiple images separately for better memory management

## 🤝 Contributing

To improve the system:
1. Test with diverse datasets
2. Tune confidence thresholds
3. Add preprocessing methods
4. Enhance validation rules
5. Report issues and suggest features

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **Ultralytics**: YOLO implementation
- **PaddleOCR**: OCR engine
- **Streamlit**: Web framework
- **OpenCV**: Computer vision library

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the documentation
3. Test with sample images
4. Verify model files and paths

---

**Version**: 2.0.0  
**Last Updated**: February 2026

Happy Traffic Analyzing! 🚗🚦
