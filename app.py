import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
from pathlib import Path
import tempfile
import time
os.environ["YOLO_CPUINFO"] = "False"

# Import custom modules
from anpr_module import ANPRDetector
from vehicle_classifier import VehicleClassifier
from utils import draw_results, create_results_dataframe

# Page configuration
st.set_page_config(
    page_title="Traffic Analysis System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .results-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'vid_processed' not in st.session_state:
    st.session_state.vid_processed = False
if 'vid_results' not in st.session_state:
    st.session_state.vid_results = None

def initialize_models():
    """Initialize YOLO models for ANPR and vehicle classification"""
    try:
        with st.spinner("Loading AI models... This may take a moment..."):
            anpr_detector = ANPRDetector("models/best2.pt")
            vehicle_classifier = VehicleClassifier("models/best.pt")
        st.success("✅ Models loaded successfully!")
        return anpr_detector, vehicle_classifier
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please ensure model files are in the 'models' directory")
        return None, None

def process_image(image, anpr_detector, vehicle_classifier):
    """Process image through both ANPR and vehicle classification"""
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # Grayscale
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:  # RGBA
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    results = []
    
    # Step 1: Detect vehicles
    with st.spinner("🚗 Detecting vehicles..."):
        vehicle_detections = vehicle_classifier.detect_vehicles(img_cv)
    
    # Step 2: Process each vehicle
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if len(vehicle_detections) == 0:
        status_text.text("No vehicles detected in image.")
        progress_bar.progress(1.0)
    
    for idx, vehicle in enumerate(vehicle_detections):
        status_text.text(f"Processing vehicle {idx + 1}/{len(vehicle_detections)}...")
        progress_bar.progress((idx + 1) / len(vehicle_detections))
        
        # Get vehicle crop
        x1, y1, x2, y2 = vehicle['bbox']
        vehicle_crop = img_cv[y1:y2, x1:x2]
        
        # Detect number plate in vehicle region
        plate_detections = anpr_detector.detect_plates(vehicle_crop)
        
        for plate in plate_detections:
            # Get plate coordinates relative to original image
            px1, py1, px2, py2 = plate['bbox']
            plate_x1 = x1 + px1
            plate_y1 = y1 + py1
            plate_x2 = x1 + px2
            plate_y2 = y1 + py2
            
            # Extract plate crop from original image
            plate_crop = img_cv[plate_y1:plate_y2, plate_x1:plate_x2]
            
            # Perform OCR
            raw_text, final_plate = anpr_detector.perform_ocr(plate_crop)
            
            # Store results
            results.append({
                'vehicle_bbox': vehicle['bbox'],
                'vehicle_type': vehicle['vehicle_type'],
                'vehicle_class': vehicle['class_name'],
                'vehicle_confidence': vehicle['confidence'],
                'plate_bbox': (plate_x1, plate_y1, plate_x2, plate_y2),
                'plate_crop': plate_crop,
                'raw_ocr': raw_text,
                'final_plate': final_plate,
                'plate_confidence': plate['confidence']
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return results, img_cv


def process_frame(frame_bgr, anpr_detector, vehicle_classifier):
    """Process a single video frame and return detection results (no Streamlit calls)."""
    results = []
    vehicle_detections = vehicle_classifier.detect_vehicles(frame_bgr)
    
    for vehicle in vehicle_detections:
        x1, y1, x2, y2 = vehicle['bbox']
        vehicle_crop = frame_bgr[y1:y2, x1:x2]
        plate_detections = anpr_detector.detect_plates(vehicle_crop)
        
        for plate in plate_detections:
            px1, py1, px2, py2 = plate['bbox']
            plate_x1 = x1 + px1
            plate_y1 = y1 + py1
            plate_x2 = x1 + px2
            plate_y2 = y1 + py2
            
            plate_crop = frame_bgr[plate_y1:plate_y2, plate_x1:plate_x2]
            raw_text, final_plate = anpr_detector.perform_ocr(plate_crop)
            
            results.append({
                'vehicle_bbox': vehicle['bbox'],
                'vehicle_type': vehicle['vehicle_type'],
                'vehicle_class': vehicle['class_name'],
                'vehicle_confidence': vehicle['confidence'],
                'plate_bbox': (plate_x1, plate_y1, plate_x2, plate_y2),
                'plate_crop': plate_crop,
                'raw_ocr': raw_text,
                'final_plate': final_plate,
                'plate_confidence': plate['confidence']
            })
    
    return results


def process_video(video_path, anpr_detector, vehicle_classifier, frame_skip=5):
    """
    Process video file frame-by-frame with deduplication.
    
    Args:
        video_path: Path to the video file
        anpr_detector: ANPRDetector instance
        vehicle_classifier: VehicleClassifier instance
        frame_skip: Process every Nth frame to balance speed/coverage
    
    Returns:
        all_results: list of per-detection dicts (deduplicated)
        sample_frames: list of (frame_rgb, frame_results) for display
        video_info: dict with fps, total_frames, duration, etc.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Could not open video file.")
        return [], [], {}
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_info = {
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration,
        'width': width,
        'height': height,
    }
    
    # Tracking structures
    seen_plates = {}        # plate_number -> best detection dict
    sample_frames = []      # up to ~10 annotated sample frames
    frames_to_sample = max(1, total_frames // (10 * frame_skip))
    
    progress = st.progress(0)
    status = st.empty()
    frame_idx = 0
    processed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        processed_count += 1
        pct = min(frame_idx / max(total_frames, 1), 1.0)
        status.text(
            f"⏳ Frame {frame_idx}/{total_frames}  •  "
            f"{processed_count} frames processed  •  "
            f"{len(seen_plates)} unique plates found"
        )
        progress.progress(pct)
        
        frame_results = process_frame(frame, anpr_detector, vehicle_classifier)
        
        for det in frame_results:
            plate = det['final_plate']
            if plate in ("INVALID", "UNREADABLE"):
                continue
            # Keep highest-confidence sighting
            if plate not in seen_plates or det['plate_confidence'] > seen_plates[plate]['plate_confidence']:
                det['frame_number'] = frame_idx
                det['timestamp'] = round(frame_idx / fps, 2)
                seen_plates[plate] = det
        
        # Sample annotated frames for preview
        if processed_count % frames_to_sample == 0 and len(sample_frames) < 12:
            annotated = draw_results(frame.copy(), frame_results)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            sample_frames.append((annotated_rgb, frame_results))
        
        frame_idx += 1
    
    cap.release()
    progress.progress(1.0)
    status.text(
        f"✅ Done — processed {processed_count} frames, "
        f"found {len(seen_plates)} unique plates"
    )
    
    all_results = list(seen_plates.values())
    return all_results, sample_frames, video_info

def main():
    # Header
    st.title("🚦 Intelligent Traffic Analysis System")
    st.markdown("### Automatic Number Plate Recognition (ANPR) + Vehicle Classification")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model confidence thresholds
        st.subheader("Detection Settings")
        vehicle_conf = st.slider("Vehicle Detection Confidence", 0.0, 1.0, 0.3, 0.05)
        plate_conf = st.slider("Plate Detection Confidence", 0.0, 1.0, 0.4, 0.05)
        
        st.markdown("---")
        
        # Information
        st.subheader("ℹ️ About")
        st.info("""
        This system performs:
        - Vehicle detection & classification
        - Number plate detection
        - OCR for plate reading
        - Complete traffic analysis
        - **Image & Video support**
        """)
        
        st.markdown("---")
        st.subheader("📋 Supported Vehicles")
        st.markdown("""
        - 🏍️ Two Wheeler (2W)
        - 🚗 Four Wheeler (4W)
        - 🛺 Three Wheeler (3W)
        - 🚌 Heavy Motor Vehicle (HMV)
        """)
    
    # ─── Mode selector: Image vs Video ───
    mode = st.radio(
        "Select input mode",
        ["📷 Image Analysis", "🎬 Video Analysis"],
        horizontal=True,
    )
    st.markdown("---")
    
    if mode == "📷 Image Analysis":
        _image_analysis_ui(vehicle_conf, plate_conf)
    else:
        _video_analysis_ui(vehicle_conf, plate_conf)


# ═══════════════════════════════════════════════════
#  IMAGE ANALYSIS UI  (unchanged logic, refactored)
# ═══════════════════════════════════════════════════
def _image_analysis_ui(vehicle_conf, plate_conf):
    """Image upload + analysis."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a traffic image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of vehicles with visible number plates",
            key="img_uploader",
        )
        
        if uploaded_file is not None:
            # Robust image loading — handles corrupted, HEIC, or truncated files
            image = None
            try:
                uploaded_file.seek(0)
                image = Image.open(uploaded_file)
                image.load()  # Force full decode to catch truncated files
            except Exception:
                # Fallback: try decoding with OpenCV
                try:
                    uploaded_file.seek(0)
                    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
                    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img_cv is not None:
                        image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                    else:
                        st.error("❌ Could not decode this image. Please try a different file.")
                except Exception:
                    st.error("❌ Could not open this image. The file may be corrupted or in an unsupported format (e.g. HEIC). Please convert it to JPG/PNG and try again.")
            
            if image is not None:
                st.image(image, caption="Uploaded Image")
            
            if image is not None and st.button("🔍 Analyze Traffic", type="primary", key="analyze_img"):
                anpr_detector, vehicle_classifier = initialize_models()
                
                if anpr_detector and vehicle_classifier:
                    vehicle_classifier.conf_threshold = vehicle_conf
                    anpr_detector.conf_threshold = plate_conf
                    
                    with st.spinner("Processing image... Please wait..."):
                        results, processed_img = process_image(
                            image, anpr_detector, vehicle_classifier
                        )
                    
                    st.session_state.results = results
                    st.session_state.processed_img = processed_img
                    st.session_state.processed = True
                    
                    st.success(f"✅ Analysis complete! Found {len(results)} vehicle(s)")
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if st.session_state.processed and st.session_state.results is not None:
            results = st.session_state.results
            processed_img = st.session_state.processed_img
            
            annotated_img = draw_results(processed_img.copy(), results)
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_img_rgb, caption="Analyzed Image")
            
            _, buffer = cv2.imencode('.jpg', annotated_img)
            st.download_button(
                label="📥 Download Annotated Image",
                data=buffer.tobytes(),
                file_name="traffic_analysis_result.jpg",
                mime="image/jpeg",
            )
        else:
            st.info("👆 Upload an image and click 'Analyze Traffic' to see results")
    
    # Detailed results section
    if st.session_state.processed and st.session_state.results:
        _show_detailed_results(st.session_state.results)


# ═══════════════════════════════════════════════════
#  VIDEO ANALYSIS UI
# ═══════════════════════════════════════════════════
def _video_analysis_ui(vehicle_conf, plate_conf):
    """Video upload + frame-by-frame analysis."""
    st.subheader("📤 Upload Video")
    uploaded_video = st.file_uploader(
        "Choose a traffic video...",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a traffic video for frame-by-frame analysis",
        key="vid_uploader",
    )
    
    if uploaded_video is not None:
        # Save to temp file (OpenCV needs a path)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        video_path = tfile.name
        
        st.video(uploaded_video)
        
        col_a, col_b = st.columns(2)
        with col_a:
            frame_skip = st.slider(
                "Process every Nth frame",
                min_value=1, max_value=30, value=5,
                help="Lower = more thorough but slower. 1 = every frame, 10 = fast scan."
            )
        with col_b:
            st.markdown("")   # spacer
            st.markdown("")
        
        if st.button("🎬 Analyze Video", type="primary", key="analyze_vid"):
            anpr_detector, vehicle_classifier = initialize_models()
            
            if anpr_detector and vehicle_classifier:
                vehicle_classifier.conf_threshold = vehicle_conf
                anpr_detector.conf_threshold = plate_conf
                
                start_time = time.time()
                all_results, sample_frames, video_info = process_video(
                    video_path, anpr_detector, vehicle_classifier,
                    frame_skip=frame_skip,
                )
                elapsed = time.time() - start_time
                
                st.session_state.vid_results = all_results
                st.session_state.vid_samples = sample_frames
                st.session_state.vid_info = video_info
                st.session_state.vid_elapsed = elapsed
                st.session_state.vid_processed = True
        
        # Clean up temp file
        try:
            os.unlink(video_path)
        except:
            pass
    
    # ── Show video results ──
    if st.session_state.get('vid_processed') and st.session_state.get('vid_results') is not None:
        results = st.session_state.vid_results
        sample_frames = st.session_state.vid_samples
        video_info = st.session_state.vid_info
        elapsed = st.session_state.vid_elapsed
        
        st.markdown("---")
        st.header("🎬 Video Analysis Results")
        
        # ── Summary metrics ──
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Duration", f"{video_info.get('duration', 0):.1f}s")
        with c2:
            st.metric("Total Frames", video_info.get('total_frames', 0))
        with c3:
            st.metric("FPS", f"{video_info.get('fps', 0):.1f}")
        with c4:
            st.metric("Unique Plates", len(results))
        with c5:
            st.metric("Processing Time", f"{elapsed:.1f}s")
        
        # ── Sample annotated frames ──
        if sample_frames:
            st.subheader("📸 Sample Frames")
            cols = st.columns(min(3, len(sample_frames)))
            for i, (frame_rgb, _) in enumerate(sample_frames[:6]):
                with cols[i % 3]:
                    st.image(frame_rgb, caption=f"Sample {i+1}", use_container_width=True)
        
        # ── Detailed results (reuse shared widget) ──
        if results:
            _show_detailed_results(results, is_video=True)
        else:
            st.warning("⚠️ No vehicles with readable number plates detected in the video.")


# ═══════════════════════════════════════════════════
#  SHARED DETAILED RESULTS WIDGET
# ═══════════════════════════════════════════════════
def _show_detailed_results(results, is_video=False):
    """Render the gallery / table / export tabs for a list of results."""
    st.markdown("---")
    st.header("📋 Detailed Results")
    
    if len(results) == 0:
        st.warning("⚠️ No vehicles with readable number plates detected")
        return
    
    tab1, tab2, tab3 = st.tabs(["🖼️ Gallery View", "📊 Table View", "📄 Export Data"])
    
    with tab1:
        for idx, result in enumerate(results, 1):
            label = f"Vehicle #{idx} - {result['vehicle_type']} - {result['final_plate']}"
            if is_video and 'timestamp' in result:
                label += f"  (@ {result['timestamp']}s)"
            
            with st.expander(label, expanded=(idx <= 5)):
                col_a, col_b, col_c = st.columns([1, 1, 1])
                
                with col_a:
                    st.markdown("**📸 Number Plate**")
                    if result['plate_crop'] is not None and result['plate_crop'].size > 0:
                        plate_rgb = cv2.cvtColor(result['plate_crop'], cv2.COLOR_BGR2RGB)
                        h, w = plate_rgb.shape[:2]
                        scale = max(1, 300 // max(w, 1))
                        if scale > 1:
                            plate_rgb = cv2.resize(
                                plate_rgb, (w * scale, h * scale),
                                interpolation=cv2.INTER_CUBIC,
                            )
                        st.image(plate_rgb)
                    else:
                        st.warning("No plate crop available")
                
                with col_b:
                    st.markdown("**🚗 Vehicle Info**")
                    st.write(f"**Type:** {result['vehicle_type']}")
                    st.write(f"**Class:** {result['vehicle_class']}")
                    st.write(f"**Confidence:** {result['vehicle_confidence']:.2%}")
                    if is_video and 'frame_number' in result:
                        st.write(f"**Frame:** {result['frame_number']}")
                        st.write(f"**Timestamp:** {result.get('timestamp', 'N/A')}s")
                
                with col_c:
                    st.markdown("**🔤 OCR Results**")
                    if result['final_plate'] not in ("INVALID", "UNREADABLE"):
                        st.success(f"✅ **{result['final_plate']}**")
                    elif result['final_plate'] == "INVALID":
                        st.warning(f"⚠️ **INVALID**")
                    else:
                        st.error(f"❌ **UNREADABLE**")
                    st.write(f"**Raw OCR:** {result['raw_ocr']}")
                    st.write(f"**Plate Conf:** {result['plate_confidence']:.2%}")
    
    with tab2:
        df = create_results_dataframe(results)
        if is_video and results and 'timestamp' in results[0]:
            # Add video-specific columns
            ts_data = []
            for r in results:
                ts_data.append({
                    'Frame': r.get('frame_number', ''),
                    'Time (s)': r.get('timestamp', ''),
                })
            ts_df = pd.DataFrame(ts_data)
            df = pd.concat([df, ts_df], axis=1)
        
        st.dataframe(df, hide_index=True)
        
        st.markdown("### 📈 Statistics")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.metric("Total Vehicles", len(results))
        with col_s2:
            valid_plates = sum(1 for r in results if r['final_plate'] not in ("INVALID", "UNREADABLE"))
            st.metric("Valid Plates", valid_plates)
        with col_s3:
            vehicle_types = pd.Series([r['vehicle_type'] for r in results])
            st.metric("Most Common", vehicle_types.mode()[0] if len(vehicle_types) > 0 else "N/A")
        with col_s4:
            avg_conf = np.mean([r['vehicle_confidence'] for r in results])
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
    
    with tab3:
        st.markdown("### 💾 Export Options")
        
        df = create_results_dataframe(results)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="traffic_analysis_results.csv",
            mime="text/csv",
        )
        
        import json
        json_data = []
        for r in results:
            entry = {
                'vehicle_type': r['vehicle_type'],
                'vehicle_class': r['vehicle_class'],
                'vehicle_confidence': float(r['vehicle_confidence']),
                'plate_number': r['final_plate'],
                'raw_ocr': r['raw_ocr'],
                'plate_confidence': float(r['plate_confidence']),
            }
            if is_video:
                entry['frame_number'] = r.get('frame_number', None)
                entry['timestamp'] = r.get('timestamp', None)
            json_data.append(entry)
        
        json_str = json.dumps(json_data, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="traffic_analysis_results.json",
            mime="application/json",
        )

if __name__ == "__main__":
    main()
