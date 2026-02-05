"""
Batch Processing Script for Traffic Analysis
Process multiple images from a directory and save results
"""

import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

from anpr_module import ANPRDetector
from vehicle_classifier import VehicleClassifier
from utils import draw_results, create_results_dataframe, save_plate_crops


def process_batch(input_dir, output_dir, anpr_model_path, vehicle_model_path,
                  vehicle_conf=0.3, plate_conf=0.4):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results
        anpr_model_path: Path to ANPR model weights
        vehicle_model_path: Path to vehicle classification model weights
        vehicle_conf: Vehicle detection confidence threshold
        plate_conf: Plate detection confidence threshold
    """
    
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    annotated_dir = output_dir / "annotated_images"
    annotated_dir.mkdir(exist_ok=True)
    
    crops_dir = output_dir / "plate_crops"
    crops_dir.mkdir(exist_ok=True)
    
    # Initialize models
    print("Loading models...")
    anpr_detector = ANPRDetector(anpr_model_path, conf_threshold=plate_conf)
    vehicle_classifier = VehicleClassifier(vehicle_model_path, conf_threshold=vehicle_conf)
    print("Models loaded successfully!\n")
    
    # Get list of image files
    input_dir = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    
    # Process each image
    all_results = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read {img_path.name}")
                continue
            
            # Detect vehicles
            vehicle_detections = vehicle_classifier.detect_vehicles(image)
            
            image_results = []
            
            # Process each vehicle
            for vehicle in vehicle_detections:
                x1, y1, x2, y2 = vehicle['bbox']
                vehicle_crop = image[y1:y2, x1:x2]
                
                # Detect plates in vehicle region
                plate_detections = anpr_detector.detect_plates(vehicle_crop)
                
                for plate in plate_detections:
                    # Get plate coordinates relative to original image
                    px1, py1, px2, py2 = plate['bbox']
                    plate_x1 = x1 + px1
                    plate_y1 = y1 + py1
                    plate_x2 = x1 + px2
                    plate_y2 = y1 + py2
                    
                    # Extract plate crop
                    plate_crop = image[plate_y1:plate_y2, plate_x1:plate_x2]
                    
                    # Perform OCR
                    raw_text, final_plate = anpr_detector.perform_ocr(plate_crop)
                    
                    # Store results
                    result = {
                        'image_name': img_path.name,
                        'vehicle_bbox': vehicle['bbox'],
                        'vehicle_type': vehicle['vehicle_type'],
                        'vehicle_class': vehicle['class_name'],
                        'vehicle_confidence': vehicle['confidence'],
                        'plate_bbox': (plate_x1, plate_y1, plate_x2, plate_y2),
                        'plate_crop': plate_crop,
                        'raw_ocr': raw_text,
                        'final_plate': final_plate,
                        'plate_confidence': plate['confidence']
                    }
                    
                    image_results.append(result)
                    all_results.append(result)
            
            # Draw and save annotated image
            if image_results:
                annotated = draw_results(image.copy(), image_results)
                output_path = annotated_dir / img_path.name
                cv2.imwrite(str(output_path), annotated)
                
                # Save plate crops
                for idx, result in enumerate(image_results):
                    crop_filename = f"{img_path.stem}_plate_{idx+1}_{result['final_plate']}.jpg"
                    crop_path = crops_dir / crop_filename
                    if result['plate_crop'] is not None and result['plate_crop'].size > 0:
                        cv2.imwrite(str(crop_path), result['plate_crop'])
        
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue
    
    # Save consolidated CSV
    if all_results:
        # Create detailed DataFrame
        data = []
        for result in all_results:
            status = "Valid" if result['final_plate'] not in ["INVALID", "UNREADABLE"] else result['final_plate']
            
            data.append({
                'Image': result['image_name'],
                'Vehicle Type': result['vehicle_type'],
                'Vehicle Class': result['vehicle_class'],
                'Vehicle Confidence': f"{result['vehicle_confidence']:.3f}",
                'Plate Number': result['final_plate'],
                'Raw OCR': result['raw_ocr'],
                'Status': status,
                'Plate Confidence': f"{result['plate_confidence']:.3f}"
            })
        
        df = pd.DataFrame(data)
        csv_path = output_dir / "batch_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Create summary statistics
        summary = {
            'Total Images Processed': len(image_files),
            'Total Vehicles Detected': len(all_results),
            'Valid Plates': sum(1 for r in all_results if r['final_plate'] not in ["INVALID", "UNREADABLE"]),
            'Invalid Plates': sum(1 for r in all_results if r['final_plate'] == "INVALID"),
            'Unreadable Plates': sum(1 for r in all_results if r['final_plate'] == "UNREADABLE")
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE!")
        print("="*60)
        print(f"Processed: {len(image_files)} images")
        print(f"Detected: {len(all_results)} vehicles")
        print(f"Valid plates: {summary['Valid Plates']}")
        print(f"\nResults saved to: {output_dir}")
        print(f"  - Annotated images: {annotated_dir}")
        print(f"  - Plate crops: {crops_dir}")
        print(f"  - Detailed results: {csv_path}")
        print(f"  - Summary: {summary_path}")
        print("="*60)
    else:
        print("\nNo vehicles detected in any images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process traffic images")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--anpr-model", type=str, default="models/best2.pt",
                        help="Path to ANPR model weights")
    parser.add_argument("--vehicle-model", type=str, default="models/best.pt",
                        help="Path to vehicle classification model weights")
    parser.add_argument("--vehicle-conf", type=float, default=0.3,
                        help="Vehicle detection confidence threshold")
    parser.add_argument("--plate-conf", type=float, default=0.4,
                        help="Plate detection confidence threshold")
    
    args = parser.parse_args()
    
    process_batch(
        input_dir=args.input,
        output_dir=args.output,
        anpr_model_path=args.anpr_model,
        vehicle_model_path=args.vehicle_model,
        vehicle_conf=args.vehicle_conf,
        plate_conf=args.plate_conf
    )
