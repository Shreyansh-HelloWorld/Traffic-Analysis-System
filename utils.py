import cv2
import numpy as np
import pandas as pd

def draw_results(image, results):
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: OpenCV image (BGR format)
        results: List of detection results
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    for result in results:
        # Get coordinates
        vx1, vy1, vx2, vy2 = result['vehicle_bbox']
        px1, py1, px2, py2 = result['plate_bbox']
        
        # Determine colors based on plate validity
        if result['final_plate'] not in ["INVALID", "UNREADABLE"]:
            vehicle_color = (0, 255, 0)  # Green
            plate_color = (0, 255, 0)
            status = "VALID"
        elif result['final_plate'] == "INVALID":
            vehicle_color = (0, 165, 255)  # Orange
            plate_color = (0, 165, 255)
            status = "INVALID"
        else:
            vehicle_color = (0, 0, 255)  # Red
            plate_color = (0, 0, 255)
            status = "UNREADABLE"
        
        # Draw vehicle bounding box
        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), vehicle_color, 3)
        
        # Draw plate bounding box
        cv2.rectangle(annotated, (px1, py1), (px2, py2), plate_color, 2)
        
        # Prepare text labels
        vehicle_label = f"{result['vehicle_type']}"
        
        # Display text for final plate
        if result['final_plate'] == "INVALID" and result['raw_ocr'] not in ["UNREADABLE", "INVALID"]:
            plate_label = result['raw_ocr']
        else:
            plate_label = result['final_plate']
        
        # Draw vehicle type label background and text
        vehicle_text_size = cv2.getTextSize(vehicle_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(
            annotated,
            (vx1, vy1 - vehicle_text_size[1] - 15),
            (vx1 + vehicle_text_size[0] + 10, vy1),
            vehicle_color,
            -1
        )
        cv2.putText(
            annotated,
            vehicle_label,
            (vx1 + 5, vy1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )
        
        # Draw plate number label background and text
        plate_text_size = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        
        # Position plate label above the plate
        label_y = py1 - 10
        if label_y < 30:  # If too close to top, put below
            label_y = py2 + plate_text_size[1] + 10
        
        cv2.rectangle(
            annotated,
            (px1, label_y - plate_text_size[1] - 10),
            (px1 + plate_text_size[0] + 10, label_y),
            plate_color,
            -1
        )
        cv2.putText(
            annotated,
            plate_label,
            (px1 + 5, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
    
    return annotated


def create_results_dataframe(results):
    """
    Create a pandas DataFrame from results
    
    Args:
        results: List of detection results
        
    Returns:
        pandas DataFrame
    """
    data = []
    
    for idx, result in enumerate(results, 1):
        # Status emoji
        if result['final_plate'] not in ["INVALID", "UNREADABLE"]:
            status = "✅ Valid"
        elif result['final_plate'] == "INVALID":
            status = "⚠️ Invalid"
        else:
            status = "❌ Unreadable"
        
        data.append({
            'Sr. No': idx,
            'Vehicle Type': result['vehicle_type'],
            'Vehicle Class': result['vehicle_class'],
            'Plate Number': result['final_plate'],
            'Raw OCR': result['raw_ocr'],
            'Status': status,
            'Vehicle Conf': f"{result['vehicle_confidence']:.2%}",
            'Plate Conf': f"{result['plate_confidence']:.2%}"
        })
    
    return pd.DataFrame(data)


def save_plate_crops(results, output_dir):
    """
    Save individual plate crops to directory
    
    Args:
        results: List of detection results
        output_dir: Directory to save crops
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, result in enumerate(results, 1):
        if result['plate_crop'] is not None and result['plate_crop'].size > 0:
            filename = f"plate_{idx}_{result['final_plate']}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, result['plate_crop'])


def create_summary_stats(results):
    """
    Create summary statistics from results
    
    Args:
        results: List of detection results
        
    Returns:
        Dictionary with summary statistics
    """
    total = len(results)
    
    if total == 0:
        return {
            'total_vehicles': 0,
            'valid_plates': 0,
            'invalid_plates': 0,
            'unreadable_plates': 0,
            'success_rate': 0.0,
            'vehicle_type_distribution': {}
        }
    
    valid = sum(1 for r in results if r['final_plate'] not in ["INVALID", "UNREADABLE"])
    invalid = sum(1 for r in results if r['final_plate'] == "INVALID")
    unreadable = sum(1 for r in results if r['final_plate'] == "UNREADABLE")
    
    vehicle_types = {}
    for r in results:
        vtype = r['vehicle_type']
        vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1
    
    return {
        'total_vehicles': total,
        'valid_plates': valid,
        'invalid_plates': invalid,
        'unreadable_plates': unreadable,
        'success_rate': (valid / total) * 100 if total > 0 else 0,
        'vehicle_type_distribution': vehicle_types
    }


def format_plate_for_display(final_plate, raw_ocr):
    """
    Format plate number for display
    
    Args:
        final_plate: Processed plate number
        raw_ocr: Raw OCR text
        
    Returns:
        Formatted string for display
    """
    if final_plate not in ["INVALID", "UNREADABLE"]:
        return final_plate
    elif final_plate == "INVALID" and raw_ocr not in ["UNREADABLE", "INVALID"]:
        return f"{raw_ocr} (Invalid)"
    else:
        return final_plate
