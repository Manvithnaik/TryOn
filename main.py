import cv2
import numpy as np
import argparse
import os
from m import get_body_measurements
from model_scaling import scale_body_model, scale_clothing_model
from model_fitting import fit_clothing_to_body
from visualization import visualize_fitted_models

def main():
    parser = argparse.ArgumentParser(description="Virtual clothing fitting using body measurements")
    parser.add_argument("--body", default="models/body.obj", help="Path to body model file")
    parser.add_argument("--shirt", default="models/shirt.obj", help="Path to shirt model file")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()
    
    # Ensure the models directory exists
    if not os.path.exists("models"):
        print("Error: 'models' directory not found.")
        return
    
    # Check if the model files exist
    if not os.path.exists(args.body) or not os.path.exists(args.shirt):
        print(f"Error: Model files not found in {args.body} or {args.shirt}")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    print("Starting body measurement capture...")
    # Capture measurements using webcam
    measurements = get_body_measurements()
    if measurements is None:
        print("Failed to get measurements. Exiting.")
        return
    
    print(f"Measurements captured: {measurements}")
    
    # Scale the body model based on measurements
    print("Scaling body model...")
    scaled_body = scale_body_model(args.body, measurements)
    
    # Scale the clothing model to match the body
    print("Scaling clothing model...")
    scaled_shirt = scale_clothing_model(args.shirt, measurements)
    
    # Fit the clothing onto the body
    print("Fitting clothing to body...")
    fitted_result = fit_clothing_to_body(scaled_body, scaled_shirt)
    
    # Visualize the result
    print("Visualizing result...")
    visualize_fitted_models(fitted_result)
    
    print("Process completed successfully")

if __name__ == "__main__":
    main()