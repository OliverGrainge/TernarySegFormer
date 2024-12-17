from models.ternary_segformer import get_ternary_segformer_model_and_processor
from models.segformer import get_segformer_model_and_processor
import os 
import torch 
import argparse
import cv2
import torch.nn.functional as F
import numpy as np
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run semantic segmentation inference on real-time video.")
    parser.add_argument('--model', type=str, choices=['segformer', 'ternary_segformer'], help='Model type to use.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the model checkpoint file.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on, e.g. "cpu" or "cuda:0".')
    return parser.parse_args()

def _load_model(args):
    if args.model == "segformer": 
        model, processor = get_segformer_model_and_processor(pretrained=True)
    elif args.model == "ternary_segformer": 
        model, processor = get_ternary_segformer_model_and_processor(pretrained=True)

    #if os.path.exists(args.checkpoint_path):
    #    sd = torch.load(args.checkpoint_path, map_location=args.device)["state_dict"]
    #    model.load_state_dict(sd)
    model.to(args.device)
    model.eval()
    return model, processor

def main():
    args = parse_args()
    model, processor = _load_model(args)

    # Initialize video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Create a fixed color palette outside the loop
    num_classes = model.config.num_labels  # or however many classes your model has
    color_palette = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Measure preprocessing time
        preprocess_start = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=[rgb_frame], return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        preprocess_time = time.time() - preprocess_start

        # Measure inference time
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = time.time() - inference_start

        # Measure postprocessing time
        postprocess_start = time.time()
        original_h, original_w, _ = frame.shape
        logits = outputs.logits
        upsampled_logits = F.interpolate(logits, size=(original_h, original_w), mode='bilinear', align_corners=False)
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Efficient vectorized coloring
        colored_map = color_palette[pred_seg]  # This directly maps class indices to colors
        overlay = cv2.addWeighted(frame, 0.5, colored_map, 0.5, 0)
        postprocess_time = time.time() - postprocess_start

        print(f"Preprocessing: {preprocess_time*1000:.1f}ms, Inference: {inference_time*1000:.1f}ms, Postprocessing: {postprocess_time*1000:.1f}ms")

        # Display the result
        cv2.imshow('Real-Time Segmentation', overlay)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
