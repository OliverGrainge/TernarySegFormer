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
    parser = argparse.ArgumentParser(description="Run semantic segmentation on video and save output with model latency.")
    parser.add_argument('--model', type=str, choices=['segformer', 'ternary_segformer'], help='Model type to use.')
    parser.add_argument('--input_video', type=str, help='Path to input video file.')
    parser.add_argument('--output_video', type=str, help='Path to save the processed video.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the model checkpoint file.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on, e.g. "cpu" or "cuda:0".')
    return parser.parse_args()

def _load_model(args):
    if args.model == "segformer": 
        model, processor = get_segformer_model_and_processor(pretrained=True)
    elif args.model == "ternary_segformer": 
        model, processor = get_ternary_segformer_model_and_processor(pretrained=True)

    # Uncomment to load a specific checkpoint
    # if os.path.exists(args.checkpoint_path):
    #     sd = torch.load(args.checkpoint_path, map_location=args.device)["state_dict"]
    #     model.load_state_dict(sd)
    model.to(args.device)
    model.eval()
    return model, processor

def process_video(args):
    model, processor = _load_model(args)

    # Open the input video
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("Cannot open video file")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = 1.0 / fps

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (frame_width, frame_height))

    # Generate a fixed color palette for classes
    num_classes = model.config.num_labels
    color_palette = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)

    real_time_elapsed = 0.0
    output_frame_count = 0  # Number of frames actually written to the output video

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file or no more frames to read.")
            break

        # Start timing inference
        frame_start_time = time.time()

        # Model processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=[rgb_frame], return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        frame_time = time.time() - frame_start_time
        real_time_elapsed += frame_time

        # Postprocessing
        logits = outputs.logits
        upsampled_logits = F.interpolate(logits, size=(frame_height, frame_width), mode='bilinear', align_corners=False)
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        colored_map = color_palette[pred_seg]
        overlay = cv2.addWeighted(frame, 0.5, colored_map, 0.5, 0)

        # Write the processed frame
        out.write(overlay)
        output_frame_count += 1

        # Print latency info
        print(f"Processed frame {output_frame_count} | Latency: {frame_time:.3f}s")

        # Calculate how many frames should have passed in real-time by now
        expected_processed_frames = int(real_time_elapsed / frame_interval)

        # Determine how many frames we are behind. This simulates missed frames.
        frames_to_skip = expected_processed_frames - output_frame_count

        if frames_to_skip > 0:
            # Skip these frames in the input capture
            for _ in range(frames_to_skip):
                ret_skip = cap.grab()  # just grab and discard
                if not ret_skip:
                    break

            # Now, to keep the final video running at the same fps and duration,
            # we duplicate the last processed frame 'frames_to_skip' times.
            for _ in range(frames_to_skip):
                out.write(overlay)  # Repeat the last processed frame
                output_frame_count += 1

            print(f"Skipped {frames_to_skip} frames and duplicated the last processed frame {frames_to_skip} times.")

        # (Optional) Limit processing for demonstration:
        # if output_frame_count == 50:
        #     break

    # Cleanup
    cap.release()
    out.release()
    print(f"Video processing complete. Saved to {args.output_video}")
    avg_latency = real_time_elapsed / max(1, output_frame_count)
    print(f"Average Frame Latency: {avg_latency:.3f}s")
    
def main():
    args = parse_args()
    process_video(args)

if __name__ == "__main__":
    main()
