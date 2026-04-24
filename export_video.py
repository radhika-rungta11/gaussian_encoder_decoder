import os
import subprocess

def export_frames_to_mp4(frame_dir, output_file, framerate=30):
    """
    Finds all png files in the frame_dir and uses FFmpeg (if available) to compile them into an mp4.
    If FFmpeg is not available, informs the user or tries ImageIO.
    """
    print(f"Exporting video from {frame_dir} to {output_file} at {framerate} FPS...")
    
    # Simplest way is using ffmpeg directly
    # Assumes frames are named like frame_0000.png, frame_0001.png
    input_pattern = os.path.join(frame_dir, "frame_%04d.png")
    
    cmd = [
        "ffmpeg", 
        "-y", # overwrite
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Export successful.")
    except Exception as e:
        print("FFmpeg not found or failed. Ensure ffmpeg is installed.")
        print(f"Error: {e}")
        print("Falling back to instructions...")
        print(f"To compile manually: ffmpeg -framerate {framerate} -i {input_pattern} -c:v libx264 -pix_fmt yuv420p {output_file}")
