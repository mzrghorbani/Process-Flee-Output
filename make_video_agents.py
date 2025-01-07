import os
import glob
from moviepy.editor import ImageSequenceClip


if __name__ == "__main__":

    # Specify the pattern for matching files
    file_pattern = 'agents_timestep_*.png'

    # Use glob to find all matching files and sort numerically
    matching_files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Extract filenames (removing the path and the .png extension) for image files
    image_files = [os.path.basename(f) for f in matching_files]
    
    # Create an animation
    clip = ImageSequenceClip(image_files, fps=2)  # Adjust fps as needed

    clip.write_videofile("agent_movements_animation.mp4", codec="libx264")

    print("Agents video created!")


