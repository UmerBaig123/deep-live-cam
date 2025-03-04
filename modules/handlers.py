from modules.ui import deep_fake_video,deep_fake_image,get_unique_faces_from_target_image
from modules.ui import get_unique_faces_from_target_video,deep_fake_live
import os
def deepfake_video(source_path,target_path,output_path):
    output = deep_fake_video(source_path, target_path, output_path)
    # output.save(output_path, format="PNG")
    # print(f"Output saved to: {output_path}")

def deepfake_image(source_path,target_path,output_path,enhance):
    output = deep_fake_image(source_path, target_path,output_path,enhance)
    output.save(output_path, format="jpeg")