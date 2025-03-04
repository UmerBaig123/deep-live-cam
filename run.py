
import gradio as gr
import torch
import cv2
import numpy as np
import shutil
from PIL import Image
import requests
import os
from gradio_webrtc import WebRTC
import sys 
import modules.globals
import onnxruntime
from modules.face_analyser import get_one_face
from modules.handlers import deepfake_image,deep_fake_video,get_unique_faces_from_target_image,get_unique_faces_from_target_video,deep_fake_live
from datetime import datetime
import random
def get_random_number():
    return random.randint(0, 1000)
print(torch.version.cuda) 
modules.globals.execution_providers=['CUDAExecutionProvider']
print(modules.globals.execution_providers)
class image_processes:
  def __init__(self) -> None:
        pass
  def ui():
    map_index = gr.State(0)
    with gr.Row():
      source_face_input = gr.Image(label="Select Source Face", type="numpy")
      target_image_input = gr.Image(label="Select Target Image", type="numpy")

    with gr.Row():
      enhance_face_checkbox = gr.Checkbox(label="Enhance Face", value=False)
      face_mapping_checkbox = gr.Checkbox(label="Map Faces", value=False)
    with gr.Row():
      mouth_mask_checkbox = gr.Checkbox(label="Mouth Mask", value=False)
      Many_faces_checkbox = gr.Checkbox(label="Many Faces", value=False)
    with gr.Column(visible=False,elem_id="face_map_column") as map_col:
      with gr.Row():
        mapping_source = gr.Image(label="Select Source Face", type="numpy")
        mapping_target = gr.Image(label="Select Target Image", type="numpy", interactive=False)
      with gr.Row():
        previous_button = gr.Button("previous",elem_classes="previous_button")
        submit_button = gr.Button("submit",elem_classes="submit_button")
        next_button = gr.Button("next",elem_classes="next_button")
    process_button = gr.Button("Process Image")
    with gr.Row():
        output_image = gr.Image(label="Processed Image", interactive=False,type="numpy")
        # download_button = gr.File(label="Download Processed Image")


    status_textbox = gr.Textbox(label="Status", interactive=False)
    mapping_source.upload(
        image_processes.update_map,
        inputs=[map_index,mapping_source]
    )
    next_button.click(
        image_processes.next_map,
        inputs=[map_index,mapping_source],
        outputs=[map_index,mapping_source,mapping_target]
    )
    previous_button.click(
        image_processes.prev_map,
        inputs=[map_index,mapping_source],
        outputs=[map_index,mapping_source,mapping_target]
    )
    process_button.click(
      image_processes.process_image,
      inputs=[source_face_input, target_image_input, status_textbox,enhance_face_checkbox,face_mapping_checkbox,mouth_mask_checkbox,Many_faces_checkbox],
      outputs=[status_textbox,output_image,map_col,mapping_target,mapping_source]
    )
    submit_button.click(
      image_processes.submit_mapping,
      inputs=[source_face_input, target_image_input, status_textbox,enhance_face_checkbox,face_mapping_checkbox],
      outputs=[status_textbox,output_image,map_col,mapping_target,mapping_source]
    )
  def process_image(source_face, target_image, status_textbox,enhance,map_faces,mouth_mask,many_faces):
      """
      Save source and target images as JPG and display the uploaded target image instead of processing.
      """
      modules.globals.mouth_mask = mouth_mask
      modules.globals.many_faces = many_faces
      now = datetime.now()
      formatted_now = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Truncate to 3 decimal places for milliseconds
      random_no = get_random_number()
      source_path = f"temp_source_output/source-{random_no}-{formatted_now}.jpeg" 
      output_path = f"temp_source_output/output-{random_no}-{formatted_now}.jpeg"
      modules.globals.output_path = output_path
      if target_image is None:
          return "No target image uploaded!", None

      modules.globals.map_faces=map_faces
      # Convert images to correct format for saving
      target_np =  target_image

      cv2.imwrite(output_path, cv2.cvtColor(target_np, cv2.COLOR_BGR2RGB))
      modules.globals.target_path = output_path
      if map_faces:
        get_unique_faces_from_target_image()
        map = modules.globals.source_target_map[0]
        source = modules.globals.source_target_map[0].get("source",None)
        return "Please map faces", None,gr.update(visible=True),cv2.cvtColor(map["target"]["cv2"], cv2.COLOR_BGR2RGB),source
      
      # Save images as JPG files
      if source_face is None:
        source_face = target_np 
      source_np = source_face
      cv2.imwrite(source_path, cv2.cvtColor(source_np, cv2.COLOR_BGR2RGB))
      deepfake_image(source_path,output_path,output_path,enhance)
      # Convert target image to correct format for display
      target_np =  cv2.imread(output_path)
      target_display = cv2.cvtColor(target_np, cv2.COLOR_BGR2RGB)
      if os.path.exists(output_path):
        os.remove(output_path)
      if os.path.exists(source_path):
        os.remove(source_path)
      return "Images deepfaked successfully!", target_display,gr.update(visible=False),None,None


  def submit_mapping(source_face, target_image, status_textbox,enhance,map_faces):
      if target_image is None:
          return "No target image uploaded!", None

      # Convert images to correct format for saving
      target_np = np.array(target_image)
      
      now = datetime.now()
      formatted_now = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Truncate to 3 decimal places for milliseconds
      random_no = get_random_number()
      source_path = f"temp_source_output/source-{random_no}-{formatted_now}.jpeg" 
      output_path = f"temp_source_output/output-{random_no}-{formatted_now}.jpeg"
      cv2.imwrite(output_path, cv2.cvtColor(target_np, cv2.COLOR_BGR2RGB))
      modules.globals.target_path = output_path
      source_np = np.array(source_face)
      # Save images as JPG
      if source_np is None:
         source_np = target_np
      cv2.imwrite(source_path, cv2.cvtColor(source_np, cv2.COLOR_BGR2RGB))
      deepfake_image(source_path,output_path,output_path,enhance)
      # Convert target image to correct format for display
      target_np =  cv2.imread(output_path)
      target_display = Image.fromarray(cv2.cvtColor(target_np, cv2.COLOR_BGR2RGB))
      
      if os.path.exists(output_path):
        os.remove(output_path)
      if os.path.exists(source_path):
        os.remove(source_path)
      return "Images deepfaked successfully!", target_display,gr.update(visible=False),None,None
  def update_map(current_index,source_image):
      opencv_source = source_image
      # Convert RGB to BGR (OpenCV uses BGR instead of RGB)
      opencv_source = cv2.cvtColor(opencv_source, cv2.COLOR_RGB2BGR)
      face = get_one_face(opencv_source)
      x_min, y_min, x_max, y_max = face["bbox"]
      modules.globals.source_target_map[current_index]["source"]={
                  "cv2": opencv_source[int(y_min): int(y_max), int(x_min): int(x_max)],
                  "face": face,
              }
  def next_map(current_index,source_image):
    if len(modules.globals.source_target_map)>current_index+1:
      current_index +=1
    source = modules.globals.source_target_map[current_index].get("source",None)
    if source is not None:
      source = Image.fromarray(cv2.cvtColor(modules.globals.source_target_map[current_index]["source"]["cv2"], cv2.COLOR_BGR2RGB))
    target =  Image.fromarray(cv2.cvtColor(modules.globals.source_target_map[current_index]["target"]["cv2"], cv2.COLOR_BGR2RGB))
    return current_index,source,target
  def prev_map(current_index,source_image):
    if current_index>0:
      current_index -=1
    source = modules.globals.source_target_map[current_index].get("source",None)
    if source is not None:
      source = Image.fromarray(cv2.cvtColor(modules.globals.source_target_map[current_index]["source"]["cv2"], cv2.COLOR_BGR2RGB))

    target =  Image.fromarray(cv2.cvtColor(modules.globals.source_target_map[current_index]["target"]["cv2"], cv2.COLOR_BGR2RGB))
    return current_index,source,target
class video_processes:
  def ui():
      map_index_vid=gr.State(0)
      with gr.Row():
          source_face_input = gr.Image(label="Select Source Face", type="pil")
          target_video_input = gr.Video(label="Upload Target Video")
      with gr.Row():
        keep_fps_video = gr.Checkbox(label="Keep FPS", value=False)
        keep_audio_video = gr.Checkbox(label="Keep Audio", value=False)
      with gr.Row():
        many_face_video = gr.Checkbox(label="Many Faces", value=False)
        map_faces_video = gr.Checkbox(label="Map Faces", value=False)
      with gr.Row():
        enhance_face_video = gr.Checkbox(label="Enhance Face", value=False)
      process_button = gr.Button("Process Video")
      with gr.Column(visible=False,elem_id="face_map_column_vid") as map_col_vid:
        with gr.Row():
          mapping_source_vid = gr.Image(label="Select Source Face", type="numpy")
          mapping_target_vid = gr.Image(label="Select Target Image", type="numpy", interactive=False)
        with gr.Row():
          previous_button_vid = gr.Button("previous",elem_classes="previous_button")
          submit_button_vid = gr.Button("submit",elem_classes="submit_button")
          next_button_vid = gr.Button("next",elem_classes="next_button")
      with gr.Column():
          output_video = gr.Video(label="Processed Video", interactive=False)
          video_status_textbox = gr.Textbox(label="Status", interactive=False)
      
      mapping_source_vid.upload(
          image_processes.update_map,
          inputs=[map_index_vid,mapping_source_vid]
      )
      process_button.click(
          video_processes.process_video,
          inputs=[source_face_input, target_video_input,keep_fps_video,keep_audio_video,many_face_video,map_faces_video,enhance_face_video],
          outputs=[output_video,video_status_textbox,map_col_vid,mapping_target_vid]
      )
      
      next_button_vid.click(
          image_processes.next_map,
          inputs=[map_index_vid,mapping_source_vid],
          outputs=[map_index_vid,mapping_source_vid,mapping_target_vid]
      )
      previous_button_vid.click(
          image_processes.prev_map,
          inputs=[map_index_vid,mapping_source_vid],
          outputs=[map_index_vid,mapping_source_vid,mapping_target_vid]
      ) 
      submit_button_vid.click(
        video_processes.submit_mapping,
        inputs=[source_face_input, target_video_input, video_status_textbox,enhance_face_video,map_faces_video],
        outputs=[video_status_textbox,output_video,map_col_vid,mapping_target_vid,mapping_source_vid]
      )
  
  def submit_mapping(source_face, target_video, status_textbox,enhance,map_faces):
      now = datetime.now()
      formatted_now = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Truncate to 3 decimal places for milliseconds
      random_no = get_random_number()
      source_path = f"temp_source_output/source-{random_no}-{formatted_now}.jpeg" 
      input_path = f"temp_source_output/input-{random_no}-{formatted_now}.mp4" 
      output_path = f"temp_source_output/output-{random_no}-{formatted_now}.mp4"
      modules.globals.output_path = output_path
      modules.globals.target_path = input_path
      modules.globals.source_path= source_path
      shutil.copy2(target_video,input_path)
      # Save files
      cv2.imwrite(source_path, cv2.cvtColor(np.array(source_face), cv2.COLOR_RGB2BGR))
      deep_fake_video(source_path,input_path,output_path,enhance)
      return output_path,output_path,gr.update(visible=False),None,None
  
  def process_video(source_face, target_video,keep_fps,keep_audio,many_faces,map_faces,enhance):
      """
      Process an uploaded video using Deep-Live-Cam.
      """
      if target_video is None:
          return None,"No source face or target video uploaded!",gr.update(visible=False),None
      modules.globals.keep_fps = keep_fps
      modules.globals.keep_audio = keep_audio
      modules.globals.many_faces = many_faces
      modules.globals.map_faces = map_faces
      now = datetime.now()
      random_no = get_random_number()
      formatted_now = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Truncate to 3 decimal places for milliseconds
      source_path = f"temp_source_output/source-{random_no}-{formatted_now}.jpeg" 
      input_path = f"temp_source_output/input-{random_no}-{formatted_now}.mp4" 
      output_path = f"temp_source_output/output-{random_no}-{formatted_now}.mp4"
      modules.globals.output_path = output_path
      modules.globals.target_path = input_path
      modules.globals.source_path= source_path
      if modules.globals.map_faces:
        get_unique_faces_from_target_video()
        return None,"Please map faces",gr.update(visible=True),cv2.cvtColor(modules.globals.source_target_map[0]["target"]["cv2"], cv2.COLOR_BGR2RGB)
      shutil.copy2(target_video,input_path)
      # Save files
      cv2.imwrite(source_path, cv2.cvtColor(np.array(source_face), cv2.COLOR_RGB2BGR))
      deep_fake_video(source_path,input_path,output_path,enhance)
      return output_path,output_path,gr.update(visible=False),None
 
def resize_image(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    """
    Resize an image while maintaining the aspect ratio.
    
    :param image: Input image (NumPy array).
    :param width: Desired width (optional).
    :param height: Desired height (optional).
    :param interpolation: Interpolation method (default: cv2.INTER_AREA for downscaling).
    :return: Resized image.
    """
    # Get original dimensions
    (h, w) = image.shape[:2]

    # If both width and height are None, return original image
    if width is None and height is None:
        return image

    # Calculate new dimensions while maintaining aspect ratio
    if width is None:
        ratio = height / float(h)
        new_dimensions = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        new_dimensions = (width, int(h * ratio))

    # Resize the image
    resized = cv2.resize(image, new_dimensions, interpolation=interpolation)
    return resized
class live_processes:
    def ui():
      
      source_face_live = gr.Image(label="Select Source Face", type="numpy")
      input_img = gr.Image(sources=["webcam"], type="numpy", width=300,height=300) 
       
      with gr.Row():
        enhance_live = gr.Checkbox(label="Enhance", value=False)
        mouth_live = gr.Checkbox(label="Mouth Mask", value=False) 
      
      output_img = gr.Image(streaming=True,width=300,height=300)
      input_img.stream(
          fn=live_processes.process_frame,
          inputs=[source_face_live,input_img, enhance_live,mouth_live],
          outputs=[output_img],time_limit=30, stream_every=0.08, concurrency_limit=30
      )
    def process_frame(source_face, target_image,enhance,mouth_mask):
        """
        Save source and target images as JPG and display the uploaded target image instead of processing.
        """
        modules.globals.map_faces=False
        modules.globals.source_target_map = []
        modules.globals.many_faces=False
        target_np =  resize_image(target_image,300,300) 
          
        modules.globals.mouth_mask = mouth_mask  
  
        if source_face is None:
          source_face = target_np 
        source_np = resize_image(source_face,300,300) 
        target_display =deep_fake_live(source_np,target_np,enhance)  
        return target_display


css='''
.next_button{
  background-color:green;
  color:white;
}
.previous_button{
  background-color:#fac125;
  color:white;
}
.submit_button{

}
'''
# Gradio Interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Deep Live Cam")


    with gr.Tab("Image Mode"):
      image_processes.ui()
    with gr.Tab("Video Mode"):
      video_processes.ui()
    with gr.Tab("Live Mode"):
      live_processes.ui()
# Launch the UI
demo.launch(share=False,debug=True, server_name="0.0.0.0")
