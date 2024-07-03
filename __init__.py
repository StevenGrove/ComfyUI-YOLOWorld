import os
os.system('pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless')
os.system('pip install opencv-python==4.7.0.72')

from . import yolo_world


NODE_CLASS_MAPPINGS = {**yolo_world.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**yolo_world.NODE_DISPLAY_NAME_MAPPINGS}

