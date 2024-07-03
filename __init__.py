import os
os.system('pip uninstall -y opencv-python opencv-contrib-python '
          'opencv-python-headless')
os.system('pip install opencv-python==4.7.0.72')
os.environ['MODEL_CACHE_DIR'] = os.path.join(os.getcwd(), 'weights')

from . import yolo_world  # noqa


NODE_CLASS_MAPPINGS = {**yolo_world.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**yolo_world.NODE_DISPLAY_NAME_MAPPINGS}
