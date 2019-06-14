import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import os
import sys
import coco
from mmrcnn import utils
from mmrcnn import model as modellib

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mobile_mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [ 'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
				'bus', 'train', 'truck', 'boat', 'traffic light',
				'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
				'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
				'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
				'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
				'kite', 'baseball bat', 'baseball glove', 'skateboard',
				'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
				'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
				'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
				'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
				'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
				'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
				'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
				'teddy bear', 'hair drier', 'toothbrush'
			  ]

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.flag = False

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack(anchor=tkinter.CENTER, expand=True)

        self.btn_start = tkinter.Button(window, text = "Start" , width = 25 , command = self.start_maskrcnn)
        self.btn_start.pack(anchor=tkinter.CENTER, expand=True)

        self.btn_stop = tkinter.Button(window, text = "Stop" , width = 25 , command = self.stop_maskrcnn)
        self.btn_stop.pack(anchor=tkinter.CENTER, expand=True)

        self.btn_quit = tkinter.Button(window, text = "Quit" , width = 25 , command = self.close_window)
        self.btn_quit.pack(anchor=tkinter.CENTER, expand=True)
		
		# After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()
        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        if self.flag:
        	ret, frame = self.vid.get_frame()

        	if ret:
        		self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        		self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)

    def start_maskrcnn(self):
    	self.flag = True

    def stop_maskrcnn(self):
    	self.flag = False

    def close_window(self):
    	exit()

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(cv2.CAP_DSHOW)
        # if not self.vid.isOpened():
        #     raise ValueError("Unable to open video source", video_source)

        # self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            
            if ret:
            	# Return a boolean success flag and the current frame converted to BGR
            	frame = cv2.flip(frame,1)
            	results = model.detect([frame], verbose=0)
            	r = results[0]
            	frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
            	return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            else:
            	return (ret, None)
        else:
        	return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "R.O.D.I.S")