#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os


class MyYolo:
    """
        Implements YOLOv3 object detection algorithm. This code is edited from https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
    """
    
    # constants
    NAMES_FILE = 'D:/our yolo v3/model/classes.names'
    WEIGHTS_FILE = 'D:/our yolo v3/model/yolov3_custom_last.weights'
    CFG_FILE = 'D:/our yolo v3/model/yolov3_custom.cfg'
    
    
    def __init__(self, model_path, min_confidence=0.5, nms_thresh=0.3,
                 object_names=None):
        """
            - model_path (str): path to folder containing .names, .cfg, 
              and .weights files
            - min_confidence (float): Minimum confidence to predict an object
            - nms_thresh (float): Non-Maximum Suppression threshold to prevent
              detection of multiple boxes for the same object
            - object_names (list<str> or None): A list of object names to detect.
              If None, all types of objects from NAMES_FILE will be detected
        """
        
        # min_confidence and nms_thresh
        self.min_confidence = min_confidence
        self.nms_thresh = nms_thresh
        
        # get labels and class_idx_filter
        labelsPath = os.path.join(model_path, MyYolo.NAMES_FILE)
        self.labels = open(labelsPath).read().strip().split("\n")
        self.class_idx_filter = []
        if object_names:
            for name in object_names:
                if name not in self.labels:
                    raise ValueError('Unknown object "{}"'.format(name))
                else:
                    idx = self.labels.index(name)
                    self.class_idx_filter.append(idx)

        # generate random colors for classes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), 
                                        dtype="uint8")
        
        # paths to the YOLO weights and model configuration
        weightsPath = os.path.join(model_path, MyYolo.WEIGHTS_FILE)
        configPath = os.path.join(model_path, MyYolo.CFG_FILE)
        
        # load our YOLO object detector trained on COCO dataset (80 classes)
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    
    def predict(self, image, is_bgr=True):
        """
            - image (numpy.ndarray): input image
            - is_bgr: whether the input image is BGR. If False, the image
              is assumed to be RGB
            
            - returns two values: 
                (1)   predictions (list<dict>): Each item is a dictionary
                      that contains 3 keys: "class_name", "box", "confidence", 
                      where "box" is a list of [x, y, width, height]
                (2)   output_image (numpy.ndarray): output image with
                      predictions drawn
        """
        (H, W) = image.shape[:2]
        # determine only the *output* layer names that we need from YOLO
        ln = [layer_name for layer_name in self.net.getUnconnectedOutLayersNames()]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=is_bgr, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(ln)
        
        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                # Note: detection contains: [xcenter, ycenter, width, height, 
                #                             probOfHasObject, ...scores...]
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # select only names from class_filter (if given)
                if self.class_idx_filter and classID not in self.class_idx_filter:
                    continue
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.min_confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    
        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.min_confidence, 
                                self.nms_thresh)
        
        # ensure at least one detection exists
        predictions = []
        out_image = image.copy()
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                pred = {}
                pred['box'] = boxes[i]
                pred['class_name'] = self.labels[classIDs[i]]
                pred['confidence'] = confidences[i]
                predictions.append(pred)
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(out_image, (x, y), (x + w, y + h), color, 4)
                text = "{}: {:.4f}".format(self.labels[classIDs[i]], confidences[i])
                cv2.putText(out_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)
        
        # return predictions and output image
        return predictions, out_image
    
    
    def predict_from_cam(self):
        """
            Start object detection from Web cam. Press 'q' to stop.
        """
        
        # define a video capture object
        vid = cv2.VideoCapture(0)
          
        while(True):
              
            # Capture the video frame by frame
            ret, frame = vid.read()
            
            # make predictions
            preds, frame = self.predict(frame)
          
            # Display the resulting frame
            cv2.imshow('yolo', frame)
              
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
          
        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
        
        
    def predict_from_video(self, src_path, dst_path):
        """
            Start object detection from Web cam. Press 'q' to stop.
        """
        
        # define a video capture object
        vid = cv2.VideoCapture(src_path)
        
        # initialize writer
        writer = None
        
        print('Processing video...')
          
        while(True):
              
            # Capture the video frame by frame
            ret, frame = vid.read()
            
            # check end of video
            if not ret:
                break
            
            # make predictions
            preds, frame = self.predict(frame)
          
            # check if the video writer is None
            if writer is None:
        		# initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(dst_path, fourcc, 30,
        			(frame.shape[1], frame.shape[0]), True)

        	# write the output frame to disk
            writer.write(frame)
          
        # After the loop release the cap object and writer
        vid.release()
        writer.release()
        
        print('Video saved to "{}'.format(dst_path))


# In[ ]:





# In[ ]:




