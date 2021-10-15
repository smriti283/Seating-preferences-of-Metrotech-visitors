# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=10,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers_c = []
trackers_p = []
trackers = []
labels = [] #added labels list initialization

trackableObjects_c = {}
trackableObjects_p = {}
rect_c =[]
rect_p =[]
boxes_c = []
boxes_p =[]
counter = 0

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
	status = "Waiting"
	rects_p = []
	rects_c = []

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % args["skip_frames"] == 0:
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers_c = []
		trackers_p = []
        
		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame.copy(), 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])
				label = CLASSES[idx] #new addition

				# if the class label is not a person, ignore it #new addition
				if (CLASSES[idx] != "person") & (CLASSES[idx]!= "chair") & (CLASSES[idx]!= "sofa"): 
					continue

				if (CLASSES[idx] == "person"): 
					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startXp, startYp, endXp, endYp) = box.astype("int")

					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(int(startXp), int(startYp), int(endXp), int(endYp))
					tracker.start_track(rgb, rect)
					rect_p = box
					boxes_p.append(rect_p)                    

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers_p.append(tracker)
					labels.append(label) #new addition

                
					# new addition - grab the corresponding class label for the detection
					# and draw the bounding box
					cv2.rectangle(frame, (startXp, startYp), (endXp, endYp),
						(0, 255, 0), 2)
					if (label == 'person'):
						cv2.putText(frame, label, (startXp, startYp - 15),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    
				if (CLASSES[idx] == "chair") | (CLASSES[idx] == "sofa"): 
					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startXc, startYc, endXc, endYc) = box.astype("int")

					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(int(startXc), int(startYc), int(endXc), int(endYc))
					tracker.start_track(rgb, rect)
					rect_c = box
					boxes_c.append(rect_c)                    

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers_c.append(tracker)
					labels.append(label) #new addition                       
                
					# new addition - grab the corresponding class label for the detection
					# and draw the bounding box
					cv2.rectangle(frame, (startXc, startYc), (endXc, endYc),
						(0, 255, 0), 2)
					if (label == 'chair') | (label == 'sofa'):                  
						cv2.putText(frame, label, (startXc, startYc - 15),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

			#for i in enumerate(boxes_p, boxes_c):
					#if ((int(boxes_p[i][0]) ==  int(boxes_c[i][0])) | (int(boxes_p[i][0]) ==  int(boxes_c[i][0]+1)) |  
						#(int(boxes_p[i][0]) ==  int(boxes_c[i][0]-1))) & ((int(boxes_p[i][3]) ==  int(boxes_c[i][3])) | 
						#(int(boxes_p[i][3]) ==  int(boxes_c[i][3]+1)) | (int(boxes_p[i][3]) ==  int(boxes_c[i][3]-1))):           
							#counter += 1
				#print(counter)                   
                    
	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for trackerp in trackers_p:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking_p"

			# update the tracker and grab the updated position
			trackerp.update(rgb)
			pos = trackerp.get_position()

			# unpack the position object
			startXp = int(pos.left())
			startYp = int(pos.top())
			endXp = int(pos.right())
			endYp = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects_p.append((startXp, startYp, endXp, endYp))
            
            # new addition- draw the bounding box from the correlation object tracker
			cv2.rectangle(frame, (startXp, startYp), (endXp, endYp),
				(0, 255, 0), 2)
			if (label == 'person'):
				cv2.putText(frame, label, (startXp, startYp - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
		# loop over the trackers
		for trackerc in trackers_c:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			trackerc.update(rgb)
			pos = trackerc.get_position()

			# unpack the position object
			startXc = int(pos.left())
			startYc = int(pos.top())
			endXc = int(pos.right())
			endYc = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects_c.append((startXc, startYc, endXc, endYc))
            
            # new addition- draw the bounding box from the correlation object tracker
			cv2.rectangle(frame, (startXc, startYc), (endXc, endYc),
				(0, 255, 0), 2)
			if (label == 'chair'):            
				cv2.putText(frame, label, (startXc, startYc - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)            

	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they were
	# moving 'up' or 'down'
	cv2.line(frame, (0, (H-100)//2),((W+500)//2, H-120), (0, 255, 255), 2)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects_p = ct.update(rects_p)
	objects_c = ct.update(rects_c)

	# loop over the tracked objects
	for (objectID, centroid) in objects_p.items():
	# check to see if a trackable object exists for the current
	# object ID
		to_p = trackableObjects_p.get(objectID, None)

	# if there is no existing trackable object, create one
		if to_p is None:
			to_p = TrackableObject(objectID, centroid)

	# otherwise, there is a trackable object so we can utilize it
	# to determine direction
		else:
		# the difference between the y-coordinate of the *current*
		# centroid and the mean of *previous* centroids will tell
		# us in which direction the object is moving (negative for
		# 'up' and positive for 'down')
			y = [c[1] for c in to_p.centroids]
			direction = centroid[1] - np.mean(y)           
			to_p.centroids.append(centroid)            

		# check to see if the object has been counted or not
		#if not to_p.counted:
			# if the direction is negative (indicating the object
			# is moving up) AND the centroid is above the center
				# line, count the object
			#if direction < 0 and centroid[1] < (H-100) // 2:
				#totalUp += 1
				#to_p.counted = True

			# if the direction is positive (indicating the object
			# is moving down) AND the centroid is below the
			# center line, count the object
			#elif direction > 0 and centroid[1] > (H-100) // 2:
				#totalDown += 1
				#to_p.counted = True
                
		# store the trackable object in our dictionary
		trackableObjects_p[objectID] = to_p                
        
        
	# loop over the tracked objects
	for (objectID, centroid) in objects_c.items():
		# check to see if a trackable object exists for the current
		# object ID
		to_c = trackableObjects_c.get(objectID, None)

		# if there is no existing trackable object, create one
		if to_c is None:
			to_c = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y = [c[1] for c in to_c.centroids]
			direction = centroid[1] - np.mean(y)           
			to_c.centroids.append(centroid)

			# check to see if the object has been counted or not
			#if not to_c.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
				#if direction < 0 and centroid[1] < (H-100) // 2:
					#totalUp += 1
					#to_c.counted = True

				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				#elif direction > 0 and centroid[1] > (H-100) // 2:
					#totalDown += 1
					#to_c.counted = True

		# store the trackable object in our dictionary
		trackableObjects_c[objectID] = to_c   

	#clist = list(objects_c.items())
	#clist_cent = []    
	#for i in range(len(clist)):
		#clist_cent.append(clist[1])       
   
        
	#cv2.circle(frame, (W-250,H-150), 4, (0, 255, 0), -1)
	#cv2.circle(frame, (W-250,H-100), 4, (0, 255, 0), -1)
	#cv2.circle(frame, (W-250,H-200), 4, (0, 255, 0), -1)
	#cv2.circle(frame, (W-250,H-180), 4, (0, 255, 0), -1)
	for objectIDc, centroid_c in objects_c.items():
		#Cx = centroid_c[0]  
		#Cy = centroid_c[1]
		#to_c = TrackableObject(objectIDc, centroid_c)        
		for objectIDp, centroid_p in objects_p.items():
			#to_p = TrackableObject(objectIDp, centroid_p)
			if (centroid_c[0] == centroid_p[0]) & (centroid_c[1] == centroid_p[1]) :
				if (to_p.counted == False) & (to_c.counted == False) & (centroid_p[1] < (H-150)):
					totalUp += 1                
					to_p.counted = True
					to_c.counted = True
				if (to_p.counted == False) & (to_c.counted == False) & (centroid_p[1] > (H-180)):
					totalDown += 1                
					to_p.counted = True
					to_c.counted = True 


		# draw both the ID of the object and the centroid of the
		# object on the output frame
		#text = "ID {}".format(objectID)
		#cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		#cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
		#cv2.rectangle(frame, (centroid[0]-20, centroid[1]-40), (centroid[0]+20, centroid[1]+40), (0, 255, 0), 2) #added line

        

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Sunlit", totalUp),
		("Shade", totalDown),
		("Status", status),
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()
