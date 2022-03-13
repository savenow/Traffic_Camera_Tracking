from threading import Thread
import cv2
import time

from queue import Queue

class FileVideoStream():
	def __init__(self, path, queueSize=2000):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		self.w = int(self.stream.get(3))
		self.h = int(self.stream.get(4))
		self.stopped = False
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)
    
	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self, (self.w, self.h)

	def update(self):
		# keep looping infinitely
		while True:
			time.sleep(0.001)
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()
				if grabbed:
					vid_timer = int(self.stream.get(cv2.CAP_PROP_POS_MSEC))
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
				# add the frame to the queue
				self.Q.put((grabbed,frame,vid_timer))
	
	def read(self):
		# return next frame in the queue
		return self.Q.get()
	
	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0
	
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
	
	def set_timer(self, min_vid_timer):
		self.stream.set(cv2.CAP_PROP_POS_MSEC, min_vid_timer)