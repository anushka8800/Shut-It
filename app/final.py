import cv2
import mediapipe as mp
import math
import numpy as np
from plyer import notification
class abc:
    def __init__(self):
        

        # Initialize Mediapipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1,static_image_mode=True,min_detection_confidence=0.8)

        # Define drawing specs
        #self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

        self.FACE_CONNECTIONS = frozenset([
            #Lips upper inner
            (78, 191),
            (191, 80),
            (80, 81),
            (81, 82),
            (82, 13),
            (13, 312),
            (312, 311),
            (311, 310),
            (310, 415),
            (415, 308),
            (308, 324),

            #Lips lower inner
            (324, 318),
            (318, 402),
            (402, 317),
            (317, 14),
            (14, 87),
            (87, 178),
            (178, 88),
            (88, 95),
            (95, 78),
        ])

    def normalized(self,x,y,w,h):
        x= min(math.floor(x*w), w-1)
        y=min(math.floor(y*h), h-1)
        return (x,y)

    def getresponse(self,frame,s,c):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe FaceMesh
        results = self.face_mesh.process(frame_rgb)
        frame1=frame.copy()
        frame1=cv2.resize(frame1, (224, 100))
        binary=frame.copy()
        binary=cv2.resize(binary, (224, 100))
        #print(results)
        # Extract the mouth landmarks
        if results.multi_face_landmarks:
            #print("fjfrjvn")
            # Loop through each face
            for face_landmarks in results.multi_face_landmarks:
                index_for_conn = dict()
                points = list()
                # Loop through each mouth landmark and draw it on the frame
                for i, landmark in enumerate(face_landmarks.landmark):
                    x,y = landmark.x, landmark.y
                    x,y = self.normalized(x,y,frame.shape[1], frame.shape[0])
                    index_for_conn[i] = (x,y)
                    
                    for conn in self.FACE_CONNECTIONS:
                        start = conn[0]
                        end=conn[1]
                        
                        if start in index_for_conn and end in index_for_conn:
                            points.append(list(index_for_conn[start]))
                            points.append(list(index_for_conn[end]))
                            cv2.line(frame, index_for_conn[start], index_for_conn[end], (255, 255, 255), 2)
                            
                points=np.asarray(points)
                bbox=cv2.boundingRect(points)
                x,y,w,h=bbox
                
                frame1=frame[y-5:y+h+5, x-5:x+w+5].copy()
                frame1=cv2.resize(frame1, (224, 100))

                gray=cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)

                _, binary=cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                
                counters, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for each in counters[1:]:
                    frame1=cv2.drawContours(frame1, [each], -1, (255, 255, 255), -1)

                gray=cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                _, binary=cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

                counters, _= cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for each in counters[1:]:
                    if cv2.contourArea(each)>s:
                        frame1=cv2.drawContours(frame1, [each], -1, (0,0,255), -1)
                        c+=1

                        if c==30:
                            notification.notify(title = "YOU ARE BREATHING FROM YOUR MOUTH",
                                                message="Its a gentle reminder to stop mouth breathing" ,timeout=2)
                            c=0
                    else:
                        c=0

        return frame,frame1,binary,c