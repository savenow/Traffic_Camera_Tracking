import numpy as np
import math

class Angle:

    def findangle(self, points):    

        p1,p2,p3 = points[0], points[1], points[2]
        Ax, Ay = p2[0]-p1[0], p2[1]-p1[1]
        Cx, Cy = p3[0]-p1[0], p3[1]-p1[1]

        a = math.atan2(Ay, Ax)
        c = math.atan2(Cy, Cx)

        if a < 0: 
            a += math.pi*2
        if c < 0: 
            c += math.pi*2
        if a>c:
            angle = (math.pi*2 + c - a)
            angle = round(np.degrees(angle))
        else: 
            angle = (c - a)
            angle = round(np.degrees(angle))   
        
        return angle