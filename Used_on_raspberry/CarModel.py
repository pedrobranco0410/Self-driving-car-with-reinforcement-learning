#Enviroment
import numpy as np
from math import *
import SensorModel
import math

class Parameters:
    '''
    This class all basic and initial parameters to run the simulation.
    Attention to update the data according to the information and technical characteristics of the components used
    '''
    
    ##Car Features
    Car_length = 0.4      #meters
    Car_width = 0.19      #meters
    Car_maxspeed = 0.3
    Car_minspeed = 0.18 
    Car_speed = 5 #m/s
    radius_margin = 0.01         #Safety margin for vehicle radius (meters)
    Car_radius =sqrt((Car_length/2)**2+(Car_width/2)**2) + radius_margin #Radius of the vehicle. A circle is created with the "car inside" so that nothing goes beyond that circle. 
    Max_acceleration = 0.5
    Max_CurvatureRadious = 30
    Min_CurvatureRadious = 1

    #LIDAR Features
    Lidar_steps=360              #Numbers of data that LIDAR receives in one complete turn
    Lidar_delta=1/3             #Time taken by the sensor to read the N points
    Lidar_stepsize = 360/Lidar_steps       #Angled representation of each LIDAR step
    Lidar_maxdistance = 8     #Maximum distance that the handle can capture an object(meters)

    epsilonmax=45
    tsb=0.1  

    
parameters = Parameters()


def UpdateCar(R,direction,Speed):

    epsilon=(parameters.tsb*Speed**2/R+parameters.Car_length/R)
    
    #If it's a reverse course
    if Speed < 0:
        epsilon=parameters.epsilonmax*pi/180
        

    #If the car turns to left
    if direction<0:
        epsilon=-epsilon
    
    #Calculate the real curvature radius
    R_real=abs(parameters.tsb*Speed**2/(epsilon)+parameters.Car_length/(epsilon)) 
    
    thetaparc=0
    
    
    #If speed differs from 0 then the point must be calculated considering the speed
    if Speed!=0:
        
        thetaparc=Speed*parameters.Lidar_delta/R_real
    
        #If the car turns to left
        if direction<0:       
            thetaparc=-thetaparc
               

        

    Turn_Angle = thetaparc*360/(2*pi)            #It will be used to set the angle of the wheels

    return (R_real)



class CarControl():

    def __init__(self):

        #State variables
        self.Speed = parameters.Car_minspeed #m/s
        self.inverse_Radious = 0
        self.LidarData = SensorModel.getLidar()

        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0
        self.Step = 0
        self.red = 0

    # ------------------------ AI control ------------------------

    def GetCar(self):
      return [self.Speed,self.inverse_Radious]

    def getState(self):
      lista = []
      for i in range(len(self.LidarData)):
          if (i < 90 or i >270):
            lista += [self.LidarData[i][1]]

      state = [lista + [self.inverse_Radious,self.Speed]]
      return state


    def step(self, action):

        if (action == 0):
          self.inverse_Radious = min(-1, self.inverse_Radious - 0.1)

        if (action == 1):
          self.inverse_Radious = min(-1,self.inverse_Radious - 0.05)

        if (action == 2):
          self.inverse_Radious = min(1, self.inverse_Radious + 0.1)

        if (action == 3):
          self.inverse_Radious = min(1, self.inverse_Radious + 0.05)

        if (action == 4):
          self.inverse_Radious = 0
        
        right = 0
        left = 0
        front = 0
        for i in range(len(self.LidarData)):
            if(i > 30 and i <90):
                right += self.LidarData[i][1]/60
            if(i > 270 and i < 330):
                left += self.LidarData[i][1]/60
            if(i < 30 or i >330):
                front += self.LidarData[i][1]/60
        
        '''if(front > right and front > left):
            self.inverse_Radious = 0
        if (right >  left):
            self.inverse_Radious = 1 - 0.5*left/right
        if (left >  right):
            self.inverse_Radious = -1 + 0.5*right/left'''

        R = 1/(self.inverse_Radious + 0.00000001)
    
        angle_wheels = math.degrees(math.asin(parameters.Car_length/(R+0.000001)))

        SensorModel.turnServo(self.inverse_Radious, angle_wheels)

        self.LidarData = SensorModel.getLidar()
        lista = []
        for i in range(len(self.LidarData)):
          if (i < 90 or i >270):
            lista += [self.LidarData[i][1]]
        
        

        state = [lista + [self.inverse_Radious,self.Speed]]
        return  state
