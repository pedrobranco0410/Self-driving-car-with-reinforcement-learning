#Enviroment
import numpy as np
from math import *
import SensorModel

class Parameters:
    '''
    This class all basic and initial parameters to run the simulation.
    Attention to update the data according to the information and technical characteristics of the components used
    '''
    
    ##Car Features
    Car_length = 0.4      #meters
    Car_width = 0.19      #meters
    Car_maxspeed = 30/3.6
    Car_minspeed = 3.6/3.6 # meters/s
    radius_margin = 0.1         #Safety margin for vehicle radius (meters)
    Car_radius =sqrt((Car_length/2)**2+(Car_width/2)**2) + radius_margin #Radius of the vehicle. A circle is created with the "car inside" so that nothing goes beyond that circle. 
    Max_acceleration = 0.5
    Max_CurvatureRadious = 25
    Min_CurvatureRadious = 1

    #LIDAR Features
    Lidar_steps=360              #Numbers of data that LIDAR receives in one complete turn
    Lidar_delta=1/10             #Time taken by the sensor to read the N points
    Lidar_stepsize = 360/Lidar_steps       #Angled representation of each LIDAR step
    Lidar_maxdistance = 1000     #Maximum distance that the handle can capture an object(meters)

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

    return (Turn_Angle)



class CarControl():

    def __init__(self):

        #State variables
        self.Speed = 4 #m/s
        self.Radious = parameters.Min_CurvatureRadious #meters
        self.AngleWheels = 0 #degrees
        self.Direction = 1
        self.LidarData = getLidar()

        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0
        self.Step = 0
        self.red = 0

    # ------------------------ AI control ------------------------

    def GetCar(self):
      return [self.Speed,self.AngleWheels]

    def getState():
      lista = []
        for i in range(len(self.LidarData)):
          if (i < 90 or i >270):
            lista += [self.LidarData[i][1]]

      state = [lista + [self.Radious,self.Direction]]
      return state


    def step(self, action, ep):

        if (action == 0):
          self.Radious = min(parameters.Max_CurvatureRadious, self.Radious + 1)

        if (action == 1):
          self.Radious = min(parameters.Max_CurvatureRadious, self.Radious + 5)

        if (action == 2):
          self.Radious = max(parameters.Min_CurvatureRadious, self.Radious - 1)

        if (action == 3):
          self.Radious = max(parameters.Min_CurvatureRadious, self.Radious - 5)
        
        if (action == 4):
          self.Direction *= -1

        self.AngleWheels= UpdateCar(abs(self.Radious), self.Direction, self.Speed)

        turnServo(AngleWheels)

        lista = []
        for i in range(len(self.LidarData)):
          if (i < 90 or i >270):
            lista += [self.LidarData[i][1]]

        state = [lista + [self.Radious,self.Direction]]
        return  state
