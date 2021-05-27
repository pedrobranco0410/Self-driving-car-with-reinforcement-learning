#the code used in the race in April 2021

#Enviroment
import numpy as np
from math import *
import SensorModel
import math
import time
import pigpio

ESC = 4
pi = pigpio.pi()

def set_speed(vitesse, sens = 1):
    pulse = 1500+sens*vitesse*5 #pulse between 1000 and 2000, 1500 is neutral
    pi.set_servo_pulsewidth(ESC, pulse)

class Parameters:
    '''
    This class all basic and initial parameters to run the simulation.
    Attention to update the data according to the information and technical characteristics of the components used
    '''
    
    ##Car Features
    Car_length = 0.4      #meters
    Car_width = 0.19      #meters
    Car_maxspeed = 30
    Car_minspeed = 20 
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

class CarControl():

    def __init__(self):

        #State variables
        self.Speed = parameters.Car_minspeed #m/s
        self.inverse_Radious = 0
        self.LidarData = SensorModel.getLidar()

        self.done = False

    # ------------------------ control ------------------------

    def GetCar(self):
      return [self.Speed,self.inverse_Radious]




    def step(self):

        self.done = 0 
        self.inverse_Radious = 0

        r_30 = 0
        r_60 = 0
        r_90 = 0

        l_30 = 0
        l_60 = 0
        l_90 = 0
        
        front = 0
        for i in range(len(self.LidarData)-1):

          if (i < 30):
              l_30 += self.LidarData[i][1]/30
          elif (i < 60):
              l_60 += self.LidarData[i][1]/30
          elif ( i < 90):
              l_90 += self.LidarData[i][1]/30
          
          if (i > 269 and i < 300):
            r_90 += self.LidarData[i][1]/30
          elif (i > 299 and i < 330):
            r_60 += self.LidarData[i][1]/30
          elif (i > 229 ):
            r_30 += self.LidarData[i][1]/30
          if( i < 10 or i > 349):
              front += self.LidarData[i][1]/20
          
        if (r_90 > l_90):
          self.inverse_Radious += max(-1 , l_90-r_90)
        else:
          self.inverse_Radious += min(1 , l_90-r_90)

        if (r_60 > l_60):
          self.inverse_Radious += max(-1 , (l_60-r_60)*0.1)
        else:
          self.inverse_Radious += min(1 , (l_60-r_60)*0.1)
        
        if (r_30 > l_30):
          self.inverse_Radious += max(-1 , (l_60-r_60)*0.01)
        else:
          self.inverse_Radious += min(1 , (l_60-r_60)*0.01)
          
        if(front < 1 and r_90 > l_90):
            self.inverse_Radious = -1
        if (front < 1 and r_60 < l_60):
            self.inverse_Radious = 1
            
        if (r_90 > 3*l_90):
          self.inverse_Radious = -1
        elif(r_90*3 < l_90):
          self.inverse_Radious = 1
        
        self.Speed = parameters.Car_minspeed + min(10, front*2)

        for i in range(len(self.LidarData)):
            if self.LidarData[i][1] <= parameters.Car_radius:
                self.done = True
        # Car moving
        
        R = 1/(self.inverse_Radious + 0.00000001)
    
        angle_wheels = math.degrees(math.asin(parameters.Car_length/(R+0.000001)))

        SensorModel.turnServo(self.inverse_Radious, angle_wheels)
        set_speed(self.Speed)

        #Updating LIDAR
        self.LidarData = SensorModel.getLidar()
        
        
        

