
from math import *
import math
from MathFunctions import*

class Parameters:
    '''
    This class all basic and initial parameters to run the simulation.
    Attention to update the data according to the information and technical characteristics of the components used
    '''
    
    ##Car Features
    Car_length = 0.4      #meters
    Car_width = 0.19      #meters
    Car_maxspeed = 0.6
    Car_minspeed = 0.2
    Car_speed = 2 #m/s
    radius_margin = 0.01         #Safety margin for vehicle radius (meters)
    Car_radius =sqrt((Car_length/2)**2+(Car_width/2)**2) + radius_margin #Radius of the vehicle. A circle is created with the "car inside" so that nothing goes beyond that circle. 
    Max_acceleration = 0.5

    #LIDAR Features
    Lidar_steps=360              #Numbers of data that LIDAR receives in one complete turn
    Lidar_delta=1/10         #Time taken by the sensor to read the N points
    Lidar_stepsize = 360/Lidar_steps       #Angled representation of each LIDAR step
    Lidar_maxdistance = 8     #Maximum distance that the handle can capture an object(meters)

    epsilonmax=45
    tsb=0.1  

    
parameters = Parameters()


#Make the circuit and asll obstacles. This function is used only for simulate the real world
def MakeEnv():
   
    #circuitext=[[0,-3.5],[0.5,-1.5],[1.5,-0.5],[3.5,0],[8,0],[8.5,-0.5],[8.5,-3],[9,-3.5],[12,-3.5],[13,-3.5],[14,-4],[14.5,-5],[14.5,-8.5], [14,-9.5],[12.5,-10],[3,-10],[1.5,-9.5],[0.5,-8.5],[0,-6.5],[0,-3.5]]
    #circuitint=[[2.5,-3.5],[3.5,-2.5],[6,-2.5],[6.5,-3],[6.5,-5.5],[7,-6],[11.5,-6],[12,-6.5],[12,-7],[11.5,-7.5], [3.5,-7.5],[2.5,-6.5],[2.5,-3.5]]
    circuitext=[[3,0],[12,0],[14,-1],[14,-5],[16,-8],[15,-11],[15,-14],[14,-16],[1,-16],[1,-2],[3,0]]
    circuitint= [[3.5,-2],[12,-2],[12,-5],[14,-8],[14,-11],[12,-14],[4,-14],[2,-12],[2,-5],[3.5,-2]]
   
    obst1=[[4.5,-8.5],[6,-8.5],[6,-9],[4.5,-9],[4.5,-8.5]]
    envir=[circuitext,circuitint]
    position = [5,-1.5]
    orientation = 0
    return(envir, position, orientation)

#This function simulates real data acquisition from LIDAR and must be replaced in the algorithm that will be placed in the car
def GetLidarSimulation(environment,position,orientation):

    """
    This function simulates the operation of the LIDAR by returning a list with the polar coordinates of 
    the location of the obstacles

    enviroment    -> List with all objects that are in the circuit as well as the edges of the circuit
    position      -> Car position [x,y]
    orientation   -> Car orientation (0,360)

    Return

    Lidar_Data    : [step_i, distance_i]      i(0, Lidar_Steps)
    """

    n=len(environment)
    LidarData=[]
    i=0
    #Going through all the sensor steps
    while i<parameters.Lidar_steps:
        j=0
        intersectparliste=[]
        #Going through all objects and edges
        while j<n:
            k=0
            
            intersect=[]
            while k<(len(environment[j])-1):
                [res,inter]=Intersection(position,i*parameters.Lidar_stepsize,environment[j][k]+environment[j][k+1]) 
                
                if res==1:
                    segment=environment[j][k]+environment[j][k+1]
                    d=[(inter[0])**2+(inter[1])**2]
                    contenu=segment+inter+d
                    
                    if (inter[0])*(cos(i*parameters.Lidar_stepsize*2*pi/360))+(inter[1])*(sin(i*parameters.Lidar_stepsize*2*pi/360))>=0:
                            intersect.append(contenu)
                k+=1
                
            if len(intersect)!=0:
                intersect.sort(key=lambda x: x[6])
                intersectparliste.append(intersect[0]) #The closest intersection
            j+=1
            
        intersectparliste.sort(key=lambda x: x[6])

        try:
          r0=sqrt(intersectparliste[0][6])
        except IndexError:
          r0 = 0
        
        alpha0=i
        
        #Ensuring that LIDAR only reads what is at its physical limit
        if(r0 > parameters.Lidar_maxdistance): r0 = parameters.Lidar_maxdistance
        
        #Calibration of the orientation of M: the first line of M must correspond to orientation
        LidarData.append([(alpha0-orientation/parameters.Lidar_stepsize)%parameters.Lidar_steps,r0])
        i+=1
    LidarData.sort(key=lambda x: x[0])
    return(LidarData)

#This function simulates the movement of the car and updates the position and orientation
def UpdateCar(inverse_R,Speed,position,orientation):

    Speed = Speed * parameters.Car_speed
    R = 1/(inverse_R + 0.00000001)
    
    angle_wheels = math.asin(parameters.Car_length/R)
    beta = math.atan(math.tan(angle_wheels)/2)

    dx = Speed * math.cos(beta+math.radians(orientation))
    dy = Speed * math.sin(beta+math.radians(orientation))
    dtheta = 2*Speed*math.sin(beta)/parameters.Car_length

    x = position[0] + dx * parameters.Lidar_delta
    y = position[1] + dy * parameters.Lidar_delta
    theta = math.radians(orientation)+dtheta * parameters.Lidar_delta

    NewPosition= [x,y]
    NewOrientation = math.degrees(theta)

    return (NewPosition, NewOrientation, angle_wheels)
