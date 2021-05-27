#The simplified model used to run the car in April 2021

import argparse
import CarModel2
import sys
import time

def Simulate():

    
    env = CarModel2.CarControl()


    while True:

      env.step()

      #print(env.GetCar())
          

    return

Simulate()