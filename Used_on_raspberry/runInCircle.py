"""This code allows the car to run in circle for a certain period of time to measure the lateral acceleration and deduce the radius
of the circle that the car makes
We used this code to compute the relation between the command send to the servo and the turn radius of the car"""

import time
import pigpio
import RPi.GPIO as GPIO
from Phidget22.Phidget import *
from Phidget22.Devices.Accelerometer import *
from Phidget22.Devices.Gyroscope import *

#Do not forget to initialize the ESC with initialize_ESC.py before running this code
ESC = 4
pi = pigpio.pi() # Connect to local Pi.

#set the speed between 0 and 100%
#forward->sens = 1 | backward->sens = -1
def set_speed(vitesse, sens = 1):
    pulse = 1500+sens*vitesse*5 #pulse between 1000 and 2000, 1500 is neutral
    pi.set_servo_pulsewidth(ESC, pulse)


##Initialise servo pin
servoPIN = 19
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 33) # GPIO 17 for PWM with 33Hz, pulse period of 30ms
p.start(2) # Initialization
#servo initalised

#Initialise IMU
#-------INITIALISATION---------------
# Create the two needed channels

def onAccelerationChange(self, acceleration, timestamp):
	#print("Acceleration: \t"+ str(acceleration[0])+ "  |  "+ str(acceleration[1])+ "  |  "+ str(acceleration[2]))
	#print("Timestamp: " + str(timestamp))
	#print("----------")
    pass

def onAngularRateUpdate(self, angularRate, timestamp):
	#print("MagneticField: \t"+ str(magneticField[0])+ "  |  "+ str(magneticField[1])+ "  |  "+ str(magneticField[2]))
	#print("Timestamp: " + str(timestamp))
	#print("----------")
    pass

accelerometer0 = Accelerometer()
gyroscope0 = Gyroscope()

accelerometer0.setOnAccelerationChangeHandler(onAccelerationChange)
gyroscope0.setOnAngularRateUpdateHandler(onAngularRateUpdate)

# The openWaitForAttachment() function will 
# hold the program until a Phidget channel 
# matching the one you specified is attached, 
# or the function times out.

accelerometer0.openWaitForAttachment(5000)
gyroscope0.openWaitForAttachment(5000)

acceleration_tab = []
gyro_tab = []
temps = []


def set_servo(angle, servo):
    
    duty = angle/18+4
    print(duty)
    servo.ChangeDutyCycle(duty)

def runInCircle(vitesse, angle, duree):
    set_servo(angle, p)
    time.sleep(1)
    set_speed(vitesse)
    start = time.time()
    while(time.time()-start<duree):
        acceleration_tab.append(accelerometer0.getAcceleration())
        gyro_tab.append(gyroscope0.getAngularRate())
        temps.append(time.time()-start)
        time.sleep(0.01)
        
    #set_speed(-15)
    time.sleep(1)
    set_speed(0)
    set_servo(0, p)
    time.sleep(1)
    
angle_servo = 14
vitesse =20 #en pourcentage

runInCircle(vitesse, angle_servo, 10)
    
    
pi.set_servo_pulsewidth(ESC, 0)

pi.stop() # Disconnect from local Raspberry Pi.
p.stop()
GPIO.cleanup()
accelerometer0.close()
gyroscope0.close()

#Save data in a txt file
N = len(acceleration_tab)
with open(f'Donnees_cercle/donnees_cercle_{time.asctime()}_angle={angle_servo}_vitesse={vitesse}.txt', 'w') as f :
    for i in range(N):
        t   = str(temps[i])
        acc = str(acceleration_tab[i][0]) + " , " + str(acceleration_tab[i][1]) + " , " + str(acceleration_tab[i][2])
        gyro = str(gyro_tab[i][0]) + " , " + str(gyro_tab[i][1]) + " , " + str(gyro_tab[i][2])
        chaine =  t + ' , ' + acc + ' , ' + gyro + ' \n'
        f.write(chaine)