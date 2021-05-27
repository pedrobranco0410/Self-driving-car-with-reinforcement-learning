import PyLidar3
import time # Time module
import RPi.GPIO as GPIO
import time
#import pigpio

servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

servo = GPIO.PWM(servoPIN, 33) # GPIO 17 for PWM with 33Hz (pulse period of 30ms)
servo.start(2) # Initialization

#ESC = 4
#pi = pigpio.pi()


port = "/dev/ttyUSB0" #windows
Obj = PyLidar3.YdLidarX4(port, 1100)    
Obj.Connect()
gen = Obj.StartScanning()

def getLidar():
    
    data = []
    
    
    lista = next(gen)
    #Obj.StopScanning()

    
    for i in range(360):
        data += [[i,lista[i]*0.001]]
    
    print(str(data[270])+ " " + str(data[0]) + "" + str(data[90]))
    
    
    return data

def turnServo(i_R, angle):
    
    servo_angle = (i_R - 0.17528)/0.0737157
    duty = angle/18+4
    servo.ChangeDutyCycle(duty)
    print(angle)
    return


# def set_speed(vitesse, sens = 1):
#     pulse = 1500+sens*vitesse*5 #pulse between 1000 and 2000, 1500 is neutral
#     pi.set_servo_pulsewidth(ESC, pulse)
    
#set_speed(10)