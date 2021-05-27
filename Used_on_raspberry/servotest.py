import RPi.GPIO as GPIO
import time

servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 33) # GPIO 17 for PWM with 33Hz (pulse period of 30ms)
p.start(2) # Initialization

def set_servo(angle, servo):
    
    duty = angle/18+4
    print(duty)
    servo.ChangeDutyCycle(duty)

while True:
    set_servo(0, p)
    print("0")
    time.sleep(5)
    set_servo(10, p)
    print("45")
    time.sleep(5)
    set_servo(-10, p)
    print("-45")
    time.sleep(5)
    set_servo(0, p)
    time.sleep(5)
        
    
