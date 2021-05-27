import time
import pigpio
import RPi.GPIO as GPIO

ESC = 4

pi = pigpio.pi() # Connect to local Pi.

# ----------- Initialisation -----------
pi.set_servo_pulsewidth(ESC, 0) # Minimum throttle.

time.sleep(1)

pi.set_servo_pulsewidth(ESC, 2000) # Maximum throttle.

time.sleep(1)

pi.set_servo_pulsewidth(ESC, 1500) # Neutral input (over 1500->forward | under 1500 -> backward)

time.sleep(5)

pi.set_servo_pulsewidth(ESC, 0) # Stop servo pulses.

# ----------- Initialisation terminée -----------

pi.set_servo_pulsewidth(ESC, 1575)

time.sleep(5)

pi.set_servo_pulsewidth(ESC, 0)

pi.stop() # Disconnect from local Raspberry Pi.


# ----------- Test de la commande du servomoteur -----------
servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)


p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz
p.start(2) # Initialization

def set_servo(angle, servo):
    
    duty = angle/18+2
    print(duty)
    servo.ChangeDutyCycle(duty)
 
while True:                                                                       
    set_servo(0, p)
    time.sleep(1.5)
    set_servo(90, p)
    time.sleep(0.5)
    set_servo(180, p)
    time.sleep(0.5)
    set_servo(0, p)
    time.sleep(0.5)