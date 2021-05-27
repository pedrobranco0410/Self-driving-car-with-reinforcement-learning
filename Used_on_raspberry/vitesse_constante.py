#to run the car at constant speed for a certain amount of time

import time
import pigpio

ESC = 4
pi = pigpio.pi()

def set_speed(vitesse, sens = 1):
    pulse = 1500+sens*vitesse*5 #pulse between 1000 and 2000, 1500 is neutral

    pi.set_servo_pulsewidth(ESC, 1901)
    
vitesse = -20
duree = 5

set_speed(vitesse)

time.sleep(duree)

pi.set_servo_pulsewidth(ESC, 0)
pi.stop()
