#Scipt d'initialisation de l'ESC

import time
import pigpio

#Initialize the ESC
def initialize_ESC(pinESC):
    pi = pigpio.pi() # Connect to local Pi.

    pi.set_servo_pulsewidth(pinESC, 0) # Minimum throttle.

    time.sleep(1)

    pi.set_servo_pulsewidth(pinESC, 2000) # Maximum throttle.

    time.sleep(1)

    pi.set_servo_pulsewidth(pinESC, 1500) # Neutral input (over 1500->forward | under 1500 -> backward)

    time.sleep(5)

    pi.set_servo_pulsewidth(pinESC, 0) # Stop servo pulses.



ESC = 4
initialize_ESC(ESC)