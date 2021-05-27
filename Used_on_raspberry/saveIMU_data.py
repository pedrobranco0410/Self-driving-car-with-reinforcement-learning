from Phidget22.Phidget import *
from Phidget22.Devices.Accelerometer import *
from Phidget22.Devices.Magnetometer import *
import time

#-------INITIALISATION---------------
# Create the two needed channels



def onAccelerationChange(self, acceleration, timestamp):
	#print("Acceleration: \t"+ str(acceleration[0])+ "  |  "+ str(acceleration[1])+ "  |  "+ str(acceleration[2]))
	#print("Timestamp: " + str(timestamp))
	#print("----------")
    pass

def onMagneticFieldChange(self, magneticField, timestamp):
	#print("MagneticField: \t"+ str(magneticField[0])+ "  |  "+ str(magneticField[1])+ "  |  "+ str(magneticField[2]))
	#print("Timestamp: " + str(timestamp))
	#print("----------")
    pass

accelerometer0 = Accelerometer()
magnetometer0 = Magnetometer()

accelerometer0.setOnAccelerationChangeHandler(onAccelerationChange)
magnetometer0.setOnMagneticFieldChangeHandler(onMagneticFieldChange)

# The openWaitForAttachment() function will 
# hold the program until a Phidget channel 
# matching the one you specified is attached, 
# or the function times out.

accelerometer0.openWaitForAttachment(5000)
magnetometer0.openWaitForAttachment(5000)

acceleration_tab = []
magnet_tab = []
temps = []
nb_measures = 10
#-----------------------------------------------

for i in range(nb_measures) :
    acceleration_tab.append(accelerometer0.getAcceleration())
    magnet_tab.append(magnetometer0.getMagneticField())
    temps.append(time.time())
    time.sleep(0.01)


#---------------END-----------


accelerometer0.close()
magnetometer0.close()
N = len(acceleration_tab)


with open('donnee.txt', 'w') as f :
    for i in range(N):
        t   = str(temps[i])
        acc = str(acceleration_tab[i])
        mag = str(magnet_tab[i])
        chaine =  t + ' , ' + acc + ' , ' + mag + ' \n'
        f.write(chaine)
