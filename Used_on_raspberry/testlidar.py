import PyLidar3
import time # Time module
from numpy import mean

port = "/dev/ttyUSB1" #windows


Obj = PyLidar3.YdLidarX4(port)
def getLidar():
    
    gen = []
    if (Obj.Connect()):
        print(Obj.GetDeviceInfo())
        gen = Obj.StartScanning()
        t = time.time() # start time 
        lista = next(gen)
        Obj.StopScanning()
        Obj.Disconnect()
        
        data = []
        for i in range(360):
            #data += [[i,lista[i]*0.001]]
            data += [lista[i]]
        #return lista
        return data
    else:
        print("Error connecting to device")
    
lista = getLidar()
print(lista)
for i in range(int(len(lista)/ 5)):
    print(int(5 * i/3.60), mean(lista[5 *i : 5 * i + 4]))