
import argparse
import numpy as np
import MathFunctions
import SimulationFunctions
import AuxFunctions
import NetworkFunctions

class TrainCar():

    def __init__(self):

        #Variables used only for simulation
        self.Env, self.Position, self.Orientation = SimulationFunctions.MakeEnv()
        self.ep =0;
        #State variables
        self.Speed = 4 #m/s
        self.Radious = parameters.Min_CurvatureRadious #meters
        self.AngleWheels = 0 #degrees
        self.Direction = 1;
        self.LidarData = SimulationFunctions.GetLidarSimulation(self.Env,self.Position,self.Orientation)

        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0
        self.Step = 0
        self.red = 0

    def run_frame(self):

        # Car moving
        NewPosition,NewOrientation, self.AngleWheels= SimulationFunctions.UpdateCar(abs(self.Radious), self.Direction, self.Speed, self.Position,self.Orientation)
    
        if ((self.Position[0] < 10 and NewPosition[0] > 10) and self.Position[1] > -6 and self.red != 1):
          self.reward += self.Speed*10/(self.Step +1)
          self.Step = 0
          self.red = 1

        if ((self.Position[0] > 10 and NewPosition[0] < 10) and self.Position[1] < -6 and self.red != 2):
          self.reward += self.Speed*10/(self.Step +1)
          self.Step = 0
          self.red = 2
          
        if ((self.Position[0] < 3.5 and NewPosition[0] > 3.5) and self.Position[1] > -5.5 and self.red != 3):
          self.reward += self.Speed* 10/(self.Step +1)
          self.Step = 0
          self.red = 3

        if ((self.Position[0] > 3.5 and NewPosition[0] < 3.5) and self.Position[1] < -5.5 and self.red != 4):
          self.reward += self.Speed* 10/(self.Step +1)
          self.Step = 0
          self.red = 4

        if ((self.Position[1] < -5.5 and NewPosition[1] > -5.5) and self.Position[0] < 6 and self.red != 5):
          self.reward += self.Speed* 10/(self.Step +1)
          self.Step = 0
          self.red = 5

        if ((self.Position[1] > -5.5 and NewPosition[1] < -5.5) and self.Position[0] > 6 and self.red != 6):
          self.reward += self.Speed*10/(self.Step +1)
          self.Step = 0
          self.red = 6

        if ((self.Position[1] < -3.5 and NewPosition[1] > -3.5) and self.Position[0] < 5 and self.red != 7):
          self.reward += self.Speed*10/(self.Step +1)
          self.Step = 0
          self.red = 7

        if ((self.Position[1] > -3.5 and NewPosition[1] < -3.5) and self.Position[0] > 5 and self.red != 8):
          self.reward += self.Speed*10/(self.Step +1)
          self.Step = 0
          self.red = 8

        if ((self.Position[0] < 6 and NewPosition[0] > 6) and self.Position[1] > -6 and self.red != 9):
          self.reward += self.Speed*10/(self.Step +1)
          self.Step = 0
          self.red = 9

        self.Orientation = NewOrientation
        self.Position = NewPosition

        #Updating LIDAR
        self.LidarData = SimulationFunctions.GetLidarSimulation(self.Env,self.Position,self.Orientation)
        
        if (self.ep % 100 == 0):
          i =0
          n=len(self.Env)
          while i<n:
              nn=len(self.Env[i])
              j=0
              while j<(nn-1):
                  plt.plot([self.Env[i][j][0],self.Env[i][j+1][0]],[self.Env[i][j][1],self.Env[i][j+1][1]],"c-.")
                  plt.scatter(self.Position[0], self.Position[1])
                  j+=1
              i+=1

        # Check car colision
        for i in range(len(self.LidarData)):
            if self.LidarData[i][1] <= parameters.Car_radius:
                self.done = True
                self.reward -= self.Speed * 100 
                if (self.ep % 100 == 0):
                  plt.show()
                break

        

    # ------------------------ AI control ------------------------

    def reset(self):

        self.Env, self.Position,self.Orientation  = SimulationFunctions.MakeEnv()
        self.Radious = parameters.Min_CurvatureRadious
        self.Orientation = 180
        self.Speed = 4
        self.Step = 0
        self.red = 0
        self.LidarData = SimulationFunctions.GetLidarSimulation(self.Env,self.Position,self.Orientation)
        
        lista = []
        for i in range(len(self.LidarData)):
          if (i < 90 or i >270):
            lista += [self.LidarData[i][1]]

        #return state
        return [lista + [self.Radious,self.Direction]]
    def GetCar(self):
      return [self.Position[0]],[self.Position[1]],[self.Orientation]

    def step(self, action, ep):
        self.ep = ep;
        self.reward = 0
        self.done = 0 
        self.Step += 1          

        if (action == 0):
          self.Radious = min(parameters.Max_CurvatureRadious, self.Radious + 1)

        if (action == 1):
          self.Radious = min(parameters.Max_CurvatureRadious, self.Radious + 5)

        if (action == 2):
          self.Radious = max(parameters.Min_CurvatureRadious, self.Radious - 1)

        if (action == 3):
          self.Radious = max(parameters.Min_CurvatureRadious, self.Radious - 5)
        
        if (action == 4):
          self.Direction *= -1
          self.reward -= 1

        self.run_frame()
        maxesqu = 0
        maxdir = 0
        maxfren = 0
        lista = []
        for i in range(len(self.LidarData)):
          if (i > 30 and i < 90):
            maxdir = max(maxdir,self.LidarData[i][1])
          if (i > 270 and i < 330):
            maxesqu = max(maxesqu,self.LidarData[i][1])
          if (i <30 or i > 330):
            maxfren = max(maxfren,self.LidarData[i][1])
          if (i < 90 or i >270):
            lista += [self.LidarData[i][1]]


        self.reward += self.Speed 
        if (maxfren > maxdir and maxfren > maxesqu):
          self.reward += self.Radious
        elif (maxesqu >= maxdir):
          self.reward += self.Direction/self.Radious
        elif (maxdir >= maxesqu):
          self.reward -= self.Direction/self.Radious 
        
        

        state = [lista + [self.Radious,self.Direction]]
        return self.reward, state, self.done


def MakeEnv2():
   
    circuitext = [[0,0] , [35,0],[35,-2],[0,-2],[0,0]]
    circuitint = []
    envir=[circuitext]
    position = [2,-1]
    orientation = 0
    return(envir, position, orientation)

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='DDQN',help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    #
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--n_threads', type=int, default=8, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',help="Compute Average reward per episode (slower)")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    #
    parser.set_defaults(render=False)
    parser.add_argument("-f", "--file", required=False) 
    return parser.parse_args(args)

def Simulate(times):

    Result = [[],[],[]]

    args = None
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)


    state_dim = (1,181)
    action_dim = 6
    algo = DDQN(action_dim, state_dim, args)
    algo.load_weights('/Model_Left')


    env = TrainCar()
    old_state, i = env.reset(), 0
    done = False

    while not done and i < times:
      a = algo.action(old_state)
      print(a)
      r,old_state, done = env.step(a,3)

      Position = [[],[]]
      Position[0],Position[1],Orientation = env.GetCar()

      Result[0] += Position[0]
      Result[1] += Position[1]
      Result[2] += Orientation

      i += 1
          

    return Result