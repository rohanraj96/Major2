import sys
sys.path.append('mpc')
sys.path.append('cv')
sys.path.append('pwm')
sys.path.append('car')
from car import *
from mpc import *
from cv import *
import time



import cvxpy
import numpy as np
from cvxpy import *
#import matplotlib.pyplot as plt
from math import *
import time

import argparse


import os
import logging

target = [-3.512,5.0]

if __name__ == '__main__':

        parser = argparse.ArgumentParser(description='Rc_Car')
         #General
        parser.add_argument('-max_s','--max_speed',default=5.0,help='desirable max_speed of the car')
        parser.add_argument('-min_s','--min_speed',default=-5.0,help='desirable min_speed of the car')
        parser.add_argument('-dt','--discrete_time',default=0.1,help='discrete time interval of the car')
        parser.add_argument('-lr','--length_car',default=1.0,help='length of the car from steering column to the back')
        parser.add_argument('-rf','--reference_velocity',default=1,help='reference velocity of the car')
        parser.add_argument('-hr','--horizons',default=20,help='number of time stamps')

        #We will further add arguments for cv as well

        args = parser.parse_args()

        rc_car = car()
        mpc = MPC(float(args.max_speed),float(args.min_speed),float(args.discrete_time),float(args.length_car),float(args.reference_velocity),args.horizons)
        rc_cv=ComputerVision()
        # rc_pwm = pwm()

        try:

                # time1=time.time()
            coeffs ,T= rc_cv.run('circle1.jpg')
            
            cte = coeffs[2]

            epsi = atan(-coeffs[1]);
            print("slope initially",epsi)

            #state_vector

            # state = np.matrix([0.0,0.0,0.0,rc_car.v,cte,epsi]).T
            state =  np.matrix([0.0,0.0,0.0,rc_car.v,cte,epsi]).T
            x =  state

            #actuator_input
            u = np.matrix([0.0,0.0]).T
            for i in range(1000):

                A, B, C = mpc.LinealizeCarModel(x,u,coeffs)
                # print(A)
                ustar,xstar,cost = mpc.CalcInput(A,B,C,x,u,coeffs)

                
                u[0,0] = mpc.GetListFromMatrix(ustar.value[0, :])[0]
                u[1, 0] = -float(ustar[1, 0].value)

                x = A*x + B*u 
                print("printing x value",x)

                plt.subplot(3, 1, 1)
    # plt.plot(target[0], target[1], 'xb')
                plt.plot(x[0], x[1], '.r')
                plt.plot(mpc.GetListFromMatrix(xstar.value[0, :]), mpc.GetListFromMatrix(xstar.value[1, :]), '-b')
                plt.axis("equal")
                plt.xlabel("x[m]")
                plt.ylabel("y[m]")
                plt.grid(True)

                plt.subplot(3, 1, 2)
                plt.cla()
                plt.plot(mpc.GetListFromMatrix(xstar.value[2, :]), '-b')
                plt.plot(mpc.GetListFromMatrix(xstar.value[3, :]), '-r')
                plt.ylim([-1.0, 1.0])
                plt.ylabel("velocity[m/s]")
                plt.xlabel("horizon")
                plt.grid(True)

                plt.subplot(3, 1, 3)
                plt.cla()
                plt.plot(mpc.GetListFromMatrix(ustar.value[0, :]), '-r', label="a")
                plt.plot(mpc.GetListFromMatrix(ustar.value[1, :]), '-b', label="b")
                plt.ylim([0, 1])
                plt.legend()
                plt.grid(True)


                dis = np.linalg.norm([x[0] - target[0], x[1] - target[1]])
                if (dis < 0.1):
                    print("Goal")
                    break

                # if KeyboardInterrupt:
                #     break


            # plt.show()



                    # print(time.time() - time1)
        except KeyboardInterrupt:
            plt.show()
            print('finished')






                        



