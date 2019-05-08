import cvxpy
import numpy as np
from cvxpy import *
#import matplotlib.pyplot as plt
from math import *
import time

class MPC:

        dt = 0
        max_s =0 
        min_s =0
        lr =0
        ref_vel= 3
        horizons = 0
        target = []

        def __init__(self,max_s,min_s,dt,lr,ref_vel,horizons):
                self.dt = dt
                self.lr =lr
                self.max_s = max_s
                self.min_s= min_s
                self.ref_vel= ref_vel
                self.horizons = horizons
                self.target = [-3.512,5.0]

        def LinealizeCarModel(self,xb,u,coeffs):
                    """
                    TODO complete model
                    """
                    # print(coeffs)
                    x = xb[0]
                    y = xb[1]
                    v = xb[2]
                    theta = xb[3]

                    cte = xb[4]
                    epsi = xb[5]



                    a = u[0]
                    beta = u[1]

                    f0 = (coeffs[0]*x*x + coeffs[1] * x + coeffs[2])
                    psides0 = atan((2*coeffs[0]*x + coeffs[1]))


                    t1 = -self.dt * v * sin(theta + beta)
                    t2 = self.dt * v * cos(theta + beta)

                    if beta==0:
                        t3 = epsi/0.01
                    else:
                        t3 = epsi/beta
                    

                    A = np.eye(xb.shape[0])
                    A[0, 2] = self.dt * cos(theta) 
                    A[1, 2] = self.dt * sin(theta)
                    A[3, 2] = self.dt * beta / self.lr
                    A[0, 3] = t1
                    A[1, 3] = t2

                    # A[4, 4] = 0
                    A[4, 2] = sin(epsi) * self.dt
                    A[5, 5] = 0


                    B = np.zeros((xb.shape[0], u.shape[0]))
                    B[2, 0] = self.dt
                    B[0, 1] = t1
                    B[1, 1] = t2
                    # B[3, 1] = self.dt * v*sin(beta)/ self.lr
                    B[4 ,1] = v * t3*cos(epsi) * self.dt
                    B[5, 1] = v/self.lr *self.dt

                    tm = np.zeros((6, 1))
                    tm[0, 0] = v * cos(theta) * self.dt 
                    tm[1, 0] = v * sin(theta) * self.dt
                    tm[2, 0] = a * self.dt
                    tm[3, 0] = v / self.lr * beta * self.dt
                    tm[4, 0] = (f0 - y) + v * sin(epsi) * self.dt
                    tm[5, 0] = (theta - psides0) + v/self.lr* beta*self.dt
                    C = xb + tm
                    C = C - A * xb - B * u

                    return A, B, C


        def NonlinearModel(self,x,u,coeffs):

                # print(x[0].value)
                # print(coeffs)
                # x_0 = x
                # f0 = coeffs[0]*x[0]*x[0] + coeffs[1] * x[0] + coeffs[2]
                # print(f0)
                # psides0 = atan(float(2*coeffs[0]*x[0] + coeffs[1]))
                # print(x[0].value)
                # print(x[1])
                # print(x[2])
                # print(x[3])
                x[0] = x[0] + x[2] * cos(x[3] + u[1]) * self.dt
                x[1] = x[1] + x[2] * sin(x[3] + u[1]) * self.dt
                x[2] = x[2] + u[0] * self.dt
                x[3] = x[3] + x[2] / self.lr * sin(u[1]) * self.dt

        

                # x[4] = ((f0 - x_0[1]) +  x_0[2] * sin(x_0[5]) * self.dt)
                # x[5] = (x_0[3]-psides0) + x_0[2]/self.lr * u[1]*self.dt

                return x

        def CalcInput(self,A,B,C,x, u,coeffs):
                

                states=[]
                x_0=x[:]
                # x_1 = x_0
                # u_1 = u
                x = Variable(x.shape[0],self.horizons+1)
                u = Variable(u.shape[0],self.horizons)

                for t in range(self.horizons):

                        # A, B, C = mpc.LinealizeCarModel(x_1,u_1,coeffs)
                        constr = [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]
                        # x_0 = A x_0
                        # constr = [x[:,t+1] == self.NonlinearModel(x[:,t],u[:,t],coeffs)]
                        constr += [x[2, t + 1] <= 5]
                        constr += [x[2, t + 1] >= -5]
                        # constr += [u[1 ,t ] <=0.43]
                        # constr += [u[1 ,t ] >=-0.43]
                        # constr += [u[0 ,t ] <=1]
                        # constr += [u[0 ,t ] >=0]
                        constr += [abs(u[:, t]) <= 0.5]

                        # constr += [u[0 ,t ] <=1]
                        # constr += [u[0 ,t ] >=-0.5]
                        # # constr += [u[1 ,t ] >=-0.436332]
                        # constr += [u[1 ,t ] >=-0.43]
                        # constr += [u[1 ,t ] <= 0.43]
                        

                        # cost = sum_squares(abs(x[0, t] - self.target[0])) * 10.0 * t
                        # cost += sum_squares(abs(x[1, t] - self.target[1])) * 10.0 * t
                        # if t == self.horizons - 1:
                        #         cost += (x[0, t + 1] - self.target[0]) ** 2 * 10000.0
                        #         cost += (x[1, t + 1] - self.target[1]) ** 2 * 10000.0
                        cost = 10*x[4,t] **2 *t
                        cost += 10*x[5,t]**2 * t
                        if t == self.horizons - 1:
                                cost += x[4,t] ** 2 * 10000.0
                                cost += x[5,t] ** 2 * 10000.0
                        cost += (x[2,t]-self.ref_vel)**2

                        states.append(Problem(Minimize(cost), constr))

                prob = sum(states)
                prob.constraints += [x[:, 0] == x_0]
                result = prob.solve()
                print(prob.value)
                print(u.value)
                # print(x.value)
                return u,x,prob.value

        def GetListFromMatrix(self,x):
                return np.array(x).flatten().tolist()

