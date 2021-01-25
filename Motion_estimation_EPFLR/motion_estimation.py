import socket
import sys
import pickle
import time
import tcplib
import EKF


IP_PORT_IN_IMU =  10030
IP_PORT_IN_SPEED = 10031
IP_PORT_OUT_SLAM =  10032
IP_PORT_OUT_CONTROL = 10033
IP_ADDR = 'localhost'
MY_MODULE_NAME = 'motion_estimation'

print('INFO:', MY_MODULE_NAME, 'starting.')


# Init :
numstates = 6
dt = 1.0/200.0

varGPS = 6.0 
varspeed = 1.0 
varacc = 1.0
varyaw = 0.1 # Variances on sensor data I had. To be changed for the car when we have the sensors
R = np.diag([varGPS**2, varGPS**2, varyaw**2, varspeed**2,varacc**2, varspeed**2])
I = np.eye(numstates)

P = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
sPos     = 0.5*8.8*dt**2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sCourse  = 0.1*dt # assume 0.1rad/s as maximum turn rate for the vehicle
sVelocity= 8.8*dt # assume 8.8m/s2 as maximum acceleration, forcing the vehicle. Again, to be changed
Q = np.diag([sPos**2, sPos**2, sCourse**2, sVelocity**2, sCourse**2, sVelocity**2])

thresholdGPS = 10
thresholdCourse = 0.5
thresholdVelocity = 5
thresholdAcceleration = 1
thresholdYawrate = 0.1
thresholds = [thresholdGPS, thresholdCourse, thresholdVelocity, thresholdAcceleration, thresholdYawrate]



# Create a TCP/IP socket for the input
socket_imu = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address_in_imu = (IP_ADDR, IP_PORT_IN_IMU)
print('INFO: \'', MY_MODULE_NAME, '\' connecting to {} port {} for input.'.format(*server_address_in_imu))
socket_imu.connect(server_address_in_imu)

socket_speed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address_in_speed = (IP_ADDR, IP_PORT_IN_SPEED)
print('INFO: \'', MY_MODULE_NAME, '\' connecting to {} port {} for input.'.format(*server_address_in_speed))
socket_speed.connect(server_address_in_speed)


# Create a TCP/IP socket for the output
socket_slam = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address_out_slam = (IP_ADDR, IP_PORT_OUT_SLAM)
print('INFO: \'', MY_MODULE_NAME, '\' connecting to {} port {} for output.'.format(*server_address_out_slam))
socket_slam.connect(server_address_out_slam)

socket_control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address_out_control = (IP_ADDR, IP_PORT_OUT_CONTROL)
print('INFO: \'', MY_MODULE_NAME, '\' connecting to {} port {} for output.'.format(*server_address_out_control))
socket_control.connect(server_address_out_control)


module_running = True
iterations = 0

try:
    while module_running:
        
	iterations += 1
        speed = tcplib.receiveData(socket_speed)
        imu_frame = tcplib.receiveData(socket_imu)
        
        measurements = np.matrix([[imu_frame[0], imu_frame[1], imu_frame[4], speed, imu_frame[3], imu_frame[5]]]).T
        if (iterations == 1):
            x = measurements
        
        
        x, P = kalman_update(x, measurements, P, Q, R, I, dt, thresholds, iterations < 50) 
        
        to_send = x.A1
        
        tcplib.sendData(socket_control, to_send)
        tcplib.sendData(socket_slam, to_send)


        if false:
            module_running = False

finally:
    sock_input.close()
    sock_output.close()
    

