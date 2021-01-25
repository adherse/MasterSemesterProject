import numpy as np

def euclidean_distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 +(y1-y2)**2)

def detect_failure(x, data, thresholds):
    valid_data = []
    
    deltaGPS = euclidean_distance(x[0],x[1],data[0],data[1])
    valid_data.append(1.0 if deltaGPS<thresholds[0] else 0.0)

    deltaCourse = min(np.abs(x[2] - data[2]), np.abs(np.mod(x[2], 2*np.pi) - np.mod(data[2], 2*np.pi)))
    valid_data.append(1.0 if deltaCourse<thresholds[1] else 0.0)
    
    deltaVelocity = np.abs(x[3] - data[3])
    deltaAcceleration = np.abs(x[4] - data[4])
    deltaYawrate = np.abs(x[5] - data[5])
    
    valid_data.append(1.0 if deltaVelocity<thresholds[2] else 0.0)
    valid_data.append(1.0 if deltaAcceleration<thresholds[3] else 0.0)
    valid_data.append(1.0 if deltaYawrate<thresholds[4] else 0.0)
    
    return valid_data

def kalman_update(x, data, P, Q, R, I, dt, thresholds, init):
    
    #x0 is x, x1 is y, x2 is heading, x3 is velocity, x4 is acceleration, x5 is yaw rate
    
    ux = (1/x[5]**2) * \
 ((x[3]*x[5] + x[4]*x[5]*dt)* \
  np.sin(x[2] + x[5]*dt) + x[4]*np.cos(x[2] + x[5]*dt) - x[3]*x[5]*np.sin(x[2]) - np.cos(x[2])*x[4])
    uy = (1/x[5]**2) * \
 ((-x[3]*x[5] - x[4]*x[5]*dt)* np.cos(x[2] + x[5]*dt) \
 + x[4]*np.sin(x[2] + x[5]*dt) + x[3]*x[5]*np.cos(x[2]) - x[4]*np.sin(x[2]))


    x[0] = x[0] + ux
    x[1] = x[1] + uy
    x[2] = x[2] + dt*x[5]
    x[3] = x[3] + dt*x[4]
    x[4] = x[4]
    x[5] = x[5]
    
    a13 = float((1/x[5]**2) * (x[4]*np.sin(x[2])- x[4]*np.sin(dt*x[5]+x[2]) - x[3]*x[5]*np.cos(x[2]) + (dt*x[4]*x[5] + x[3]*x[5])*np.cos(dt*x[5]+x[2]))) 
    a23 = float((1/x[5]**2) * (-x[4]*np.cos(x[2])+ x[4]*np.cos(dt*x[5]+x[2]) - x[3]*x[5]*np.sin(x[2]) - (-dt*x[4]*x[5] - x[3]*x[5])*np.sin(dt*x[5]+x[2]))) 
    
    a14 = float((1/x[5]**2) * (-x[5]*np.sin(x[2]) + x[5]*np.sin(dt*x[5]+x[2])))
    a24 = float((1/x[5]**2) * (x[5]*np.cos(x[2]) - x[5]*np.cos(dt*x[5]+x[2])))
    
    a15 = float((1/x[5]**2) * (dt*x[5]*np.sin(dt*x[5]+x[2])-np.cos(x[2])+np.cos(dt*x[5] + x[2])))
    a25 = float((1/x[5]**2) * (-dt*x[5]*np.cos(dt*x[5]+x[2])-np.sin(x[2])+np.sin(dt*x[5] + x[2])))
    
    a161 = float((1/x[5]**2) * (-dt*x[4]*np.sin(dt*x[5]+x[2])+dt*(dt*x[4]*x[5]+x[3]*x[5])*np.cos(dt*x[5]+x[2])-x[3]*np.sin(x[2])+(dt*x[4]+x[3])*np.sin(dt*x[5]+x[2])))
    a162 = float((1/x[5]**3) * (-x[4]*np.cos(x[2])+x[4]*np.cos(dt*x[5]+x[2])-x[3]*x[5]*np.sin(x[2])+(dt*x[4]*x[5]+x[3]*x[5])*np.sin(dt*x[5]+x[2])))
    a16 = a161-2*a162
    
    a261 = float((1/x[5]**2) * (dt*x[4]*np.cos(dt*x[5]+x[2])-dt*(-dt*x[4]*x[5]-[3]*x[5])*np.sin(dt*x[5]+x[2])+x[3]*np.cos(x[2])+(-dt*x[4]-x[3])*np.cos(dt*x[5]+x[2])))
    a262 = float((1/x[5]**3) * (-x[4]*np.sin(x[2])+x[4]*np.sin(dt*x[5]+x[2])+x[3]*x[5]*np.cos(x[2])+(-dt*x[4]*x[5]-x[3]*x[5])*np.cos(dt*x[5]+x[2])))
    a26 = a161-2*a162


    Jf = np.matrix([[1.0, 0.0, a13, a14, a15, a16],
                    [0.0, 1.0, a23, a24, a25, a26],
                    [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                    [0.0, 0.0, 0.0, 1.0, dt, 0.0], 
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
    
    # Project the error covariance ahead
    P = Jf*P*Jf.T + Q
    
    # Measurement Update (Correction)
    # ===============================
    # Measurement Function
    hx = np.matrix([[float(x[0])],
                    [float(x[1])],
                    [float(x[2])],
                    [float(x[3])],
                    [float(x[4])],
                    [float(x[5])]])
    
    
    if GPS[filterstep]:
        if init:
            Jh = np.diag([1.0,1.0,1.0,1.0,1.0,1.0])
        else:
            valid_data = detect_failure(x, data, thresholds)
            Jh = np.diag([valid_data[0],valid_data[0],valid_data[1],valid_data[2],valid_data[3],valid_data[4]])
    else:
        if init:
            Jh = np.diag([0.0,0.0,1.0,1.0,1.0,1.0])
        else:
            valid_data = detect_failure(x, data, thresholds)
            Jh = np.diag([0,0,valid_data[1],valid_data[2],valid_data[3],valid_data[4]])

          
    K = (P*Jh.T) * np.linalg.inv(Jh*P*Jh.T + R)

    # Update the estimate via
    Z = data
    y = Z - (hx)
    x = x + (K*y)

    # Update the error covariance
    P = (I - (K*Jh))*P
    
    return (x, P)

