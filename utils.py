import random
import math
import numpy as np 

def aug_matrix(w1, h1, w2, h2, 
               angle_range=(-45, 45), 
               scale_range=(0.5, 1.5), 
               trans_range=(-0.3, 0.3)):
    ''' 
    first Translation, then rotate, final scale.
        [sx, 0, 0]       [cos(theta), -sin(theta), 0]       [1, 0, dx]       [x]
        [0, sy, 0] (dot) [sin(theta),  cos(theta), 0] (dot) [0, 1, dy] (dot) [y]
        [0,  0, 1]       [         0,           0, 1]       [0, 0,  1]       [1]
    '''
    dx = (w2-w1)/2.0
    dy = (h2-h1)/2.0
    matrix_trans = np.array([[1.0, 0, dx],
                             [0, 1.0, dy],
                             [0, 0,   1.0]])

    angle = random.random()*(angle_range[1]-angle_range[0])+angle_range[0]
    scale = random.random()*(scale_range[1]-scale_range[0])+scale_range[0]
    scale *= np.min([float(w2)/w1, float(h2)/h1])
    alpha = scale * math.cos(angle/180.0*math.pi)
    beta = scale * math.sin(angle/180.0*math.pi)

    trans = random.random()*(trans_range[1]-trans_range[0])+trans_range[0]
    centerx = w2/2.0 + w2*trans
    centery = h2/2.0 + h2*trans
    H = np.array([[alpha, beta, (1-alpha)*centerx-beta*centery], 
                  [-beta, alpha, beta*centerx+(1-alpha)*centery],
                  [0,         0,                            1.0]])

    H = H.dot(matrix_trans)[0:2, :]
    return H 
