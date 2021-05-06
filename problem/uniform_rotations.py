import numpy as np

def generate_uniform_rotation(max_ang = np.pi):
  """Generate a Uniform Rotation matrix with max rotation
  angle max_ang. Matrices are generated using the subgroup
  algorithm. We assume the rotation is small, i.e. 
  max_ang is in (0,pi].
  float: max_ang, in (0,pi]
  return: rotation matrix, (3,3) numpy array
  """
  d = 3 # dimension
  # for nesting
  A = np.zeros((d,d))
  A[0,0] = 1.0
  # e1
  e1 = np.zeros(d)
  e1[0] = 1.0
  # comute a 2d rotation
  ang = np.random.uniform(-max_ang,max_ang)
  Q2 = np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]])
  # nest it in the 3d matrix
  A[1:,1:] = Q2.copy()
  # generate a point uniformly within angle max_ang of e_1
  while True:
    # generate v on the unit sphere
    v = np.random.randn(d)
    v =v/np.linalg.norm(v)
    # check angle
    if np.arccos(v @ e1) <= max_ang:
      # success: exit the while loop
      break
  # compute y (i.e. x from 3.1b)
  y = (e1 - v)/np.linalg.norm(e1-v)
  # compute household reflection
  Q = (np.eye(d) - 2*np.outer(y,y)) @ A
  return Q

if __name__ == "__main__":
  #num points
  n = 1000
  d = 3 # dim
  # amount to rotation 
  max_ang = np.pi/2 # choose 0<= max_ang <= pi
  # point to rotate
  x = np.array([1,0,0])
  R = np.zeros((n,d))
  for i in range(n):
    Q = generate_uniform_rotation(max_ang)
    # rotate the point
    R[i] = (Q@x).copy()
  
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(R[:,0],R[:,1],R[:,2])
  ax.scatter(x[0],x[1],x[2],'k')
  ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1);
  plt.show()
  
