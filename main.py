print("a")
import tensorflow as tf
import numpy as np
import PIL

from PIL import Image

#import pillow-PIL
#from Pillow import PIL

#import PIL.image #use this as a wrapper here
import matplotlib


from io import BytesIO
from IPython.display import clear_output, Image, display

def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  clear_output(wait = True)
  display(Image(data=f.getvalue()))
  #here need to show image instead of ...<IPython.core.display.Image object>  - meant for jupityer notebook
  #f.show()
  #Image.show(data=f.getvalue())
  #img.show()
  #matplotlib.pyplot.imshow(f)

print("c")

#sess = tf.InteractiveSession()
sess = tf.compat.v1.InteractiveSession()


def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

print("d")

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

print("e")

def laplace(x):
  """Compute the 2D laplacian of an array"""
  laplace_k = make_kernel([[0.25, 0.5, 0.25],
                           [0.5, -3., 0.5],
                           [0.25, 0.5, 0.25]])
  return simple_conv(x, laplace_k)

print("f")


N = 500

# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)

# Some rain drops hit a pond at random points
for n in range(40):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()

DisplayArray(u_init, rng=[-0.1, 0.1])


#Define TensorFlow input, variables and finally two operations. Then group them and run session. All details below.
# Parameters:
# eps -- time resolution
# damping -- wave damping
# c -- wave speed # Not a part of original code version
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())
c = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# Discretized PDE update rules
U_ = U + eps * Ut
Ut_ = Ut + eps * ((c ** 2) * laplace(U) - damping * Ut)

# Operation to update the state
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))

# Initialize state to initial conditions
tf.global_variables_initializer().run()

#Finally run 1000 steps and display pond after every step.
# Run 1000 steps of PDE
for i in range(1000):
  # Step simulation
  step.run({eps: 0.03, damping: 0.04, c: 3.0})
  DisplayArray(U.eval(), rng=[-0.1, 0.1])

sess.close()