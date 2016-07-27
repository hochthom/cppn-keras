# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 19:36:30 2016

@author: hochthom
"""

import time
import numpy as np
from scipy.interpolate import interp1d
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import initializations
from PIL import Image



def my_init(shape, name=None):
    return initializations.normal(shape, scale=1.8, name=name)

def create_grid(x_dim=32, y_dim=32, scale=1.0):
    '''
    returns a vector of x and y coordinates and th ecorresponding radius from 
    the centre of the image.
    '''
    N = 0.5*(x_dim + y_dim)
    x1 = np.linspace(-1.0*x_dim/N*scale, 1.0*x_dim/N*scale, x_dim)
    y1 = np.linspace(-1.0*y_dim/N*scale, 1.0*y_dim/N*scale, y_dim)
    X, Y = np.meshgrid(x1, y1)
    x1 = np.ravel(X).reshape(-1, 1) 
    y1 = np.ravel(Y).reshape(-1, 1)
    r1 = np.sqrt(x1**2 + y1**2)
    return x1, y1, r1

def interpolate_z(z, n_frames=25, mode=None):
    '''
    Interpolate movement through latent space with spline approximation.
    '''
    x_max = float(z.shape[0])
    if mode is not None:
        x_max += 1
        if 'smooth' in mode:
            x_max += 2
        
    xx = np.arange(0, x_max)
    zt = []
    for k in range(z.shape[1]):
        yy = list(z[:,k])
        if mode is not None:
            yy.append(z[0,k])
            if 'smooth' in mode:
                yy = [z[-1,k]] + yy + [z[1,k]]
        fz = interp1d(xx, yy, kind='cubic')
        if 'smooth' in mode:
            x_new = np.linspace(1, x_max-2, num=n_frames, endpoint=False)
        else:
            x_new = np.linspace(0, x_max-1, num=n_frames, endpoint=False)
        zt.append(fz(x_new))
    
    return np.column_stack(zt)

def create_image(model, x, y, r, z):
    '''
    create an image for the given latent vector z 
    '''
    # create input vector
    Z = np.repeat(z, x.shape[0]).reshape((-1,x.shape[0]))
    X = np.concatenate([x, y, r, Z.T], axis=1)

    img = model.predict(X)
    img = (img - img.min()) / (img.max()-img.min()) * 255
    img = img.reshape(y_dim, x_dim)
    img = img.astype(np.uint8)
    return img

def create_image_seq(model, x, y, r, z, n_frames=25, mode=None):
    '''
    create a list of images with n_frames between the given latent vectors in z
    '''
    # create all z values
    zt = interpolate_z(z, n_frames, mode)
       
    images = []
    for k in range(zt.shape[0]):
        print 'Creating image %3i/%i ...' % (k+1, zt.shape[0])
        images.append(create_image(model,x,y,r,zt[k,:]))
    
    return images



SINGLE_IMAGE = True
    
n_z = 32
x_dim = 1280
y_dim = 720
x, y, r = create_grid(x_dim=x_dim, y_dim=y_dim, scale=10.0)
# create latent space
z = np.random.normal(0, 1, (5,n_z))

# create neural network
model = Sequential([
    Dense(32, init=my_init, input_dim=n_z+3),
    Activation('tanh'),
    Dense(32, init=my_init),
    Activation('tanh'),
    Dense(32, init=my_init),
    Activation('tanh'),
    Dense(1),
    Activation('sigmoid'),
])

model.compile(optimizer='rmsprop', loss='mse')


# create images
t_start = time.time()

if SINGLE_IMAGE:
    img = create_image(model, x, y, r, z[0,:])
    im = Image.fromarray(img)
    im.save('test.png')
else:
    img_seq = create_image_seq(model, x, y, r, z, n_frames=250, mode='smooth')
    for k, img in enumerate(img_seq):
        im = Image.fromarray(img)
        im.save('seq/img_%03i.png' % k)

    if True:        
        from moviepy.editor import *
        
        tmp = [np.dstack(3*[img]).astype("uint8") for img in img_seq]
        clip = ImageSequenceClip(tmp, fps=25)
        clip.write_videofile("video.avi", fps=25, codec='libx264')

t_stop = time.time()
print 'done in %.1f seconds.' % (t_stop-t_start)



