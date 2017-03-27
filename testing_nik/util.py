# These functions definitions should be updated from their corresponding notebook files
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from exceptions import ValueError
try:
    from IPython import display
except:
    1
eps = np.finfo(np.float32).eps
MEAN = 0.171676928961 


def check_nan(x):
    return np.isnan(np.sum(x))

def number_of_nan(x):
    return np.count_nonzero(np.isnan(x))



def get_histogram(im_np,hist_size):
    counts, _ = np.histogram(im_np, bins=hist_size, density=False , range=[0,1])
    probs = (counts + 0.0)/im_np.size + eps
    return probs

def clean_array(x,min=None,max=None):
    x = np.nan_to_num(x)
    if min > None:
        x[x<min] = min
    if max > None:
        x[x>max] = max
    return x

def clean_image(x,min=0,max=1):
    x = np.nan_to_num(x)
    x[x<min] = min
    x[x>max] = max
    return x


def create_positive_training_set(image_np,num_samples = 10, delta_X = 5,orig_size = 562, target_size = 170,offset =196,
                                 doTransform = False, transform = None , hist_size = 100 ):
    # Currently For each image 
    # Written in create_similar_slightly_jittered_dataset.ipynb
    image_np = clean_array(image_np,min = 0,max = 1)
    img = Image.fromarray(np.resize(image_np,(orig_size,orig_size)))
    if doTransform:
        if transform == 'histogram':
            jittered_dataset = np.zeros((num_samples,hist_size))
    else:
        jittered_dataset = np.zeros((num_samples,target_size*target_size))
    for counter in range(num_samples):
        theta = np.random.randint(0,360)
        x = np.random.randint(-delta_X,delta_X+1)
        y = np.random.randint(-delta_X,delta_X+1)
        jittered_img = img.rotate(theta).crop((offset + x ,             offset + y ,
                                              offset + x + target_size, offset + y + target_size  ))
        jittered_img_np = np.asarray(jittered_img.getdata())
        if doTransform:
            if transform == 'histogram':
                jittered_img_np = get_histogram(jittered_img_np,hist_size = hist_size)
            # TO DO = blur the jittered_img/ jitter_img_np to look more like local_map.
        jittered_dataset[counter,:] = jittered_img_np
    return (None, jittered_dataset)

def create_negative_training_set(image_np,num_samples = 10, delta_X = 5,orig_size = 562, target_size = 256,offset = 153,doTransform = False,transform = None , hist_size = 100):
    # Currently For each image 
    # Written in create_similar_slightly_jittered_dataset.ipynb
    img = Image.fromarray(np.resize(image_np,(orig_size,orig_size)))
    if doTransform:
        if transform == 'histogram':
            jittered_dataset = np.zeros((num_samples,hist_size))
    else:
        jittered_dataset = np.zeros((num_samples,target_size*target_size))
    for counter in range(num_samples):
        theta = np.random.randint(0,360)
        x = np.random.randint(delta_X,offset)
        y = np.random.randint(delta_X,offset)
        x,y = random_sign(x) , random_sign(y)
        jittered_img = img.rotate(theta).crop((offset + x ,             offset + y ,
                                              offset + x + target_size, offset + y + target_size  ))
        jittered_img_np = np.asarray(jittered_img.getdata())
#        centered_dataset[counter,:] = centered_img
        if doTransform:
            if transform == 'histogram':
                jittered_img_np = get_histogram(jittered_img_np,hist_size = hist_size)
            # TO DO = blur the jittered_img/ jitter_img_np to look more like local_map.
            # TO DO = blur the jittered_img/ jitter_img_np to look more like local_map.
        jittered_dataset[counter,:] = jittered_img_np
    return (None, jittered_dataset)
   
def create_negative_training_set_energy_threshold(image_np,num_samples = 10,threshold = None,delta_X = 5,orig_size = 562, target_size = 170,offset = 196):
    # Currently For each image 
    # Written in create_similar_slightly_jittered_dataset.ipynb
    img = Image.fromarray(np.resize(image_np,(orig_size,orig_size)))
    jittered_dataset = np.zeros((num_samples,target_size*target_size))
    for counter in range(num_samples):
        energyFlag =True
        energyCounter = 0 
        while energyFlag:
            theta = np.random.randint(0,360)
            x = np.random.randint(delta_X,offset)
            y = np.random.randint(delta_X,offset)
            x,y = random_sign(x) , random_sign(y)
            jittered_img = img.rotate(theta).crop((offset + x ,             offset + y ,
                                                  offset + x + target_size, offset + y + target_size  ))
            jittered_img_np = np.asarray(jittered_img.getdata())
            energyFlag = energyThreshold(jittered_img_np,threshold = threshold)
            if energyFlag:
                if not energyCounter:
                    1
#                    print "Rejected threshold",
                else:
                    1
#                    print (energyCounter + 1 ) , 
                energyCounter += 1
            if energyCounter > 5:
#                print "Ignoring this Image"
#                print ""
                raise ValueError('A very specific bad thing happened')
        jittered_dataset[counter,:] = jittered_img_np
    return (None, jittered_dataset)

def energyThreshold(im_np,threshold = None):
    if threshold is None:
        threshold = im_np.size/2
    num_zero = (im_np == 0).sum()
    return (num_zero > threshold)


def random_sign(x):
    r = np.random.randint(0,2)
    if r:
        return x
    else:
        return -x 

def crop_square_dummy(image_1d,orig_size,target_size,offset = None):
    offset = (orig_size - target_size)/2
    image_1d.shape = (orig_size, orig_size)
    image_1d = image_1d[offset:offset + target_size,
                  offset:offset + target_size]
    image_1d.shape =  np.resize(image_1d, [1, target_size * target_size])
    return image_1d


def rotate_coords(x, y, theta, ox, oy):
    """Rotate arrays of coordinates x and y by theta radians about the
    point (ox, oy).

    """
    s, c = np.sin(theta), np.cos(theta)
    x, y = np.asarray(x) - ox, np.asarray(y) - oy
    return x * c - y * s + ox, x * s + y * c + oy

def createRotationMask(init_size, theta, final_size):
    theta = -theta
    sh, sw = init_size,init_size
    ox,oy = init_size/2,init_size/2
    cx, cy = rotate_coords([0, sw, sw, 0], [0, 0, sh, sh], theta, ox, oy)
    dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))
    dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))
    sx, sy = rotate_coords(dx + cx.min(), dy + cy.min(), -theta, ox, oy)
    sx, sy = sx.round().astype(int), sy.round().astype(int)
    mask = (0 <= sx) & (sx < sw) & (0 <= sy) & (sy < sh)
    offset1 = (-final_size + dw)/2;
    offset2 = (final_size + dw)/2;
    return mask[offset1:offset2,offset1:offset2],sy[offset1:offset2,offset1:offset2],sx[offset1:offset2,offset1:offset2]

def createTuples(init_size,theta_samples,final_size):
    params_tuple = []
    for theta in theta_samples:
        params_tuple.append(createRotationMask(init_size,theta* np.pi / 180,final_size)) 
    return params_tuple

def rotateImagesFromTuples(img2D,theta_samples,final_size,params_tuple): 
    rotated_images = np.zeros((len(theta_samples),final_size**2))
    for k,v in enumerate(theta_samples):
        mask,sy,sx = params_tuple[k]
        rotated_images[k] = img2D[sy, sx].flatten()
    return rotated_images


def crop_square(image_1d,orig_size,target_size,offset = None):
    if offset is None:
        offset = (orig_size - target_size)/2
    image = np.resize(image_1d, [orig_size, orig_size])
    image = image[offset:offset + target_size,
                  offset:offset + target_size]
    image = np.resize(image, [1, target_size * target_size])
    return image

def crop(image, x_offset, y_offset, x_target_size, y_target_size):
    x_large = math.sqrt(image.shape[0])
    image = np.resize(image, [x_large, x_large])
    image = image[x_offset:x_offset + x_target_size,
                  y_offset:y_offset + y_target_size]
    image = np.resize(image, [1, x_target_size * y_target_size])
    return image


def show_image(array,ch = 1, shape = 256):
    if ch > 1:
        img = np.resize(array,(ch,shape,shape))
        img = np.swapaxes(img,1,2)
        plt.imshow(img.T)
    else:
        img = np.resize(array,(shape,shape))
        plt.imshow(img, cmap='gray')
        
def show_multiple_images(img_arr, shape_arr):
    k = 1
    for im,shape in zip(img_arr,shape_arr):
        plt.figure()
        plt.title('Image No ' + str(k))
        show_image(im,1,shape=shape)
        k = k + 1

        
def sliding_windows(prior_image,window_h = 256,window_w = 256):
    w, h = prior_image.shape
    strided_image = np.lib.stride_tricks.as_strided(prior_image, 
                                                    shape=[w - window_w + 1, h - window_h + 1, window_w, window_h],
                                                    strides=prior_image.strides + prior_image.strides)
    return strided_image

def sub_sample(strided_images, stride_x = 1, stride_y = 1):
    sub_sampled_images = strided_images[::stride_x,::stride_y]
    return sub_sampled_images

def display_sliding(windowed_images,original = None,i_values = None,skip_images = 10 ,sleeping_time = 0.1):
    from IPython import display
    if i_values is None:
        i_values = range(0,windowed_images.shape[0],skip_images)
    for i in i_values:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        fig = plt.gcf()
        fig.set_size_inches(14.5, 7.5)
        plt.subplot(221)
        plt.imshow(windowed_images[i,0,:,:],cmap='gray')
        plt.title('X sliding window')
        plt.subplot(222)
        plt.imshow(windowed_images[0,i,:,:], cmap='gray')
        plt.title('Y sliding window')
        plt.subplot(224)
        plt.imshow(windowed_images[i,i,:,:], cmap='gray')
        plt.title('Diagonal sliding window')
        if original is not None:
            plt.subplot(223)
            plt.imshow(original, cmap='gray')
            plt.title('Original prior_map')
        time.sleep(sleeping_time)
        
        
def visualize_distance(distance_mat):
    print distance_mat.shape
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(distance_mat, interpolation='nearest', cmap=plt.cm.jet)
    plt.colorbar()
    mini =  np.unravel_index(distance_mat.argmin(),distance_mat.shape)
    minimum_at = str(mini[0]) + " , " + str(mini[1])  
    plt.title("minimum at " + minimum_at)
    plt.show()


def normalize(X,toNormalize):
    if toNormalize:
        norms = np.linalg.norm(X,axis = 1)
        norms[norms == 0 ] = 1
        R = X/norms[:,None]
    else:
        R = X
    return R


def getIndices(total_images,per_example,pos_shape):
    pos_pairs = np.zeros((total_images*per_example,2),dtype=int)
    neg_pairs = np.zeros((total_images*per_example,2),dtype=int)
    for i in range(total_images):
        for j in range(per_example):
            pos_pairs[i*per_example +j] = np.array([i*(per_example+1) , i*(per_example+1) + 1 +j])
            neg_pairs[i*per_example +j] = np.array([i*(per_example+1) , i*(per_example+1) + 1 +j])
    #print "positive pairs"       
    #print pos_pairs
    neg_pairs = neg_pairs + pos_shape[0]
    return pos_pairs, neg_pairs
    
    
def replaceEmptyByMean(X,toMean):
    if toMean:
        mean = X.mean()
        X[X==0] = mean
        print "Mean is " , mean 
    return X

def replaceEmptyByGlobalMean(X,toMean):
    if toMean:
        mean = MEAN
        X[X==0] = mean
    return X

def visualizationAnimation(npMatrix1,npMatrix2,num_images = None ,target_shape = 170, delay = 1.0 ,
                           skip = 1 , inches = (14.5, 7.5)):
    if not num_images:
        num_images = npMatrix1.shape[0]
    for i in range(num_images):
        fig = plt.gcf()
        fig.set_size_inches(inches[0], inches[1])
        plt.subplot(221)
        show_image(npMatrix1[i],1,target_shape)
        plt.title('Dataset 1')
        plt.subplot(222)
        show_image(npMatrix2[i],1,target_shape)
        plt.title('Centered Image')
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(delay)
    