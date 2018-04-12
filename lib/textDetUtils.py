'''
This file defines the required util functions for text detection pre- and post-processing.

NOTE:
'''
from __future__ import print_function 
from keras.applications.vgg16 import preprocess_input
from scipy.spatial import distance as dist
import numpy as np
import cv2
from PIL import Image
import requests
import numpy as np
from StringIO import StringIO
from matplotlib import pyplot
from skimage.morphology import dilation
from skimage.morphology import label as bwlabel

def get_image_from_url(url) :
    '''read a web image from url
    INPUT:
        url = string, web address
    OUTPUT:
        img = np.ndarray
    '''
    response = requests.get(url)
    img = np.array(Image.open(StringIO(response.content)))
    return img

def read_image( file_path ) :
    if ('http' in file_path) :
        return get_image_from_url( file_path )
    else :
        return cv2.imread( file_path, 1 )[...,::-1]

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def decode_one_bbox_mask( bbox_mask, img, proba ) :
    '''Decode one bbox from mask
    INPUT: 
      bbox_mask = np.ndarray, H x W, binary mask
      img = np.ndarray, H x W x 3, original text image
      proba = np.ndarray, H x W, text proba map
    OUTPUT:
      corner_pts = list, each elem is a tuple of coordinates (x, y)
      pr = float, in [0,1], higher is better
      wrapped = np.ndarray, h x w x 3, wrapped text region for OCR
    '''
    # 1. first pass
    contours, hierarchy = cv2.findContours(bbox_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rotrect = cv2.minAreaRect( contours[0] )
    box = cv2.cv.BoxPoints(rotrect)
    #lower_left, upper_left, upper_right, lower_right = [ np.array(pt) for pt in box ]
    upper_left, upper_right, lower_right, lower_left = order_points( np.row_stack( [ np.array(pt) for pt in box ] ) )
    bw = int( np.round( np.sqrt( np.sum( ( upper_right - upper_left ) ** 2 ) ) ) )
    bh = int( np.round( np.sqrt( np.sum( ( upper_right - lower_right ) ** 2 ) ) ) )
    # 2. estimate relax 
    lh = int( min( bw, bh ) )
    sk = max( 2, int( np.round( ( lh * 1.8 - lh ) *.5 ) ) )
    mask_relax = dilation( bbox_mask, np.ones( [sk,sk] ) )
    # 3. second pass
    contours, hierarchy = cv2.findContours(mask_relax,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rotrect = cv2.minAreaRect( contours[0] )
    box = cv2.cv.BoxPoints(rotrect)
    #lower_right, upper_left, upper_right, lower_right = [ np.array(pt) for pt in box ]
    upper_left, upper_right, lower_right, lower_left = order_points( np.row_stack( [ np.array(pt) for pt in box ] ) )
    # 4. estimate transform
    bw = int( np.round( np.sqrt( np.sum( ( upper_right - upper_left ) ** 2 ) ) ) )
    bh = int( np.round( np.sqrt( np.sum( ( upper_right - lower_right ) ** 2 ) ) ) )
    new_pts = np.row_stack( [ [bw, bh], [0,bh], [0, 0], [bw,0] ] ).astype( np.float32 )
    old_pts = np.row_stack( [ lower_right, lower_left, upper_left, upper_right] ).astype( np.float32 )
    M = cv2.getAffineTransform( old_pts[:3], new_pts[:3] )
    # 5. prepare wrapped text image
    wrapped = cv2.warpAffine( img, M, (bw, bh))
    corner_pts = [ upper_left, upper_right, lower_right, lower_left ]
    pr = np.mean( proba[ bbox_mask > 0  ] )
    return corner_pts, pr, wrapped

def decode_text_bboxes( image, proba, labeled_regions ) :
    nb_regs = np.max( labeled_regions )
    results = []
    for reg_idx in range( 1, nb_regs + 1 ) :
        mask = ( labeled_regions == reg_idx ).astype('uint8')
        try :
            corner_pts, pr, wrapped = decode_one_bbox_mask( mask, image, proba )
        except :
            continue
        results.append( [ corner_pts, pr, wrapped ] )
    return results

def visualize_text_proba_map( img, y, scriptID=None ) :
    pyplot.figure(figsize=(10,6.7))
    pyplot.imshow( img )
    pyplot.axis('off')
    pyplot.title('Input')
    pyplot.figure(figsize=(10,6.7))
    pyplot.imshow( y)
    pyplot.axis('off')
    pyplot.title('Text Proba Map ' + ( ' scriptID={}'.format( scriptID ) if scriptID is not None else '' ) )
    pyplot.figure(figsize=(10,6.7))
    pyplot.imshow( np.round( np.float32(img) * y).astype('uint8') )
    pyplot.title('Overlaid')
    pyplot.axis('off')
    
def visualize_individual_regions( decoded_results, proba_threshold = 0.55, area_threshold = 16 ** 2 ) :
    for idx, (corner_pts, pr, wrapped) in enumerate( decoded_results ) :
        h, w = wrapped.shape[:2]
        reg_idx = idx + 1
        if ( pr > proba_threshold ) and ( h * w > area_threshold ) :
            ratio = float(h)/w
            pyplot.figure( figsize=(10,10/ratio))
            pyplot.imshow( wrapped )
            pyplot.title( 'Idx{}-Pr{:.3f}'.format( reg_idx, pr ) )
        else :
            print("INFO: skip", reg_idx, "proba =", pr, "area =", h * w)
####################################################################################################
# DON'T Change Any Code Beyond This Line
####################################################################################################
def convert_imageArray_to_inputTensor( img, apply_padding=True ) :
    '''convert an raw uint8 image array to an input network tensor
    INPUT:
        img = np.ndarray, image array
              size= HxW or HxWx3, dtype=uint8
        apply_padding = bool, whether or not force padding
    OUTPUT:
        x   = np.ndarray, network input tensor
              size=1xH'xW'x3, where H' and W' are padded versions of H and W
              dtype='float32
    '''
    # 1. convert a gray-scale image to RGB if necessary
    if ( img.ndim == 2 ) :
        img = np.dstack( [img for ch in range(3) ] )
    # 2. pad borders if necessary
    if ( apply_padding ) :
        h, w = img.shape[:2]
        hp = int( np.ceil(h/16.) * 16 )
        wp = int( np.ceil(w/16.) * 16 )
        pad_y = hp - h
        pad_x = wp - w
        if ( pad_y == 0 ) and ( pad_x == 0 ) :
            img_pad = img
        else :
            # always pad zeros afterwards
            img_pad = np.pad( img, ([0, pad_y], [0,pad_x], [0,0] ), mode='constant', constant_values = 255 )
    else :
        img_pad = img
    # 3. convert uint8 image to float32
    x = np.float32( img_pad )
    # 4. pad a 3D image array to a 4D tensor
    x = np.expand_dims( x, axis = 0 )
    # 5. substract means learned from image-net
    x = preprocess_input( x )
    return x

def decode_scriptID( dist ) :
    '''decode scriptID from a given distribution
    INPUT:
        dist = 1D or 2D np.ndarray
               if 1D, size of 7,
               if 2D, size of nb_samples-by-7
    OUTPUT:
        scriptID = 1D np.ndarray
    '''
    assert dist.ndim <=2
    if ( dist.ndim == 1 ) :
        dist = dist.reshape([1,-1])
    idx = np.argmax( dist, axis=-1 ).tolist()
    lut = np.array( [ 'NonText',  # Non-text
                      'Latin',    # US
                      'Hebrew',   # Israel
                      'Cyrillic', # Russia
                      'Arabic',   # Arabic
                      'Chinese',  # Chinese
                      'UnknownScript', # Unknown Language
                     ] )
    return lut[idx]
