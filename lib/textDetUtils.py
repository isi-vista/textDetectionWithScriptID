'''
This file defines the required util functions for text detection pre- and post-processing.

NOTE:
'''
from __future__ import print_function
from keras.applications.vgg16 import preprocess_input
from scipy.spatial import distance as dist
import numpy as np
import json
import cv2
from PIL import Image
import requests
import os
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
    if ( img.ndim == 2 ) :
        img = np.dstack([img,img,img])
    else :
        if ( img.shape[-1] == 4 ) :
            img = img[...,:3]
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

def decode_one_bbox_mask_cv3( bbox_mask, img, proba ) :
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
    _, contours, hierarchy = cv2.findContours(bbox_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rotrect = cv2.minAreaRect( contours[0] )
    box = cv2.boxPoints(rotrect)
    #lower_left, upper_left, upper_right, lower_right = [ np.array(pt) for pt in box ]
    upper_left, upper_right, lower_right, lower_left = order_points( np.row_stack( [ np.array(pt) for pt in box ] ) )
    bw = int( np.round( np.sqrt( np.sum( ( upper_right - upper_left ) ** 2 ) ) ) )
    bh = int( np.round( np.sqrt( np.sum( ( upper_right - lower_right ) ** 2 ) ) ) )
    # 2. estimate relax
    lh = int( min( bw, bh ) )
    sk = max( 2, int( np.round( ( lh * 1.8 - lh ) *.5 ) ) )
    mask_relax = dilation( bbox_mask, np.ones( [sk,sk] ) )
    # 3. second pass
    _, contours, hierarchy = cv2.findContours(mask_relax,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rotrect = cv2.minAreaRect( contours[0] )
    box = cv2.boxPoints(rotrect)
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

from hashlib import md5

def set_vals_in_image3d( image3d, new_image3d, mask, channel_idx=None ) :
    if ( new_image3d.ndim == 3 ) :
        ir, ig, ib = [ image3d[...,k] for k in range(3) ]
        nr, ng, nb = [ new_image3d[...,k] for k in range(3) ]
        for old_ch, new_ch in zip( [ir, ig, ib], [nr, ng, nb]) :
            old_ch[mask] = new_ch[mask]
        return np.dstack( [ir, ig, ib])
    else :
        new_ch_list = []
        for k in range(3) :
            old_ch = image3d[...,k]
            if ( k==channel_idx ) :
                old_ch[mask] = new_image3d[mask]
            new_ch_list.append( old_ch )
        return np.dstacK( new_ch_list )

def script_proba_minmax( script_proba_all ) :
    idx = np.argmin( script_proba_all[:,1], axis = 0 )
    return script_proba_all[idx]

def text_proba_minmax( text_proba_all ) :
    text_proba_max = np.argmax( text_proba_all[...,-1], axis = 0 )
    proba_max = np.max( text_proba_all, axis = 0 )
    proba_min = np.min( text_proba_all, axis = 0 )
    valid = (proba_max[...,2]>proba_max[...,1]*1.1 ).astype('int32')
    text = proba_max[...,2] * valid + proba_min[...,2] * (1-valid)
    nontext = proba_min[...,0]
    border = 1 - nontext - text
    text_proba = np.dstack([nontext, border, text])
    membership = text_proba.argmax( axis = -1 )
    regions = bwlabel( membership == 2, neighbors=8 )
    num_regs = np.max( regions )
    new_text_proba = np.array( text_proba )
    for reg_idx in range(1,num_regs+1 ):
        mask = regions == reg_idx
        res_indices = np.unique( text_proba_max[mask] )
        res_contrast_list = []
        for res_idx in res_indices :
            this_text = text_proba_all[res_idx][...,2]
            this_border = text_proba_all[res_idx][...,1]
            res_contrast = np.mean(this_text[mask]/(1e-3+this_border[mask]) )
            res_contrast_list.append( res_contrast )
        best_res_idx = res_indices[np.argmax( res_contrast_list )]
        new_text_proba = set_vals_in_image3d( new_text_proba, text_proba_all[best_res_idx], mask )
    return new_text_proba

def text_proba_minmax_doc( text_proba_all ) :
    text_proba_max = np.argmax( text_proba_all[...,-1], axis = 0 )
    proba_max = np.max( text_proba_all, axis = 0 )
    proba_min = np.min( text_proba_all, axis = 0 )
    nontext = proba_min[...,0]
    #border = proba_max[...,1]
    #text = 1 - nontext - border
    text = proba_max[...,-1]
    border = 1 - nontext - text
    text_proba = np.dstack([nontext, border, text])
    membership = text_proba.argmax( axis = -1 )
    regions = bwlabel( membership == 2, neighbors=8 )
    num_regs = np.max( regions )
    new_text_proba = np.array( text_proba )
    for reg_idx in range(1,num_regs+1 ):
        mask = regions == reg_idx
        res_indices = np.unique( text_proba_max[mask] )
        res_contrast_list = []
        for res_idx in res_indices :
            this_text = text_proba_all[res_idx][...,2]
            this_border = text_proba_all[res_idx][...,1]
            res_contrast = np.mean(this_text[mask]/(1e-3+this_border[mask]) )
            res_contrast_list.append( res_contrast )
        best_res_idx = res_indices[np.argmax( res_contrast_list )]
        new_text_proba = set_vals_in_image3d( new_text_proba, text_proba_all[best_res_idx], mask )
    return new_text_proba
"""
def text_proba_minmax_doc( text_proba_all ) :
    text_proba_max = np.argmax( text_proba_all[...,-1], axis = 0 )
    norm_text_proba_all = text_proba_all / np.max( text_proba_all, axis=(1,2), keepdims=True )
    proba_max = np.max( norm_text_proba_all, axis = 0 )
    proba_min = np.min( norm_text_proba_all, axis = 0 )
    nontext = proba_min[...,0]
    text = proba_max[...,-1]
    border = proba_max[...,1] #1 - nontext - text
    text_proba = np.dstack([nontext, border, text])
    membership = text_proba.argmax( axis = -1 )
    regions = bwlabel( membership == 2, neighbors=8 )
    num_regs = np.max( regions )
    new_text_proba = np.array( text_proba )
    text_proba_all = norm_text_proba_all
    for reg_idx in range(1,num_regs+1 ):
        mask = regions == reg_idx
        res_indices = np.unique( text_proba_max[mask] )
        res_contrast_list = []
        for res_idx in res_indices :
            this_text = text_proba_all[res_idx][...,2]
            this_border = text_proba_all[res_idx][...,1]
            res_contrast = np.mean(this_text[mask]/(1e-3+this_border[mask]) )
            res_contrast_list.append( res_contrast )
        best_res_idx = res_indices[np.argmax( res_contrast_list )]
        new_text_proba = set_vals_in_image3d( new_text_proba, text_proba_all[best_res_idx], mask )
    return new_text_proba
"""
def get_topK_scriptID( script_proba, topK = 2 ) :
    lut = [ 'NonText', 'Latin', 'Hebrew', 'Cyrillic', 'Arabic', 'Chinese', 'TextButUnknown']
    indices = np.argsort( script_proba )[::-1]
    scriptIDs = ["{}-{:.2f}".format(lut[indices[k]], script_proba[indices[k]] ) for k in range( topK )]
    return " | ".join( scriptIDs )

def uncompress_jpeg_buffer( bbox_lut ) :
    jpeg_buf = bbox_lut['jpgbuf']
    reg_img = cv2.imdecode( np.array( jpeg_buf, dtype='uint8' ), 1 ) # bgr
    reg_img = reg_img[...,::-1] # rgb
    return reg_img

def get_bbox_polygon( bbox_lut ) :
    cnt_x = bbox_lut['cntx']
    cnt_y = bbox_lut['cnty']
    pts = np.column_stack( [cnt_x, cnt_y] )
    return pts

def lazy_but_slow_decoder( file_path,
                  textDet_model,
                  scriptID_model,
                  output_dir=None,
                  min_dec_side=256,
                  max_dec_side=3520,
                  num_resolutions=20,
                  proba_threshold=.55,
                  area_threshold=256,
                  contrast_threshold=24,
                  return_proba=False,
                  verbose=1,
                  ) :
    """
    INPUTS:
        file_path = str, path to a local file or URL to a web image
        textDet_model = keras model, pretrained text detection model
        scriptID_model = keras model, pretrained script ID model
        output_dir = None or str, dir to save detected and corrected text regions,
                     if None, then save as JPEG buffer in output lut
        min_dec_side = int, minimum size of a resized image
        max_dec_side = int, maximum size of a resized image
        num_resolutions = int, total number of scales to resize
        proba_threshold = float in (0,1), the minimum text probability to accept a text region
        area_threshold = float, the minimum number of pixels to accept a text region
        contrast_threshold = float in (0,255), the minimum intensity standard deviation to accept a text region
        return_proba = bool, if true, return the raw outputs of both models
        verbose = bool, if true, print out state messages
    OUTPUTS:
        output_lut = dict, containing all decoded results including
                     'filename' -> input image file
                     'md5'      -> image md5 tag
                     'Pr(XXX)'  -> script ID probability of a known scriptID class XXX
                     'bboes'    -> list of bounding box dictionaries, where each element is a dict of
                           'cntx'     -> bbox's x coordinates
                           'cnty'     -> bbox's y coordinates
                           'proba'    -> text probility of this region
                           'area'     -> bbox area
                           'contrast' -> bbox contrast
                           'imgfile'  -> file path the dumped text region image, when output_dir is given
                           'jpgbuf'   -> jpeg buffer (list of uint8) for the text region image, when output_dir is None
        proba_map = ( text_proba, script_proba )
                    - text_proba, i.e. a text probability map of textDet_model, size of imgHeight-by-imgWidth-by-3
                    - script proba, i.e. a script ID probability map, size of 1-by-7
    NOTE:
        output_lut is JSON compatible, in other words, you may dump output_lut to a file by simply doing
            json.dump( output_lut, open( output_file, 'wb' ) )
        and restore it by doing
            output_lut = json.load( open( output_file ) )
    """
    # 1. read image
    img = read_image( file_path )
    # 1.b resize image if necessary
    ih, iw = img.shape[:2]
    text_proba_list, script_proba_list = [], []
    resize_factor_list = np.linspace( float(min_dec_side)/max(ih,iw), min(6,float(max_dec_side)/max(ih,iw)), num_resolutions )
    if ( np.min( np.abs( resize_factor_list - 1 ) > .05 ) ) :
        resize_factor_list = resize_factor_list.tolist() + [1]
    else :
        resize_factor_list = resize_factor_list.tolist()
    if ( verbose ) :
        print("INFO: begin decoding", file_path )
    for resize_factor in resize_factor_list :
        if ( np.abs( resize_factor-1 ) < 2e-2 ):
            rimg = img
        elif ( resize_factor > 1 ) :
            # enlarge input
            nh, nw = int(ih*resize_factor), int(iw*resize_factor)
            rimg = cv2.resize( img, (nw, nh), interpolation=cv2.INTER_CUBIC )
        else :
            # shrink input
            nh, nw = int(ih*resize_factor), int(iw*resize_factor)
            rimg = cv2.resize( img, (nw, nh), interpolation=cv2.INTER_AREA )
        # 2. convert input image to network tensor
        if ( verbose ) :
            print("INFO: now analyzing the scale = {:.2f}".format( resize_factor ))
        rh, rw = rimg.shape[:2]
        if ( max(rh, rw) > 4096 ) :
            break
        elif ( max(rh, rw) < 64 ) :
            continue
        else :
            pass
        x = convert_imageArray_to_inputTensor( rimg )
        # 3. predict text probability map
        text_proba  = textDet_model.predict(x)
        text_proba = text_proba[0,:rh,:rw] # since we always take one sample at a time
        # 3.b resize back to original size
        if ( np.abs( resize_factor-1 ) < 1e-2 ):
            pass
        elif ( resize_factor > 1 ) :
            # shrink proba
            text_proba = cv2.resize( text_proba, (iw, ih), interpolation=cv2.INTER_AREA )
        else :
            # enlarge proba
            text_proba = cv2.resize( text_proba, (iw, ih), interpolation=cv2.INTER_CUBIC )
        # 4. predict script ID
        script_proba = scriptID_model.predict(x[:,::2,::2])
        # 5. update
        text_proba_list.append( np.expand_dims( text_proba, axis=0 ) )
        script_proba_list.append( script_proba )
    text_proba_all = np.concatenate( text_proba_list, axis=0 )
    script_proba_all = np.row_stack( script_proba_list )
    # 6. minimax decode
    if (verbose) :
        print("INFO: begin minimax decoding")
    text_proba = text_proba_minmax( text_proba_all )
    script_proba = script_proba_minmax( script_proba_all )
    scriptID = decode_scriptID( script_proba )
    # 6.b sample connected component analysis based decoder
    membership = text_proba.argmax( axis = -1 )
    regions = bwlabel( membership == 2, neighbors=8 )
    if (img.ndim==2) :
        img = np.dstack([img for k in range(3)])
    decoded_results = decode_text_bboxes( img, text_proba[...,-1], regions )
    # 6. save detection results
    # modify this section to save additional/different information
    valid_indices = []
    prefix = '{}'.format( md5( img ).hexdigest() )
    output_lut = { 'filename' : file_path,
                   'md5' : prefix,
                   'bboxes' : [] }
    for sid, proba in zip( [ 'NonText', 'Latin', 'Hebrew', 'Cyrillic', 'Arabic', 'Chinese', 'TextButUnknown'], script_proba.astype(np.float64) ) :
        output_lut['Pr({})'.format( sid ) ] = proba
    bboxes = []
    for idx, (bbox, proba, wrapped_image) in enumerate( decoded_results ):
        if ( proba < proba_threshold ) :
            if ( verbose ) :
                print ("  Reject", idx, "due to low proba,", proba)
            continue
        upper_left, upper_right, lower_right, lower_left = bbox
        cnt = np.row_stack([upper_left, upper_right, lower_right, lower_left])
        area = cv2.contourArea( cnt )
        if ( area < area_threshold ) :
            if ( verbose ) :
                print ("  Reject", idx, "due to small area,", proba, area)
            continue
        contrast = np.max( wrapped_image.std(axis=(0,1)) )
        if ( contrast < contrast_threshold ) :
            if ( verbose ) :
                print ("  Reject", idx, "due to low contrast,", proba, area, contrast)
            continue
        cnt_x = [ pt[0] for pt in np.round( cnt ).astype('int64') ]
        cnt_y = [ pt[1] for pt in np.round( cnt ).astype('int64') ]
        this_bbox_lut = { 'cntx' : cnt_x, 'cnty' : cnt_y, 'proba' : np.float64(proba), 'area' : np.float64(area), 'contrast' : np.float64(contrast) }
        if ( output_dir is not None ) :
            output_file = os.path.join( output_dir, prefix + '-{:03d}.png'.format(idx) )
            if (verbose) :
                print("INFO: save text region to {}".format( output_file ) )
            cv2.imwrite( output_file, wrapped_image[...,::-1] )
            this_bbox_lut['imgfile'] = output_file
        else :
            output_file = None
            _, buf =  cv2.imencode( 'tmp.jpg', wrapped_image[...,::-1], [int(cv2.IMWRITE_JPEG_QUALITY),99] )
            this_bbox_lut['jpgbuf'] = buf.tolist()
        valid_indices.append(idx)
        bboxes.append( this_bbox_lut )
    output_lut[ 'bboxes' ] = bboxes
    valid_results = [decoded_results[k] for k in valid_indices]
    if ( not return_proba ) :
        return output_lut
    else :
        return output_lut, (text_proba, script_proba)

def find_top_K( val_borders, all_borders, top_K = 5, thresh=.95 ) :
    top_K_list = []
    num_val = float(np.sum( val_borders ))
    for k in range( top_K ) :
        iou_list = []
        for idx, res_border in enumerate( all_borders ) :
            merged = res_border + val_borders
            iou = np.sum( merged == 2)/np.sum( merged >=1 ).astype('float32')
            iou_list.append( iou )
        idx = np.argmax( iou_list )
        val_borders[ all_borders[idx]>0 ] = 0
        all_borders[idx] = 0
        top_K_list.append(idx)
        num_rest = float(np.sum( val_borders ))
        if ( num_rest < (1-thresh) * num_val ) :
            break
    return top_K_list

def find_top_K_acc( acc_val_borders, all_borders, top_K = 5, thresh=.99 ) :
    top_K_list = []
    num_val = float(np.sum( acc_val_borders ))
    for k in range( top_K ) :
        iou_list = []
        for idx, res_border in enumerate( all_borders ) :
            iou = np.sum( acc_val_borders[res_border==1] )
            iou_list.append( iou )
        idx = np.argmax( iou_list )
        if (k>=1) and (top_K_list[-1]==idx) :
            break
        acc_val_borders[ all_borders[idx]>0 ] = 0
        all_borders[idx] = 0
        top_K_list.append(idx)
        num_rest = float(np.sum( acc_val_borders ))
        if ( num_rest < (1-thresh) * num_val ) :
            break
    return top_K_list

paperLut ={ #'A0': [841,1189],
            #'A1': [594, 841],
            #'A2': [420, 594],
            #'A3': [297, 420],
            'A4': [210, 297],
            #'A5': [148, 210],
            #'A6': [105, 148],
            #'A7': [ 74, 105],
            #'A8': [ 52,  74],
            #'A9': [ 37,  52],
            #'A10': [ 26, 37],
            'letter' : [216,279],
            #'ledger' : [279,432],
            #'ANSIC'  : [432,559],
            #'ANSID'  : [559,864],
            #'ANSIE'  : [864,1118], 
}

def parse_paper_size( image_width, image_height ) :
    if ( image_height < image_width ) :
        image_width, image_height = image_height, image_width
    i_ratio = float(image_width)/image_height
    iou = 0
    sofar_best = None
    for name, (pw, ph) in paperLut.iteritems() :
        p_ratio = float(pw)/ph
        p_iou = min(i_ratio, p_ratio)/max(i_ratio, p_ratio)
        if ( p_iou > iou ) :
            iou = p_iou
            sofar_best = name
    return sofar_best, paperLut[sofar_best]

def compute_fontsize_in_pixels_for_letter( image_height, font_size=10 ) :
    known_letter_height = 1650.
    known_coef_pixel_height_to_fontsize = 1.5
    return font_size * known_coef_pixel_height_to_fontsize / known_letter_height * image_height

def compute_fontsize_in_pixels_for_paper( image_height, font_size=10, paper_name='letter' ) :
    if ( paper_name == 'letter' ) :
        return compute_fontsize_in_pixels_for_letter( image_height, font_size )
    else :
        target_paper_height = paperLut[paper_name][1]
        letter_paper_height = float(paperLut['letter'][1])
        return compute_fontsize_in_pixels_for_letter( image_height, font_size ) * target_paper_height / letter_paper_height

def get_horizontal_border( border_bin, minLineLength, maxLineGap, thickness=3 ) :
    edges = (border_bin).astype('uint8') * 255
    horizontal = np.zeros( border_bin.shape, dtype='uint8' )
    lines = cv2.HoughLinesP(edges,1,np.pi/180,minLineLength//2,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        slope = float(y2-y1)/(x2-x1)
        if ( abs( slope ) < 1 ) :
            cv2.line(horizontal,(x1,y1),(x2,y2),1,thickness)
    return horizontal

def compute_horizontal_border( border_edge, est_min_lh ) :
    minFontSize = max(3,int(est_min_lh/2) )
    minLineLength = 5 * minFontSize
    maxLineGap = est_min_lh * 2
    hborder = get_horizontal_border( border_edge, minLineLength, maxLineGap, thickness=minFontSize//2 )
    return hborder
def select_best_resolutions( ref, text_proba_all, thresh=.99 ) :
    valid_idx = range( len(text_proba_all) )
    sel_idx = []
    num_val = np.sum( ref > .33 )
    #print "num_val =", num_val, "thresh=",(1-thresh)*num_val
    while 1 :
        iou_list = []
        for idx in valid_idx:
            hyp = text_proba_all[idx,...,1]
            iou = np.sum(np.minimum( hyp, ref )>.33)/(1.+np.sum(np.maximum( hyp, ref )>.33))
            iou_list.append(iou)
        # get best idx
        best_vidx = np.argmax( iou_list )
        best_idx = valid_idx[best_vidx]
        # update selected idx list
        sel_idx.append( best_idx )
        # update valid idx list
        valid_idx.remove( best_idx )
        # check stop condition
        ref = np.clip( ref - text_proba_all[best_idx,...,1], 0, 1 )
        num_rest = np.sum( ref > .33 )
        #print "num_rest =", num_rest
        if ( num_rest < (1-thresh)*num_val ) :
            break
    return sel_idx
def lazy_but_slow_doc_decoder( file_path,
                  textDet_model,
                  scriptID_model,
                  output_dir=None,
                  num_resolutions=10,
                  proba_threshold=.5,
                  contrast_threshold=None,
                  return_proba=False,
                  verbose=1,
                  ) :
    """
    INPUTS:
        file_path = str, path to a local file or URL to a web image
        textDet_model = keras model, pretrained text detection model
        scriptID_model = keras model, pretrained script ID model
        output_dir = None or str, dir to save detected and corrected text regions,
                     if None, then save as JPEG buffer in output lut
        num_resolutions = int, total number of scales to resize
        proba_threshold = float in (0,1), the minimum text probability to accept a text region
        area_threshold = float, the minimum number of pixels to accept a text region
        contrast_threshold = float in (0,255), the minimum intensity standard deviation to accept a text region
        return_proba = bool, if true, return the raw outputs of both models
        verbose = bool, if true, print out state messages
    OUTPUTS:
        output_lut = dict, containing all decoded results including
                     'filename' -> input image file
                     'md5'      -> image md5 tag
                     'Pr(XXX)'  -> script ID probability of a known scriptID class XXX
                     'bboes'    -> list of bounding box dictionaries, where each element is a dict of
                           'cntx'     -> bbox's x coordinates
                           'cnty'     -> bbox's y coordinates
                           'proba'    -> text probility of this region
                           'area'     -> bbox area
                           'contrast' -> bbox contrast
                           'imgfile'  -> file path the dumped text region image, when output_dir is given
                           'jpgbuf'   -> jpeg buffer (list of uint8) for the text region image, when output_dir is None
        proba_map = ( text_proba, script_proba )
                    - text_proba, i.e. a text probability map of textDet_model, size of imgHeight-by-imgWidth-by-3
                    - script proba, i.e. a script ID probability map, size of 1-by-7
    NOTE:
        output_lut is JSON compatible, in other words, you may dump output_lut to a file by simply doing
            json.dump( output_lut, open( output_file, 'wb' ) )
        and restore it by doing
            output_lut = json.load( open( output_file ) )
    """
    # 1. read image
    img = read_image( file_path )
    # 1.b resize image if necessary
    ih, iw = img.shape[:2]
    text_proba_list, script_proba_list = [], []
    paper_name = parse_paper_size( ih, iw )
    est_min_lh = compute_fontsize_in_pixels_for_paper( max(ih,iw), font_size=6 )
    est_max_lh = compute_fontsize_in_pixels_for_paper( max(ih,iw), font_size=100 )
    lh_threshold = compute_fontsize_in_pixels_for_paper( max(ih,iw), font_size=8 )
    proba_threshold = min( max(.25,proba_threshold*est_min_lh/12.), proba_threshold )
    contrast_threshold = np.max( np.std( img, axis=(0,1) ) ) * .5 or contrast_threshold
    print ("use proba_thresh =", proba_threshold, "contrast_thresh =", contrast_threshold, 'lineheight_thresh=', lh_threshold )
    #resize_factor_list = np.linspace( 20./est_max_lh, 30./est_min_lh, num_resolutions )
    resize_factor_list = [ 20./compute_fontsize_in_pixels_for_paper( max(ih,iw), font_size=fs ) for fs in [6,8,12,16,20,24,32,40,60,100] ] \
                       + [ 30./compute_fontsize_in_pixels_for_paper( max(ih,iw), font_size=fs ) for fs in [6,8,12,16,20,24,32,40,60,100] ]
    resize_factor_list = np.sort( resize_factor_list )
    print("use resize_factors=", resize_factor_list )
    if ( np.min( np.abs( resize_factor_list - 1 ) > .05 ) ) :
        resize_factor_list = resize_factor_list.tolist() + [1]
    else :
        resize_factor_list = resize_factor_list.tolist()
    if ( verbose ) :
        print("INFO: begin decoding", file_path )
    for resize_factor in resize_factor_list :
        if ( np.abs( resize_factor-1 ) < 2e-2 ):
            rimg = img
        elif ( resize_factor > 1 ) :
            # enlarge input
            nh, nw = int(ih*resize_factor), int(iw*resize_factor)
            rimg = cv2.resize( img, (nw, nh), interpolation=cv2.INTER_CUBIC )
        else :
            # shrink input
            nh, nw = int(ih*resize_factor), int(iw*resize_factor)
            rimg = cv2.resize( img, (nw, nh), interpolation=cv2.INTER_AREA )
        # 2. convert input image to network tensor
        if ( verbose ) :
            print("INFO: now analyzing the scale = {:.2f}".format( resize_factor ))
        rh, rw = rimg.shape[:2]
        if ( max(rh, rw) > 4096 ) :
            break
        elif ( max(rh, rw) < 64 ) :
            continue
        else :
            pass
        x = convert_imageArray_to_inputTensor( rimg )
        # 3. predict text probability map
        text_proba  = textDet_model.predict(x)
        text_proba = text_proba[0,:rh,:rw] # since we always take one sample at a time
        # 3.b resize back to original size
        if ( np.abs( resize_factor-1 ) < 1e-2 ):
            pass
        elif ( resize_factor > 1 ) :
            # shrink proba
            text_proba = cv2.resize( text_proba, (iw, ih), interpolation=cv2.INTER_AREA )
        else :
            # enlarge proba
            text_proba = cv2.resize( text_proba, (iw, ih), interpolation=cv2.INTER_CUBIC )
        # 4. predict script ID
        script_proba = scriptID_model.predict(x[:,::2,::2])
        # 5. update
        text_proba_list.append( np.expand_dims( text_proba, axis=0 ) )
        script_proba_list.append( script_proba )
    text_proba_all = np.concatenate( text_proba_list, axis=0 )
    script_proba_all = np.row_stack( script_proba_list )
    all_borders = ( text_proba_all[...,1] > .5 ).astype('int')
    twopass = False
    if ( not twopass )  :
        if 1:
            #acc_val_borders = np.sum( all_borders, axis=0 )
            #top_K_list = find_top_K_acc( acc_val_borders, np.array(all_borders), thresh=.99)
            val_borders = ( np.sum( all_borders, axis=0 ) > 1 ).astype('int')
            top_K_list = find_top_K_acc( val_borders, np.array(all_borders), thresh=.99)
        else :
            if 0 :
                val_borders = ( np.sum( all_borders, axis=0 ) > 1 ).astype('int')
                cthresh = .99
                top_K_list = find_top_K( val_borders, np.array(all_borders), thresh=cthresh)
            else :
                border_edge = np.sum( all_borders, axis=0 ) > 1
                val_borders = compute_horizontal_border( border_edge, est_min_lh )*max(2,num_resolutions//4)+np.sum( all_borders, axis=0 )
                cthresh = .99
                top_K_list = find_top_K_acc( val_borders, np.array(all_borders), thresh=cthresh)

        print (top_K_list)
        text_proba_sel = text_proba_all[top_K_list]
        script_proba_sel = script_proba_all[top_K_list]
        # 6. minimax decode
        if (verbose) :
            print("INFO: begin minimax decoding")
        text_proba = text_proba_minmax_doc( text_proba_sel )
        script_proba = script_proba_minmax( script_proba_sel )
        scriptID = decode_scriptID( script_proba )
    else :
        text_proba = text_proba_minmax_doc( text_proba_all )
        ref = text_proba[...,1]
        sel_idx = select_best_resolutions( ref, text_proba_all, thresh=.99 )
        text_proba = text_proba_minmax_doc( text_proba_all[sel_idx] )
        script_proba = script_proba_minmax( script_proba_all[sel_idx] )
        scriptID = decode_scriptID( script_proba )
        if (verbose) :
            print ("INFO: two-pass, selected resolution=", sel_idx )
        #non_text = np.min( text_proba_all[sel_idx,...,0], axis=0 )
        #border = np.minimum( np.sum(text_proba_all[sel_idx,...,1],axis=0 ), 1-non_text )
        #border = np.max(text_proba_all[sel_idx,...,1],axis=0 )
        #text = 1 - non_text - border
        #text_proba = np.dstack([non_text, border, text ] )
    # 6.b sample connected component analysis based decoder
    membership = text_proba.argmax( axis = -1 )
    regions = bwlabel( membership == 2, neighbors=8 )
    if (img.ndim==2) :
        img = np.dstack([img for k in range(3)])
    decoded_results = decode_text_bboxes( img, text_proba[...,-1], regions )
    # 6. save detection results
    # modify this section to save additional/different information
    valid_indices = []
    prefix = '{}'.format( md5( np.ascontiguousarray(img) ).hexdigest() )
    output_lut = { 'filename' : file_path,
                   'md5' : prefix,
                   'bboxes' : [] }
    for sid, proba in zip( [ 'NonText', 'Latin', 'Hebrew', 'Cyrillic', 'Arabic', 'Chinese', 'TextButUnknown'], script_proba.astype(np.float64) ) :
        output_lut['Pr({})'.format( sid ) ] = proba
    bboxes = []
    for idx, (bbox, proba, wrapped_image) in enumerate( decoded_results ):
        if ( proba < proba_threshold ) :
            if ( verbose ) :
                print ("  Reject", idx, "due to low proba,", proba)
            continue
        upper_left, upper_right, lower_right, lower_left = bbox
        cnt = np.row_stack([upper_left, upper_right, lower_right, lower_left])
        #area = cv2.contourArea( cnt )
        lh = min( np.max( cnt, axis=0 ) - np.min(cnt, axis=0) )
        if ( lh < lh_threshold ) :
            if ( verbose ) :
                print ("  Reject", idx, "due to short height,", proba, lh)
            continue
        contrast = np.max( wrapped_image.std(axis=(0,1)) )
        if ( contrast < contrast_threshold ) :
            if ( verbose ) :
                print ("  Reject", idx, "due to low contrast,", proba, lh, contrast)
            continue
        cnt_x = [ pt[0] for pt in np.round( cnt ).astype('int64') ]
        cnt_y = [ pt[1] for pt in np.round( cnt ).astype('int64') ]
        this_bbox_lut = { 'cntx' : cnt_x, 'cnty' : cnt_y, 'proba' : np.float64(proba), 'lineheight' : np.float64(lh), 'contrast' : np.float64(contrast) }
        if ( output_dir is not None ) :
            output_file = os.path.join( output_dir, prefix + '-{:03d}.png'.format(idx) )
            if (verbose) :
                print("INFO: save text region to {}".format( output_file ) )
            cv2.imwrite( output_file, wrapped_image[...,::-1] )
            this_bbox_lut['imgfile'] = output_file
        else :
            output_file = None
            _, buf =  cv2.imencode( 'tmp.jpg', wrapped_image[...,::-1], [int(cv2.IMWRITE_JPEG_QUALITY),99] )
            this_bbox_lut['jpgbuf'] = buf.tolist()
        valid_indices.append(idx)
        bboxes.append( this_bbox_lut )
    output_lut[ 'bboxes' ] = bboxes
    valid_results = [decoded_results[k] for k in valid_indices]
    if ( not return_proba ) :
        return output_lut
    else :
        return output_lut, (text_proba, script_proba)

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
