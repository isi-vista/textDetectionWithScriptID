'''
This file defines the required util functions for text detection pre- and post-processing.

NOTE:
'''
from __future__ import print_function
from future.utils import iteritems
from keras.applications.vgg16 import preprocess_input
from scipy.spatial import distance as dist
import numpy as np
from hashlib import md5
import json
import cv2
from PIL import Image
from datetime import datetime
import requests
import os
try :
    from StringIO import StringIO
except :
    from io import StringIO
from matplotlib import pyplot
from skimage.morphology import dilation
from skimage.filters import threshold_otsu, threshold_niblack,  threshold_sauvola

MINSIDE=256.
MAXSIDE=3200.

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
            img = img[...,1:]
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

def parse_detection_results( url, output_lut, text_proba=None, script_proba=None, show_region=True) :
    img = read_image( url )
    ih, iw = img.shape[:2]
    resize_factor = output_lut['resize']
    if ( abs(resize_factor-1)> 0.02 ) :
        nh, nw = int(ih*resize_factor), int(iw*resize_factor)
        img = cv2.resize( img, (nw, nh), interpolation=cv2.INTER_AREA if (resize_factor<1) else cv2.INTER_CUBIC )
    # 1. decode proba if necessary
    if ( script_proba is not None ) and ( text_proba is not None ):
        scriptID = get_topK_scriptID( script_proba )
        visualize_text_proba_map( img, text_proba, scriptID )
    for key, val in iteritems(output_lut) :
        if ( 'Pr' in key ) :
            print('{:>20} = {:.2f}'.format(key, val))
    debug = np.array( img, dtype='uint8' )
    for idx, bbox_lut in enumerate( output_lut['bboxes'] ) :
        proba = bbox_lut['proba']
        lh  = bbox_lut['lineheight']
        contrast = bbox_lut['contrast']
        if ( bbox_lut.has_key('imgfile') ) :
            reg_img = cv2.imread( bbox_lut['imgfile'], 1 )[...,::-1]
        elif ( bbox_lut.has_key('jpgbuf')) :
            reg_img = uncompress_jpeg_buffer( bbox_lut )
        reg_pts = get_bbox_polygon( bbox_lut )
        if ( show_region ) :
            pyplot.figure()
            pyplot.imshow( reg_img )
            pyplot.title('Reg#{:d} : Pr={:.2f} Lh={:.2f} Contrast={:.2f}'.format( idx, proba, lh, contrast ) )
        cv2.polylines( debug, [reg_pts], True, (0,255,0), thickness=5 )
    pyplot.figure( figsize=(10,10))
    pyplot.imshow( debug )
    pyplot.title('detected text regions')

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


paperLut ={ 'A4': [210, 297],
            'letter' : [216,279],
           }

def parse_paper_size( image_width, image_height ) :
    if ( image_height < image_width ) :
        image_width, image_height = image_height, image_width
    i_ratio = float(image_width)/image_height
    iou = 0
    sofar_best = None
    for name, (pw, ph) in iteritems(paperLut) :
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

def estimate_dominant_fontsize( img, dark_text=None, verbose=0 ) :
    ih, iw = img.shape[:2]
    kernel = np.array( [1,2,3,4,5,4,3,2,1], dtype=np.float32)
    kernel = kernel / np.sum( kernel )
    paper_name, _ = parse_paper_size( ih, iw )
    est_lh = compute_fontsize_in_pixels_for_paper( max(ih,iw), font_size=10, paper_name=paper_name )
    # binarize image
    if ( img.ndim == 2 ) :
        gimg = img
    else :
        gimg = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    #mimg = threshold_otsu( gimg )
    mimg = threshold_sauvola(gimg, window_size=min( min(ih,iw)//20*2+1, int(est_lh*5)//2*2+1) )
    if ( dark_text is not None ) :
        if ( dark_text ) :
            text = (gimg < mimg).astype('uint8') * 255
        else :
            text = (gimg > mimg).astype('uint8') * 255
    else :
        bimg = (gimg < mimg)
        if ( np.sum(bimg) < bimg.size//2 ) :
            text = bimg.astype('uint8') * 255
            verbose_print( verbose, "INFO: estimated case = DARK TEXT on bright background")
        else :
            text = (1-bimg.astype('uint8')) * 255
            verbose_print( verbose, "INFO: estimated case = BRIGHT TEXT on dark background")
    cvCCA = cv2.connectedComponentsWithStats(text, 8, cv2.CV_32S)
    num_regs,labels,reg_stats,centroids = cvCCA
    # estimate dominated fontsize
    rel_lut = dict()
    for label in range(num_regs):
        # retrieving the width of the bounding box of the component
        width = reg_stats[label, cv2.CC_STAT_WIDTH]
        # retrieving the height of the bounding box of the component
        height = reg_stats[label, cv2.CC_STAT_HEIGHT]
        if ( height > est_lh * .75 ) and ( height < est_lh * 10):
            if ( height not in rel_lut ) :
                rel_lut[ height ] = 0
            rel_lut[ height ] += width
    keys = list(rel_lut)
    vals = list(rel_lut.values())
    vall = np.sum( vals )
    obs = np.zeros( np.max(keys)+1)
    for k, v in zip( keys, vals ) :
        obs[k] = float(v)/vall
    n_obs = np.convolve( obs, kernel, mode='same')
    dominant_fontsize = np.argmax( n_obs )
    return dominant_fontsize


def time_stamp() :
    return '{}'.format( datetime.now() )[2:22]

def verbose_print( verbose=0, *objects ) :
    if ( verbose == 0 ) :
        pass
    elif ( verbose == 1 ) :
        print( ' '.join( [ str(obj) for obj in objects ] ) )
    else :
        print( time_stamp(), '|', ' '.join( [ str(obj) for obj in objects ] ) )
    return

def decode_horizontal_text_bbox( text_proba, text_label, img, reg_stats ) :
    results = []
    ih, iw = img.shape[:2]
    for reg_idx in range( 1, len( reg_stats ) ) :
        # 1. get CC bounding box info
        reg_left, reg_top, reg_width, reg_height, _ = reg_stats[reg_idx]
        # 2. relax text border
        sk_h = max( 2, int(np.round(np.round(reg_height*.8)*.5)) )
        sk_w = max( sk_h * 2, reg_height//2 )
        reg_right = min(iw, reg_left+reg_width+sk_w)
        reg_bot = min(ih, reg_top+reg_height+sk_h)
        reg_left = max(0, reg_left-sk_w)
        reg_top = max(0, reg_top-sk_h)
        reg_height = reg_bot - reg_top + 1
        reg_width = reg_right - reg_left + 1
        # 3. crop region for fast analysis
        reg_proba = text_proba[reg_top:reg_top+reg_height+1,reg_left:reg_left+reg_width+1]
        reg_mask = text_label[reg_top:reg_top+reg_height+1,reg_left:reg_left+reg_width+1] == reg_idx
        reg_img = img[reg_top:reg_top+reg_height+1,reg_left:reg_left+reg_width+1]
        # 4. compute text probability
        pr = np.mean( reg_proba[ reg_mask] )
        upper_left  = [ reg_left, reg_top ]
        upper_right = [ reg_left + reg_width, reg_top ]
        lower_left  = [ reg_left, reg_top + reg_height ]
        lower_right = [ reg_left + reg_width, reg_top + reg_height ]
        corner_pts = [ upper_left, upper_right, lower_right, lower_left ]
        results.append( [corner_pts, pr, reg_img ] )
    return results

def decode_one_bbox( text_proba, text_label, img, reg_stats, reg_idx ) :
    ih, iw = img.shape[:2]
    try :
        # 1. get CC bounding box info
        reg_left, reg_top, reg_width, reg_height, _ = reg_stats[reg_idx]
        reg_proba = text_proba[reg_top:reg_top+reg_height+1,reg_left:reg_left+reg_width+1]
        reg_mask = text_label[reg_top:reg_top+reg_height+1,reg_left:reg_left+reg_width+1] == reg_idx
        pr = np.mean( reg_proba[ reg_mask ] )
        # 2. relax text border
        sk_w = sk_h = max(3,min(reg_width,reg_height)//2)
        reg_right = min(iw, reg_left+reg_width+sk_w)
        reg_bot = min(ih, reg_top+reg_height+sk_h)
        reg_left = max(0, reg_left-sk_w)
        reg_top = max(0, reg_top-sk_h)
        reg_height = reg_bot - reg_top + 1
        reg_width = reg_right - reg_left + 1
        # 2. crop region for fast analysis
        reg_proba = text_proba[reg_top:reg_top+reg_height+1,reg_left:reg_left+reg_width+1]
        reg_mask = ( text_label[reg_top:reg_top+reg_height+1,reg_left:reg_left+reg_width+1] == reg_idx).astype('uint8')
        reg_img = img[reg_top:reg_top+reg_height+1,reg_left:reg_left+reg_width+1]
        # 3. rotated text decoding
        _, contours, hierarchy = cv2.findContours(reg_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        rotrect = cv2.minAreaRect( contours[0] )
        box = cv2.boxPoints(rotrect)
        upper_left, upper_right, lower_right, lower_left = order_points( np.row_stack( [ np.array(pt) for pt in box ] ) )
        bw = int( np.round( np.sqrt( np.sum( ( upper_right - upper_left ) ** 2 ) ) ) )
        bh = int( np.round( np.sqrt( np.sum( ( upper_right - lower_right ) ** 2 ) ) ) )
        # 3.b estimate relax
        lh = int( min( bw, bh ) )
        sk = max( 3, int( np.round( ( lh * 1.8 - lh ) *.5 ) ) )
        mask_relax = dilation( reg_mask, np.ones( [sk,sk] ) )
        # 3.c second pass
        _, contours, hierarchy = cv2.findContours(mask_relax,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        rotrect = cv2.minAreaRect( contours[0] )
        box = cv2.boxPoints(rotrect)
        upper_left, upper_right, lower_right, lower_left = order_points( np.row_stack( [ np.array(pt) for pt in box ] ) )
        # 4. estimate transform
        bw = int( np.round( np.sqrt( np.sum( ( upper_right - upper_left ) ** 2 ) ) ) )
        bh = int( np.round( np.sqrt( np.sum( ( upper_right - lower_right ) ** 2 ) ) ) )
        new_pts = np.row_stack( [ [bw, bh], [0,bh], [0, 0], [bw,0] ] ).astype( np.float32 )
        old_pts = np.row_stack( [ lower_right, lower_left, upper_left, upper_right] ).astype( np.float32 )
        M = cv2.getAffineTransform( old_pts[:3], new_pts[:3] )
        # 5. prepare wrapped text image
        wrapped = cv2.warpAffine( reg_img, M, (bw, bh))
        corner_pts = [ [x+reg_left, y+reg_top] for x, y in [ upper_left, upper_right, lower_right, lower_left ] ]
        # 6. update
        return [ corner_pts, pr, wrapped ]
    except :
        return None

def decode_rotated_text_bbox( text_proba, text_label, img, reg_stats, n_jobs=8 ) :
    use_para = False
    try :
        from sklearn.externals.joblib import Parallel, delayed
        use_para = True
    except :
        pass
    all_reg_indices = range( 1, len( reg_stats ) )
    if ( use_para ) :
        try :
            # use parallel process
            results = Parallel( n_jobs=n_jobs, verbose=0, backend='threading' )( delayed( decode_one_bbox )( text_proba, text_label, img, reg_stats, reg_idx ) for reg_idx in all_reg_indices )
            return filter( lambda r : r is not None, results )
        except :
            pass
    results = []
    for reg_idx in all_reg_indices :
        res = decode_one_bbox( text_proba, text_label, img, reg_stats, reg_idx )
        if ( res is not None ) :
            results.append( res )
    return results


def reject_one_text_region( idx, bbox_dec_result, prefix, save_mode='buffer', output_dir=None, proba_threshold=0, contrast_threshold=0, lh_threshold=0, verbose=0 ) :
    bbox, proba, wrapped_image = bbox_dec_result
    if ( proba < proba_threshold ) :
        verbose_print( verbose, "INFO: reject #", idx, "due to low proba,", proba, "<", proba_threshold )
        return None
    upper_left, upper_right, lower_right, lower_left = bbox
    cnt = np.row_stack([upper_left, upper_right, lower_right, lower_left])
    #area = cv2.contourArea( cnt )
    lh = min( np.max( cnt, axis=0 ) - np.min(cnt, axis=0) )
    if ( lh < lh_threshold ) :
        verbose_print( verbose, "INFO: reject #", idx, "due to short height,", lh, "<", lh_threshold )
        return None
    contrast = np.max( wrapped_image.std(axis=(0,1)) )
    if ( contrast < contrast_threshold ) :
        verbose_print( verbose, "INFO: reject #", idx, "due to low contrast,", contrast, "<", contrast_threshold)
        return None
    cnt_x = [ pt[0] for pt in np.round( cnt ).astype('int64') ]
    cnt_y = [ pt[1] for pt in np.round( cnt ).astype('int64') ]
    this_bbox_lut = { 'cntx' : cnt_x,
                      'cnty' : cnt_y,
                      'proba' : np.float64(proba),
                      'lineheight' : np.float64(lh),
                      'contrast' : np.float64(contrast) }
    if ( save_mode == 'disk' ):
        output_file = os.path.join( output_dir, prefix + '-{:03d}.png'.format(idx) )
        verbose_print( verbose, "INFO: successfully save text region to {}".format( output_file ) )
        cv2.imwrite( output_file, wrapped_image[...,::-1] )
        this_bbox_lut['imgfile'] = output_file
    elif ( save_mode == 'buffer' ) :
        _, buf =  cv2.imencode( 'tmp.jpg', wrapped_image[...,::-1], [int(cv2.IMWRITE_JPEG_QUALITY),99] )
        this_bbox_lut['jpgbuf'] = buf.tolist()
    else :
        pass
    return this_bbox_lut

def reject_text_regions( decoded_results, prefix, save_mode='buffer', output_dir=None, proba_threshold=0, contrast_threshold=0, lh_threshold=0, verbose = 1, n_jobs = 8 ) :
    if ( output_dir is None ) :
        this_output_dir = None
    else :
        this_output_dir = os.path.join( output_dir, prefix )
        if ( not os.path.isdir( this_output_dir ) ) :
            os.makedirs( this_output_dir )
    use_para = False
    try : 
        from sklearn.externals.joblib import Parallel, delayed
        use_para = True
    except :
        pass
    if use_para :
        try :
            bboxes = Parallel( n_jobs=n_jobs, verbose=0, backend='threading')( delayed( reject_one_text_region )( idx, bbox_dec_result, prefix, save_mode, this_output_dir, proba_threshold, contrast_threshold, lh_threshold, verbose=0 ) for idx, bbox_dec_result in enumerate( decoded_results ) )
            return filter( lambda b : b is not None, bboxes )
        except :
            pass
    bboxes = []
    for idx, bbox_dec_result in enumerate( decoded_results ):
        this_bbox_lut = reject_one_text_region( idx, bbox_dec_result, prefix, save_mode, this_output_dir, proba_threshold, contrast_threshold, lh_threshold, verbose )
        if ( this_bbox_lut is not None ) :
            bboxes.append( this_bbox_lut )
    return bboxes

def simple_decoder( file_path,
                    textDet_model,
                    scriptID_model=None,
                    output_dir=None,
                    dom_font=None,
                    dark_text=None,
                    rotated_text=False,
                    proba_threshold=.5,
                    lh_threshold=15,
                    contrast_threshold=24.,
                    return_proba=True,
                    n_jobs=1,
                    verbose=2) :
    """
    INPUTS:
        ------------------------------------------------------------------------
        | Mandentory Parameters
        ------------------------------------------------------------------------
        file_path = str, path to a local file or URL to a web image
        textDet_model = keras model, pretrained text detection model
        scriptID_model = None or keras model, pretrained script ID model
                         if None, then no scriptID classification
        output_dir = None or str or 'SKIP', dir to save detected and corrected text regions,
                     if None, then text regions as JPEG buffers
                     if SKIP, then not save text regions
        ------------------------------------------------------------------------
        | Prior Knowledge Parameters (better performance if they are provided)
        ------------------------------------------------------------------------
        dom_font = int or None, dominant fontsize height in terms of pixels
                   if None, then apply automatic estimation
        dark_text = bool or None, whether texts on image is dark/black or not
                    if None, then apply automatic estimation
        rotated_text = bool, whether text regions are rotated or not
                       if False, then faster decoding is applied
        ------------------------------------------------------------------------
        | Simple Rules to Reject A Text Region
        ------------------------------------------------------------------------
        proba_threshold = float in (0,1), the minimum text probability to accept a text region
        lh_threshold = float, the minimum line height to accept a text region
        contrast_threshold = float in (0,255), the minimum intensity standard deviation to accept a text region
        ------------------------------------------------------------------------
        | Others
        ------------------------------------------------------------------------
        return_proba = bool, if true, return the raw outputs of both models
        n_jobs = int, if greater than 1, use multiple CPUs
        verbose = bool, if true, print out state messages

    OUTPUTS:
        output_lut = dict, containing all decoded results including
                     'filename' -> input image file
                     'resize'   -> resize factor for text detection analysis
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
    """
    verbose_print( verbose, "INFO: begin text detection for {}".format( file_path ) )
    # 1. read image
    img = read_image( file_path )
    # 1.b resize image if necessary
    ih, iw = img.shape[:2]
    verbose_print( verbose, "INFO: original image size = ({},{})".format( ih, iw ) )
    # 1.c estimate resize factor
    target_lh = 22.5
    if ( dom_font is None ) :
        paper_name, _ = parse_paper_size( ih, iw )
        min_lh = compute_fontsize_in_pixels_for_paper( max(ih,iw), font_size=8, paper_name='letter' )
        max_lh = compute_fontsize_in_pixels_for_paper( max(ih,iw), font_size=48, paper_name='letter' )
        est_lh = min( max( estimate_dominant_fontsize( img, dark_text, verbose ), min_lh ), max_lh )
        verbose_print( verbose, "INFO: estimated input doc paper={}".format( paper_name ) )
        verbose_print( verbose, "INFO: estimated dominant line height on original is {} pixels high".format(est_lh))
    else :
        est_lh = float(dom_font)
    # 1.d compute resize factor
    resize_factor = max( .33, target_lh/est_lh )# map the dominatant fontsize to 25 pixel high
    verbose_print( verbose, "INFO: resize input by {:.2f} to match line height {}".format( resize_factor, target_lh ))
    if ( abs(resize_factor-1)> 0.05 ) :
        if (resize_factor<1) :
            method = cv2.INTER_AREA
        else :
            method = cv2.INTER_CUBIC
        nh, nw = int(ih*resize_factor), int(iw*resize_factor)
        img = cv2.resize( img, (nw, nh), interpolation=method )
    else :
        resize_factor = 1
        nh, nw = ih, iw
    # 2. convert input image to network tensor
    verbose_print( verbose, "INFO: begin FCN text detection")
    x = convert_imageArray_to_inputTensor( img )
    # 3. predict text probability map
    text_proba  = textDet_model.predict(x)
    text_proba = text_proba[0,:nh,:nw] # since we always take one sample at a time
    if ( scriptID_model is not None ) :
        script_proba = scriptID_model.predict(x)[0]
    else :
        script_proba = None
    verbose_print( verbose, "INFO: done FCN text detection")
    # 4. decode individual text bbox
    membership = text_proba.argmax( axis = -1 )
    text_mask = (membership==2).astype('uint8')
    num_regs,labels,reg_stats,centroids = cv2.connectedComponentsWithStats( text_mask, 8, cv2.CV_32S )
    if ( not rotated_text ) :
        verbose_print( verbose, "INFO: localize horizontal text bounding boxes" )
        decoded_results = decode_horizontal_text_bbox( text_proba[...,-1], labels, img, reg_stats )
    else :
        verbose_print( verbose, "INFO: localize rotated text bounding boxes" )
        decoded_results = decode_rotated_text_bbox( text_proba[...,-1], labels, img, reg_stats, n_jobs=n_jobs )
    # 5. save outputs
    prefix = '{}'.format( md5( np.ascontiguousarray(img) ).hexdigest() )
    output_lut = { 'filename' : file_path,
                   'resize' : np.float64(resize_factor),
                   'md5' : prefix,}
    for sid, proba in zip( [ 'NonText', 'Latin', 'Hebrew', 'Cyrillic', 'Arabic', 'Chinese', 'TextButUnknown'], script_proba.astype(np.float64) ) :
        output_lut['Pr({})'.format( sid ) ] = proba
    if ( output_dir is None ) :
        save_mode = 'buffer'
        verbose_print( verbose, "INFO: begin save detection results, image mode = JPEG buffer")
    elif ( os.path.isdir( output_dir ) ) :
        save_mode = 'disk'
        verbose_print( verbose, "INFO: begin save detection results, image mode = image file path")
    else :
        save_mode = None
        verbose_print( verbose, "INFO: begin save detection results, image mode = skip dump images")
        verbose_print( verbose+1,"WARNING: only text bboxes are save, but NOT images.")
    bboxes = reject_text_regions( decoded_results,
                                  prefix=prefix,
                                  save_mode=save_mode,
                                  output_dir=output_dir,
                                  proba_threshold=proba_threshold,
                                  contrast_threshold=contrast_threshold,
                                  lh_threshold=lh_threshold,
                                  verbose=verbose,
                                  n_jobs=n_jobs)
    verbose_print( verbose, "INFO: done text detection for", file_path )
    output_lut['bboxes'] = bboxes
    if ( not return_proba ) :
        return output_lut
    else :
        return output_lut, (text_proba, script_proba)

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
        return np.dstack( new_ch_list )


def script_proba_minmax( script_proba_all ) :
    idx = np.argmin( script_proba_all[:,1], axis = 0 )
    return script_proba_all[idx]


def text_proba_minmax( text_proba_all ) :
    proba_max = np.max( text_proba_all, axis = 0 )
    proba_min = np.min( text_proba_all, axis = 0 )
    border, text = proba_max[...,1],proba_max[...,2]
    nontext = proba_min[...,0]
    text_proba = np.dstack([nontext, border*2, text ])
    return text_proba/np.sum(text_proba,axis=-1,keepdims=True)

def lazy_decoder( file_path,
                  textDet_model,
                  scriptID_model=None,
                  num_resolutions=5,
                  output_dir=None,
                  proba_threshold=.33,
                  lh_threshold=8,
                  contrast_threshold=32.,
                  return_proba=True,
                  n_jobs=1,
                  verbose=2) :
    """
    INPUTS:
        ------------------------------------------------------------------------
        | Mandentory Parameters
        ------------------------------------------------------------------------
        file_path = str, path to a local file or URL to a web image
        textDet_model = keras model, pretrained text detection model
        scriptID_model = None or keras model, pretrained script ID model
                         if None, then no scriptID classification
        output_dir = None or str or 'SKIP', dir to save detected and corrected text regions,
                     if None, then text regions as JPEG buffers
                     if SKIP, then not save text regions
        ------------------------------------------------------------------------
        | Simple Rules to Reject A Text Region
        ------------------------------------------------------------------------
        proba_threshold = float in (0,1), the minimum text probability to accept a text region
        lh_threshold = float, the minimum line height to accept a text region
        contrast_threshold = float in (0,255), the minimum intensity standard deviation to accept a text region
        ------------------------------------------------------------------------
        | Others
        ------------------------------------------------------------------------
        return_proba = bool, if true, return the raw outputs of both models
        n_jobs = int, if greater than 1, use multiple CPUs
        num_resolutions = int, default 3
        verbose = bool, if true, print out state messages

    OUTPUTS:
        output_lut = dict, containing all decoded results including
                     'filename' -> input image file
                     'resize'   -> resize factor for text detection analysis
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
    """
    verbose_print( verbose, "INFO: begin text detection for {}".format( file_path ) )
    # 1. read image
    img = read_image( file_path )
    # 1.b resize image if necessary
    ih, iw = img.shape[:2]
    verbose_print( verbose, "INFO: original image size = ({},{})".format( ih, iw ) )
    # 1.c estimate resize factor
    resize_factor_list = np.linspace( MINSIDE/min(ih,iw), min(6,MAXSIDE/max(ih,iw)), num_resolutions )
    if ( np.min( np.abs( resize_factor_list - 1 ) > .05 ) ) :
        resize_factor_list = resize_factor_list.tolist() + [1]
    else :
        resize_factor_list = resize_factor_list.tolist()
    text_proba_list, script_proba_list = [], []
    for coef, resize_factor in enumerate( resize_factor_list ):
        if ( abs(resize_factor-1)> 0.05 ) :
            if (resize_factor<1) :
                method = cv2.INTER_AREA
            else :
                method = cv2.INTER_CUBIC
            nh, nw = int(ih*resize_factor), int(iw*resize_factor)
            rimg = cv2.resize( img, (nw, nh), interpolation=method )
        else :
            resize_factor = 1
            nh, nw = ih, iw
            rimg = np.array(img, dtype='uint8')
        verbose_print( verbose, "INFO: begin text detection for factor", coef)
        verbose_print( verbose, "INFO: resize input by {:.2f} to match line height {}".format( resize_factor, (nh, nw) ))
        if ( max(nh, nw) > MAXSIDE ) :
            break
        elif ( min(nh, nw) < MINSIDE ) :
            continue
        else :
            pass
        # 2. convert input image to network tensor
        x = convert_imageArray_to_inputTensor( rimg )
        # 3. predict text probability map
        text_proba  = textDet_model.predict(x)
        text_proba = text_proba[0,:nh,:nw] # since we always take one sample at a time
        # 3.b resize back to original size
        if ( np.abs( resize_factor-1 ) < 1e-2 ):
            pass
        elif ( resize_factor > 1 ) :
            # shrink proba
            text_proba = cv2.resize( text_proba, (iw, ih), interpolation=cv2.INTER_AREA )
        else :
            # enlarge proba
            text_proba = cv2.resize( text_proba, (iw, ih), interpolation=cv2.INTER_CUBIC )
        if ( scriptID_model is not None ) :
            script_proba = scriptID_model.predict(x)[0]
        else :
            script_proba = None
        verbose_print( verbose, "INFO: done text detection for factor", coef)
        # 3. update                             
        text_proba_list.append( np.expand_dims( text_proba, axis=0 ) )
        script_proba_list.append( script_proba )
    # 4. fuse results
    text_proba_all = np.concatenate( text_proba_list, axis=0 )
    script_proba_all = np.row_stack( script_proba_list )
    # 6. minimax decode
    verbose_print( verbose, "INFO: begin minimax decoding")
    text_proba = text_proba_minmax( text_proba_all )
    script_proba = script_proba_minmax( script_proba_all )
    scriptID = decode_scriptID( script_proba )   
    # 4. decode individual text bbox
    membership = text_proba.argmax( axis = -1 )
    text_mask = (membership==2).astype('uint8')
    num_regs,labels,reg_stats,centroids = cv2.connectedComponentsWithStats( text_mask, 8, cv2.CV_32S )
    verbose_print( verbose, "INFO: localize rotated text bounding boxes" )
    decoded_results = decode_rotated_text_bbox( text_proba[...,-1], labels, img, reg_stats, n_jobs=n_jobs )
    # 5. save outputs
    prefix = '{}'.format( md5( np.ascontiguousarray(img) ).hexdigest() )
    output_lut = { 'filename' : file_path,
                   'resize' : np.float64(1.),
                   'md5' : prefix,}
    for sid, proba in zip( [ 'NonText', 'Latin', 'Hebrew', 'Cyrillic', 'Arabic', 'Chinese', 'TextButUnknown'], script_proba.astype(np.float64) ) :
        output_lut['Pr({})'.format( sid ) ] = proba
    if ( output_dir is None ) :
        save_mode = 'buffer'
        verbose_print( verbose, "INFO: begin save detection results, image mode = JPEG buffer")
    elif ( os.path.isdir( output_dir ) ) :
        save_mode = 'disk'
        verbose_print( verbose, "INFO: begin save detection results, image mode = image file path")
    else :
        save_mode = None
        verbose_print( verbose, "INFO: begin save detection results, image mode = skip dump images")
        verbose_print( verbose+1,"WARNING: only text bboxes are save, but NOT images.")
    bboxes = reject_text_regions( decoded_results,
                                  prefix=prefix,
                                  save_mode=save_mode,
                                  output_dir=output_dir,
                                  proba_threshold=proba_threshold,
                                  contrast_threshold=contrast_threshold,
                                  lh_threshold=lh_threshold,
                                  verbose=verbose,
                                  n_jobs=n_jobs)
    verbose_print( verbose, "INFO: done text detection for", file_path )
    output_lut['bboxes'] = bboxes
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
