#! /usr/bin/env python
"""
This script wraps two types text detection models

1. Text detection model using the "Self-Organized Text Detection" method
2. ScriptID detection

Created on 04/22/2018
Contact: Dr. Yue Wu
Email: yue_wu@isi.edu
"""

from __future__ import print_function
import sys
import os
import argparse
import json
sys.path.insert(0,'/nfs/isicvlnas01/share/opencv-3.1.0/lib/python2.7/site-packages/')

# 1. set path
bin_path = os.path.realpath(__file__)
repo_root = os.path.join( os.path.dirname( bin_path ), os.path.pardir )
assert os.path.isdir( repo_root ), "ERROR: can't locate git repo for text detection"
model_dir = os.path.join( repo_root, 'model' )
scriptID_weight = os.path.join( model_dir, 'sciptIDModel.h5' )
assert os.path.isfile( scriptID_weight ), "ERROR: can't locate script-ID classification model"
docum_weight  = os.path.join( model_dir, 'textDetDocumModel.h5' )
assert os.path.isfile( docum_weight ), "ERROR: can't locate text detection model"
scene_weight  = os.path.join( model_dir, 'textDetSceneModel.h5' )
assert os.path.isfile( scene_weight ), "ERROR: can't locate text detection model"
data_dir  = os.path.join( repo_root, 'data' )
lib_dir   = os.path.join( repo_root, 'lib' )
sys.path.insert( 0, lib_dir )

known_image_ext = ['jpg','jpeg','png','tif','tiff','bmp'] # append more if neceesary

def check_valid_image_file( file_path ) :
    if ( 'http' in file_path ) :
        # this is a url image
        return True
    else :
        ext = file_path.rsplit('.')[-1].lower()
        return ext in known_image_ext

def read_image_file_list( file_path ) :
    with open( file_path ) as IN :
        lines = [ line.strip() for line in IN.readlines() ]
    valid_files = []
    for this_file in lines :
        is_valid = check_valid_image_file( this_file )
        if ( is_valid ) :
            valid_files.append( this_file )
    return valid_files

def unify_inputs( input_files, verbose=0 ) :
    unified_input_files = []
    for this_file in input_files :
        if ( check_valid_image_file( this_file ) ) :
            # this is a url link
            unified_input_files.append( this_file )
        else :
            # this is something not a url nor image file
            # try to interpret it as a file list
            print_warn = False
            try :
                list_files = read_image_file_list( this_file )
                if ( len( list_files )>0 ) :
                    unified_input_files += list_files
                else :
                    print_warn = True
            except :
                print_warn = True
            if ( print_warn ) :
                verbose_print( verbose, "WARNING: fail to interpreted input{} as an image file or a list of image files".format( this_file ) )
    return unified_input_files


if __name__ == '__main__' :
    '''Text Detection with ScriptID supports
    Usage:
        textDetection.py -h
    '''
    parser = argparse.ArgumentParser( description = 'Text Detection with ScriptID supports' )
    parser.add_argument( '-i', action = 'append', dest = 'input_files', default = [], help = 'input test image files' )
    parser.add_argument( '-o', action = 'store', dest = 'output_dir', default = './', help = 'output detection dir (./)' )
    parser.add_argument( '-v', action = 'store', dest = 'verbose', type = int, default = 0, help = 'verbosity level (0), higher means more print outs')
    parser.add_argument( '-tl', '--threshLineHeight', action = 'store', dest = 'th_lineHeight', type = int, default = 15, help = 'minimal text region height (15)' )
    parser.add_argument( '-tp', '--threshTextProba', action = 'store', dest = 'th_textProb', type = float, default = .5, help = 'minimal text region probability (.5)' )
    parser.add_argument( '-tc', '--threshContrast', action = 'store', dest = 'th_contrast', type = float, default = 16., help = 'minimal text region contrast (16)' )
    parser.add_argument( '-m', '--mode', action = 'store', dest = 'mode', type = str, choices=['doc0', 'doc1','scene','custom'], default = 'docum', help = 'working mode in {doc0(black and horizontal text), doc1(black text), scene, custom}' )
    parser.add_argument( '-mt', '--modelType', action = 'store', dest = 'model_type', type = str, choices=['line','word'], default = 'line', help = 'detector type in {line, word}' )
    parser.add_argument( '-dt', '--decType', action = 'store', dest = 'dec_type', type = str, choices=['simple','lazy'], default = 'simple', help = 'decoder type in {simple, lazy}' )
    parser.add_argument( '-nj', '--nJobs', action = 'store', dest = 'n_jobs', type = int, default = 1, help = 'number of parallel cpu jobs' )
    parser.add_argument( '-nr', '--nRes', action = 'store', dest = 'n_res', type = int, default = 5, help = 'number of analysis resultions' )
    parser.add_argument( '--domFont', action='store', dest = 'dom_font', type=int, default=None, help= 'dominant text height in pixels' )
    parser.add_argument( '--darkText', action='store_true', dest = 'dark_text', default=None, help= 'whether or not texts are darker' )
    parser.add_argument( '--rotText', action='store_true', dest = 'rotated_text', default=None, help= 'whether or not texts are rotated' )
    parser.add_argument( '--jpegBuffer', action='store_true', dest = 'jpeg_buffer', default=False, help= 'whether or not save image inside json' )
    parser.add_argument( '--version', action = 'version', version = '%(prog)s 1.0' )
    #----------------------------------------------------------------------------
    # parse program arguments
    #----------------------------------------------------------------------------
    args = parser.parse_args()
    verbose = args.verbose
    input_files = args.input_files
    output_dir = args.output_dir
    th_lineHeight = args.th_lineHeight
    th_textProb = args.th_textProb
    th_contrast = args.th_contrast
    model_type = args.model_type
    dec_type = args.dec_type
    mode = args.mode
    n_jobs = args.n_jobs
    dom_font = args.dom_font
    dark_text = args.dark_text
    rotated_text = args.rotated_text or True
    jpeg_buffer = args.jpeg_buffer
    num_resolutions = args.n_res
    # overwrite parameters is mode is specified
    if ( mode != 'custom' ) :
        if ( 'doc' in mode ) :
            model_type = 'line'
            dec_type = 'simple'
            if ( mode == 'doc0' ) :
                dark_text = True
                rotated_text = False
            elif ( mode == 'doc1' ) :
                dark_text = True
                rotated_text = True
        else :
            model_type = 'word'
            dec_type = 'lazy'
    # load text detection modules
    import textDetCore
    import textDetDec
    verbose_print = textDetDec.verbose_print
    verbose_print( verbose, "INFO: begin text detection with scriptID supports")
    #----------------------------------------------------------------------------
    # parse input image files and check output dirs
    #----------------------------------------------------------------------------
    input_files = unify_inputs( input_files, verbose )
    verbose_print( verbose, "INFO: in total load {} image files".format( len(input_files) ) )
    if not os.path.isdir( output_dir ) :
        try :
            os.makedirs( output_dir )
        except :
            pass
        if ( not os.path.isdir( output_dir ) ) :
            verbose( verbose+1, "ERROR: cannot access output dir", output_dir )
            exit(-1)
    #----------------------------------------------------------------------------
    # prepare models
    #----------------------------------------------------------------------------
    scriptID_model = textDetCore.create_scriptID_model()
    scriptID_model.load_weights( scriptID_weight )
    textDet_model = textDetCore.create_textDet_model()
    if ( model_type == 'line' ) :
        textDet_model.load_weights( docum_weight )
    else :
        textDet_model.load_weights( scene_weight )
    verbose_print( verbose, "INFO: successfully initialize text detection models" )
    #----------------------------------------------------------------------------
    # prepare the text decoder
    #----------------------------------------------------------------------------
    if ( dec_type == 'simple' ) :
        verbose_print( verbose, "INFO: use SIMPLE DECODER (single resolution) with parameter settings:")
        verbose_print( verbose, " " *5, "output_dir =", output_dir )
        verbose_print( verbose, " " *5, "use_jpeg_buffer =", jpeg_buffer )
        verbose_print( verbose, " " *5, "dom_font =", dom_font )
        verbose_print( verbose, " " *5, "dark_text =", dark_text )
        verbose_print( verbose, " " *5, "rotated_text =", rotated_text )
        verbose_print( verbose, " " *5, "proba_threshold =", th_textProb )
        verbose_print( verbose, " " *5, "lh_threshold =", th_lineHeight )
        verbose_print( verbose, " " *5, "contrast_threshold =", th_contrast )
        verbose_print( verbose, " " *5, "n_jobs =", n_jobs )
        def text_decoder( file_path ) :
            return textDetDec.simple_decoder( file_path,
                                              textDet_model,
                                              scriptID_model=scriptID_model,
                                              output_dir=None if jpeg_buffer else output_dir,
                                              dom_font=dom_font,
                                              dark_text=dark_text,
                                              rotated_text=rotated_text,
                                              proba_threshold=th_textProb,
                                              lh_threshold=th_lineHeight,
                                              contrast_threshold=th_contrast,
                                              return_proba=False,
                                              n_jobs=n_jobs,
                                              verbose=verbose)
    elif ( dec_type == 'lazy' ) :
        verbose_print( verbose, "INFO: use LAZY DECODER (multi-resolution) with parameter settings:")
        verbose_print( verbose, " " *5, "output_dir =", output_dir )
        verbose_print( verbose, " " *5, "use_jpeg_buffer =", jpeg_buffer )
        verbose_print( verbose, " " *5, "num_resolutions =", num_resolutions )
        verbose_print( verbose, " " *5, "proba_threshold =", th_textProb )
        verbose_print( verbose, " " *5, "lh_threshold =", th_lineHeight )
        verbose_print( verbose, " " *5, "contrast_threshold =", th_contrast )
        verbose_print( verbose, " " *5, "n_jobs =", n_jobs )
        def text_decoder( file_path ) :
            return textDetDec.lazy_decoder( file_path,
                                            textDet_model=textDet_model,
                                            scriptID_model=scriptID_model,
                                            num_resolutions=num_resolutions,
                                            output_dir=None if jpeg_buffer else output_dir,
                                            proba_threshold=th_textProb,
                                            lh_threshold=th_lineHeight,
                                            contrast_threshold=th_contrast,
                                            return_proba=False,
                                            n_jobs=n_jobs,
                                            verbose=verbose)
    else :
        verbose_print( verbose+1, "ERROR: the decoder type {} is UNKNOWN".format( dec_type ) )
    #----------------------------------------------------------------------------
    # main loop
    #----------------------------------------------------------------------------
    output_summary = os.path.join( output_dir, 'results.csv' ) 
    with open( output_summary, 'w' ) as OUT :
        OUT.write('')
    verbose_print( verbose, "-" * 100 )
    for idx, file_path in enumerate( input_files ) :
        success = False
        output_jf = '-1'
        try :
            output_lut = text_decoder( file_path )
            output_jf = os.path.join( output_dir, output_lut['md5'] + '.json' )
            json.dump( output_lut, open( output_jf, 'w' ), indent=4, sort_keys=True )
            verbose_print( verbose, "INFO: successfully dump decoded results to file", output_jf )
            success = True
        except Exception as e:
            verbose_print( verbose, "WARNING: fail to process sample", file_path, e )
        verbose_print( verbose, "-" * 100 )
        if ( success ) :
            with open( output_summary, 'a+' ) as OUT :
                OUT.write( ','.join( [file_path, output_jf]) + '\n' )
    verbose_print( verbose+1, "INFO: successfully dump all decoded results to file", output_summary )
                
