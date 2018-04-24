# TextDetectionWithScriptID
Fully Convolutional Neural Network for Text Detection in Scene/Document with Script ID supports.

## 1. Overview
This repo contains two types of deep neural networks, 
- **pixel-level Text Detection Classification Network (TDCN)** with specializations in 
  - **Word-Level** `SceneText`, e.g. a street-view image.
  - **Line-Level** `DocumentText`, e.g. a scanned letter.
  
  It classifies each pixel in an input image into one for the following three categories:
  
  |**class index**| **class name** | **color channel**| **Description**|
  |:-------------:|:-------------:|:-------------:|:-------------:|
  | 0 | NonText | Red | Any non-text content |
  | 1 | Border  | Green | Pixels on text borders |
  | 2 | Text    | Blue | Text Pixels |
  
- **page-level Script ID Classification Network (SICN)**

  It classifies an input image into one of the following categories: 
  
  | **scriptID index** | **scriptID name** | **Country** |
  |:------------------:|:-----------------:|:-----------:|
  | 0 | NonText | N/A |
  | 1 | Latin | US, UK, etc. |
  | 2 | Hebrew | Israel |
  | 3 | Cyrillic | Russia, Ukraine, etc. | 
  | 4 | Arabic | Iran, Saudi Arabia, etc. |
  | 5 | Chinese | China, HongKong, etc. |
  | 6 | TextButUnknown | N/A |

where TDCN models are mainly based on the ICCV17 paper [**Self-organized Text Detection with Minimal Post-processing via Border Learning**](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Self-Organized_Text_Detection_ICCV_2017_paper.pdf).

All models are trained with the `Keras` deep nerual network library with the `TensorFlow` backend.

## 2. Structure
This repo contains the following data
- `data`: sample image data for testing.
- `lib` : core model definitions and util functions.
- `model` : pretrained model weights.
  - `textDetSceneModel.h5`: pretrained TDCN weight for scene text detection
  - `textDetDocumModel.h5`: pretrained TDCN weight for document text detection
  - `scriptIDModel.h5`: pretrained SICN weight
- `notebook` : Python2 demo notebook.
- `bin` : Python2/3 command-line tool.

## 3. Dependency
This repo depends on the core deep learning libraries
- `Keras`: >=2.0.7
- `TensorFlow`: >=1.1.0

and image processing libraries
- `OpenCV-Python`: >=3.1.0
- `Skimage`: >= 0.13.0

## 4. Usage
### 4.1 Basic Models
```python
# 1. TDCN
textDet_model = textDetCore.create_textDet_model()
textDet_model.load_weights( textDet_weight )
# 2. SICN
scriptID_model = textDetCore.create_scriptID_model()
scriptID_model.load_weights( scriptID_weight )
```

### 4.2 Simple Decoder
Simple decoder is written w.r.t. the use case of text detection for a document image or a text input image where text regions can be expressed as a set of rectangular bounding boxes.
```python
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
                               'jpgbuf'   -> jpeg buffer for the text region image, when output_dir is None
            proba_map = ( text_proba, script_proba )
                        - text_proba, i.e. a text probability map, size of imgHeight-by-imgWidth-by-3
                        - script proba, i.e. a script ID probability map, size of 1-by-7                                   
    """
```


## 4.3 Lazy (but slow) Deocder
This is a more complicated decoder for scene text detection. This extra complexity is due to the fact that the dominant fontsize assumption may no longer hold for a scene text image, and we can't guess the actual fontsize height based on the input image size. We, therefore, analyze a given scene text image at a number of resolution scales to capature text regions of very different fontsizes.

```python
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
                           'jpgbuf'   -> jpeg bufferfor the text region image, when output_dir is None
        proba_map = ( text_proba, script_proba )
                    - text_proba, i.e. a text probability map, size of imgHeight-by-imgWidth-by-3
                    - script proba, i.e. a script ID probability map, size of 1-by-7
    """                                            
```

## 4.4 Command-Line Tool
Finally, we also provide a command-line tool for text detection. 

```shell
python bin/textDetection.py -h
usage: textDetection.py [-h] [-i INPUT_FILES] [-o OUTPUT_DIR]
                        [-t {full,textDet,postProc}]
                        [-m {doc0,doc1,scene,custom}] [-v VERBOSE]
                        [-tl TH_LINEHEIGHT] [-tp TH_TEXTPROB]
                        [-tc TH_CONTRAST] [-mt {line,word}]
                        [-dt {simple,lazy}] [-nj N_JOBS] [-nr N_RES]
                        [--domFont DOM_FONT] [--darkText] [--rotText]
                        [--jpegBuffer] [--version]

Text Detection with ScriptID supports

optional arguments:
  -h, --help            show this help message and exit

required arguments:
  -i INPUT_FILES, --inputFile INPUT_FILES
                        input test image files or cached json files
  -o OUTPUT_DIR, --outputDir OUTPUT_DIR
                        output detection dir (./)
  -t {full,textDet,postProc}, --task {full,textDet,postProc}
                        tasks in {full, textDet, postProc}

optional arguments:
  -m {doc0,doc1,scene,custom}, --mode {doc0,doc1,scene,custom}
                        working mode in {doc0(black and horizontal text),
                        doc1(black text), scene, custom}
  -v VERBOSE            verbosity level (0), higher means more print outs
  -tl TH_LINEHEIGHT, --threshLineHeight TH_LINEHEIGHT
                        minimal text region height (15)
  -tp TH_TEXTPROB, --threshTextProba TH_TEXTPROB
                        minimal text region probability (.5)
  -tc TH_CONTRAST, --threshContrast TH_CONTRAST
                        minimal text region contrast (16)
  -mt {line,word}, --modelType {line,word}
                        detector type in {line, word}
  -dt {simple,lazy}, --decoderType {simple,lazy}
                        decoder type in {simple, lazy}
  -nj N_JOBS, --nJobs N_JOBS
                        number of parallel cpu jobs
  -nr N_RES, --nRes N_RES
                        number of analysis resultions
  --domFont DOM_FONT    dominant text height in pixels
  --darkText            whether or not texts are darker
  --rotText             whether or not texts are rotated
  --jpegBuffer          whether or not save image inside json
  --version             show program's version number and exit
```
## 4.4 Demo Code
You may find ipython2 notebook under `notebook`. Alternatively, you are welcome to use our provided google colab notebooks (open and clone your own)
- [python2 notebook](https://drive.google.com/file/d/1uhcA6nYZoBcHmEdnJUHLp4WeXsPN6OIB/view?usp=sharing)
- [python3 notebook](https://drive.google.com/file/d/13TRD6yeBtCo2IP5tJklvwpJcy91jXyJD/view?usp=sharing)

# 5. Citation and Contact
If you use this repo for academic purposes, please cite the following paper.

```latex
@INPROCEEDINGS{WuICCV2017, 
author={Y. Wu and P. Natarajan}, 
booktitle={2017 IEEE International Conference on Computer Vision (ICCV)}, 
title={Self-Organized Text Detection with Minimal Post-processing via Border Learning}, 
year={2017}, 
volume={}, 
number={}, 
pages={5010-5019}, doi={10.1109/ICCV.2017.535}, 
ISSN={}, 
month={Oct},}
```

For questions, please contact Dr. Yue Wu (`yue_wu@isi.edu`).
