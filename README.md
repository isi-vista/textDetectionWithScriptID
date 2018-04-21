# TextDetWithScriptID
Fully Convolutional Neural Network for Text Detection in Scene/Document with Script ID supports.

## 1. Overview
This repo contains two types of deep neural networks, 
- **pixel-level Text Detection Classification Network (TDCN)** with specializations in 
  - `SceneText`, e.g. a street-view image.
  - `DocumentText`, e.g. a scanned letter.
  
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
### 34.1 Basic Models
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
det_results = simple_decoder( 
                file_path, # image url or disk file path 
                textDet_model, # TDCN model with pretrained weights
                scriptID_model, # SICN model with pretrained weights 
                resize_factor = 2., # resize factor for text analysis
                proba_threshold = .33, # text proba threshold to reject low confidence regions
                area_threshold = 100., # area threshold to reject small regions
                visualize = True, # pyplot text detection results 
                )
```



