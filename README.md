# Hands AI

Recognizing hands and point direction in images.

## Usage

- Requires Python>3.6, OpenCV2>3.4, Torch>1, Fastai lib
- Download pretrained models from http://www.ollihuotari.com/data/hands_ai/ and put them in data/models directory
- see live predictions: `python live_predict.py --help`
  - e.g.
    - `python live_predict.py` - tries to use opencv camera input with defaults
    - `python live_predict.py -f example.avi` - using video file
    - `python live_predict.py -i gstreamer -m 'ModelDarknetCustomized.load_default_03_320()' --cap_args 3 640 4 480  --img_size 320 416 --display-size 2 --display-size-screen 0.5` - using gstreamer input with custom image sizes. Good when predicting in Jetson tx2
    - `python live_predict.py --display-size 2 -m 'ModelDarknetCustomized.load_default_full_512()'`- using bigger model
    - live predict also can use [phue-lib](https://github.com/studioimaginaire/phue) to control lights (example video below)

## Examples

**Using to control lights** (Deployed to Jetson tx2 with infrared video camera)

![Hue stuff](https://uc2d29f43de42ca7a41b4f66a48f.previews.dropboxusercontent.com/p/orig/AAQaqFTi336ag9zVpWGhPaWU6yP9HyBfV9VzUN75h6DXQgi3kB4RHilBgIzPmCkgCaEZJGzfcjgpjZ_T3sNfBgEbgt9i9WNWBAfcFyeHWylqaR2BI0Fo33njsnEk1LYXVi2jRo0d38VHuix4Hun1EuFz0oESh7sZ8q7NQFKxW7WAmgRhIQuZy-AwzfrhAFcp1yiobfOnxBwdad70m8MSkb5s/p.gif?size=416x320&size_mode=3)

**Classes it recognizes**

...

## Pretrained models and data

#### Models

- See notebooks at ... to see how they were trained and how effetive they are
  - ... Bigger network
  - ... Smaller - Can be run in Jetson tx2 in realtime
- Basically model is cnn which outputs (x, y, &#238;, &#309;, objectness, p_class_1, p_class_2, ...).
  - much like [Yolo](https://pjreddie.com/darknet/yolo/) but with angle instead of bounding boxes.

#### Data

Data not available currently. Maybe some day. I'll have to clean it (I don't want to explain my nephew's mother why there is a video of her son pointing at things in the Internet.)

- Data labeling related tools are at [label_tools](https://github.com/holli/hands_ai/tree/master/label_tools) folder.
- See notebook at ... for some examples
- Basically I took some videos. Run them through hands_ai and (chainer)[https://github.com/DeNA/Chainer_Realtime_Multi-Person_Pose_Estimation] to get some pre made points. Go through video frame by frame. Opencv tracking to track points between frames. Then at each frame choose either one of the models, tracked points or manually clicking.







