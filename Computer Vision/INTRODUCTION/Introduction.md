# Introduction

## Object Detection

Object detection is a computer vision technique that allows us to identify and locate objects in an image or video. With this kind of identification and localization, object detection can be used to count objects in a scene and determine and track their precise locations, all while accurately labeling them.

## Use of Deep Learning to Solve This Task

The simplest deep learning approach, and a widely used one, for detecting objects in images – Convolutional Neural Networks or CNNs. If your understanding of CNNs is a little rusty, summarize the inner workings of a CNN for you. Take a look at the below summarize the working of CNN.

![1.png][images/1.png]

We pass an image to the network, and it is then sent through various convolutions and pooling layers. Finally, we get the output in the form of the object’s class.For each input image, we get a corresponding class as an output.


## Region-Based Convolutional Neural Network(R-CNN)

RCNN uses selective search to extract these boxes from an image (these boxes are called regions).The main idea is composed of two steps. First, using selective search, it identifies a manageable number of bounding-box object region candidates (“region of interest” or “RoI”). And then it extracts CNN features from each region independently for classification.

### Network Design

![2.bmp](attachment:2.bmp)

#### Selective Search:

1. Generate initial sub-segmentation, we generate many candidate regions
2. Use greedy algorithm to recursively combine similar regions into larger ones 
3. Use the generated regions to produce the final candidate region proposals 

### Workflow

How R-CNN works can be summarized as follows:

1. Pre-train a CNN network on image classification tasks; for example, VGG or ResNet trained on ImageNet dataset. The classification task involves N classes.
2. Propose category-independent regions of interest by selective search (~2k candidates per image). Those regions may contain target objects and they are of different sizes.
3. Region candidates are warped to have a fixed size as required by CNN.
4. Continue fine-tuning the CNN on warped proposal regions for K + 1 classes; The additional one class refers to the background (no object of interest). In the fine-tuning stage, we should use a much smaller learning rate and the mini-batch oversamples the positive cases because most proposed regions are just background.
5. Given every image region, one forward propagation through the CNN generates a feature vector. This feature vector is then consumed by a binary SVM trained for each class independently.
   The positive samples are proposed regions with IoU (intersection over union) overlap threshold >= 0.3, and negative samples are irrelevant others.
6. To reduce the localization errors, a regression model is trained to correct the predicted detection window on bounding box correction offset using CNN features.

## Loss Function

#### Bounding Box Regression

![5.bmp](attachment:5.bmp)

![3.bmp](attachment:3.bmp)

![4.bmp](attachment:4.bmp)

### Advantages and DisAdvantages

* It takes a huge amount of time to train the network as you would have to classify 2000 region proposals per image.
* It cannot be implemented real time as it takes around 47 seconds for each test image.
* The selective search algorithm is a fixed algorithm. Therefore, no learning is happening at that stage. This could lead to the generation of bad candidate region proposals.
* High computation time as each region is passed to the CNN separately also it uses three different model for making predictions.

## FAST R-CNN

Fast RCNN, we feed the input image to the CNN, which in turn generates the convolutional feature maps. Using these maps, the regions of proposals are extracted. We then use a RoI pooling layer to reshape all the proposed regions into a fixed size, so that it can be fed into a fully connected network.To make R-CNN faster improved the training procedure by unifying three independent models into one jointly trained framework and increasing shared computation results, named Fast R-CNN. Instead of extracting CNN feature vectors independently for each region proposal, this model aggregates them into one CNN forward pass over the entire image and the region proposals share this feature matrix. Then the same feature matrix is branched out to be used for learning the object classifier and the bounding-box regressor. In conclusion, computation sharing speeds up R-CNN.
The reason “Fast R-CNN” is faster than R-CNN is because you don’t have to feed 2000 region proposals to the convolutional neural network every time. Instead, the convolution operation is done only once per image and a feature map is generated from it.


### Network Design

![6.bmp](attachment:6.bmp)

###  Workflow

How Fast R-CNN works is summarized as follows; many steps are same as in R-CNN:

1.First, pre-train a convolutional neural network on image classification tasks.
2.Propose regions by selective search (~2k candidates per image).
3.Alter the pre-trained CNN:
   * Replace the last max pooling layer of the pre-trained CNN with a RoI pooling layer. The RoI pooling layer outputs fixed-length feature vectors of region proposals. Sharing the CNN computation makes a lot of sense, as many region proposals of the same images are highly overlapped.
   * Replace the last fully connected layer and the last softmax layer (K classes) with a fully connected layer and softmax over K + 1 classes.
4.Finally the model branches into two output layers:
   * A softmax estimator of K + 1 classes (same as in R-CNN, +1 is the “background” class), outputting a discrete probability distribution per RoI.
   * A bounding-box regression model which predicts offsets relative to the original RoI for each of K classes.

![7.bmp](attachment:7.bmp)

![8.bmp](attachment:8.bmp)

The image is provided as an input to a convolutional network which provides a convolutional feature map. Instead of using selective search algorithm on the feature map to identify the region proposals, a separate network is used to predict the region proposals. The predicted region proposals are then reshaped using a RoI pooling layer which is then used to classify the image within the proposed region and predict the offset values for the bounding boxes.

## Loss Function

![9.bmp](attachment:9.bmp)

### Advantages and DisAdvantages


1.Fast R-CNN is much faster in both training and testing time. However, the improvement is not dramatic because the region proposals are generated separately by another model and that is very expensive.

2.Fast RCNN has certain problem areas. It also uses selective search as a proposal method to find the Regions of Interest, which is a slow and time consuming process. It takes around 2 seconds per image to detect objects, which is much better compared to RCNN. But when we consider large real-life datasets, then even a Fast RCNN doesn’t look so fast anymore.

3.Selective search is slow and hence computation time is still high.


## YOLO-You Only Look Once

All of the previous object detection algorithms use regions to localize the object within the image. The network does not look at the complete image. Instead, parts of the image which have high probabilities of containing the object. YOLO or You Only Look Once is an object detection algorithm much different from the region based algorithms seen above. In YOLO a single convolutional network predicts the bounding boxes and the class probabilities for these boxes.

### Network Design


![10.bmp](attachment:10.bmp)

How YOLO works is that we take an image and split it into an SxS grid, within each of the grid we take m bounding boxes. For each of the bounding box, the network outputs a class probability and offset values for the bounding box. The bounding boxes having the class probability above a threshold value is selected and used to locate the object within the image.

YOLO is orders of magnitude faster(45 frames per second) than other object detection algorithms. The limitation of YOLO algorithm is that it struggles with small objects within the image, for example it might have difficulties in detecting a flock of birds. This is due to the spatial constraints of the algorithm.

 ### Work Flow

The whole system can be divided into two major components: Feature Extractor and Detector; both are multi-scale. When a new image comes in, it goes through the feature extractor first so that we can obtain feature embeddings at three (or more) different scales. Then, these features are feed into three (or more) branches of the detector to get bounding boxes and class information.

![12.bmp](attachment:12.bmp)

Let us consider an example below, where the input image is 416 x 416, and stride of the network is 32. As pointed earlier, the dimensions of the feature map will be 13 x 13. We then divide the input image into 13 x 13 cells.

![13.bmp](attachment:13.bmp)

Then, the cell (on the input image) containing the center of the ground truth box of an object is chosen to be the one responsible for predicting the object. In the image, it is the cell which marked red, which contains the center of the ground truth box (marked yellow).

Now, the red cell is the 7th cell in the 7th row on the grid. We now assign the 7th cell in the 7th row on the feature map (corresponding cell on the feature map) as the one responsible for detecting the dog.

Now, this cell can predict three bounding boxes. Which one will be assigned to the dog's ground truth label? In order to understand that, we must wrap out head around the concept of anchors.

Note that the cell we're talking about here is a cell on the prediction feature map. We divide the input image into a grid just to determine which cell of the prediction feature map is responsible for prediction

### LOSS FUNCTION

The function is a composition of multiple SSEs. During training, this loss function is optimized to improve the predictions of the network. SSE has a benefit over other loss functions as it is easier to use and optimize. It is noteworthy that usually loss functions are chosen or designed keeping the ease of optimization in mind. For example, a cross entropy loss function is a negative logarithmic function which is smooth and convex. Both these properties make it easier and quicker to optimize hence, improving training time and results. Following is the formula for the SSE function used by YOLO:

![11.bmp](attachment:11.bmp)

### Advantages and Disadvantages

1.The model imposes strong spatial constraints on the bounding box predictions. This is so because the model uses a fully connectd layer to regress the bounding box. As mentioned in YOLO v1: Part2, each layer can only predict 1 class object. The grid cells predict two bounding boxes but a grid cell can only have 1 object in it. The presence of multiple objects results in shared space amongst bounding boxes with the different objects. This overlap causes confusion for the fully connected layer.
Thus, this limits the number of nearby objects the model can predict.

2.The model samples down the input image to an SxS grid where every grid cell is responsible for making bounding box predictions. Thus, due to the downsampling the model uses rather coarse features to predict the bounding boxes.

3.It finds it difficult to localize small objects or groups of small objects. Hence, the main source of errors is localization.
