**Object Localization**

-Shriyashish Mishra

This is a summary compilation of Object Localization using Convolutional Neural Network.

Object Localization in layman terms means locating what we&#39;re looking for by detecting all the objects and background in the image.

Summary of steps that we follow:

1. **Convolution:**

To the input image, we apply feature detectors/Kernel/filters that allow to delete all the unnecessary details and give a proper feature map that provides all the important features to be classified and/or modified. This is how it&#39;s done:

We move up and down the input image and feature detector is applied on each 3\*3 matrix that we obtain giving us a net result as a Feature Map.The movement of the feature detector on the image is called stride.

The Feature Maps are compressed versions of the input images. Since we do lose some information while compressing it, it&#39;s also important that the elements in the feature map are integral. Elements that are important enough to detect or recognise the object.
 Applying several filters on the input image, we get several Feature Maps.

**ReLU Layer:**
 All input images have several non linear elements such as different colours, borders, transitions between pixels, etc. And we need to keep maintaining the nonlinearity of the image. But by applying filters, we get some linearity in the resultant image and hence we apply ReLU(Rectified Linear Unit) to remove linearity. It&#39;s an activation function for the outputs of the CNN neurons where the gradient is always high.

1. **Pooling:**

This is done to enhance the spatial invariance in our images, i.e, it:

1. Reduces size of the images.
2. Avoids overfitting of data.
3. Preserves the main features needed.

It&#39;s like a filter that goes through the feature maps summarising the features.
 There are several types of pooling such as mean pooling, max pooling, sum pooling and so on. The one I prefer is the max pooling. It gives the maximum element from each region of the Feature Map on which the max pooling is done. The following is an example:

**3. Flattening:**

After Pooling is done, all the images are flattened to get a long vector or column of values into an ANN. So, basically just take the numbers row by row, and put them into this one long column. For example:

**4. Full Connection:**

Once the flattening is done, the flattened layer behaves as the input layer, we have the fully connected layers which are the hidden layers of CNN and we have the output layer:

The main aim to do so is to make more attributes that describe the classes (x and y) better. When the predictions are made whether it&#39;s x or y, we find errors which are called the loss function and that&#39;s when we come across a cross entropy function that minimizes the loss function in order to maximize the performance of our network. There are several types of errors such as classification error, mean square error, cross entropy error and so on.

The process involves forward and backward propagation leading to many iterations and epox and in the end we have a fully connected Neural Network- CNN that recognizes images and classifies them.

**Code:**

#Importing the Keras Libraries and packages

**from keras.models import Sequential**

#initializes the NN in a sequential manner

**from keras.layers import Convolution2D**

#for the convolutional layers and 2D images--deals with images

**from keras.layers import MaxPooling2D**

#for the Pooling layers

**from keras.layers import Flatten**

#for the flattening layers--converts the pooling features into large connected vectors that act as inputs to the NN

**from keras.layers import Dense**

#adds the fully connected layers into an ANN

#Initialising the CNN

**classifier = Sequential()** #an object of the class

#Step-1 :- Convolution

**classifier.add(Convolution2D(32, 3, 3, input\_shape= (64, 64, 3), activation=&#39;relu&#39;))**

#Step-2 Pooling

**classifier.add(MaxPooling2D(pool\_size=(2,2)))**

#Step-3 Flattening

**classifier.add(Flatten())**

#Step-4 Full Connection

**classifier.add(Dense(output\_dim=128,activation=&#39;relu&#39;))**

**classifier.add(Dense(output\_dim=1,activation=&#39;sigmoid&#39;))**

#output\_dim is the no. of nodes in the hidden layer taken between the no. of input nodes and the no. of output nodes...too small will not make the classifier a good model whereas too big no. is highly compute sensitive.
 #In the output layer, we use the sigmoid function instead of relu because we need a binary output. We use Softmax activation function if we have more than 2 outcomes.

#Step-5 Compiling the CNN

**classifier.compile(optimizer=&#39;adam&#39;,loss=&#39;binary\_crossentropy&#39;,metrics=[&#39;accuracy&#39;])**

#The optimizer parameter is used to choose the stochastic gradient descent algorithm(to update the parameters minimizing the loss function which in this case is the Binary Cross entropy) Adam algorithm(one of the best optimizers)

#Fitting images to CNN

**from keras.preprocessing.image import ImageDataGenerator**

#used for image augmentation(preprocessing an image to avoid overfitting)

**train\_datagen = ImageDataGenerator(**

**rescale=1./255,**

**shear\_range=0.2,**

**zoom\_range=0.2,**

**horizontal\_flip=True)**
#rescale-puts pixel state value between 1/255 so that all the pixel values will be between 0 and 1

#shear\_range-to apply random transvections

#zoom\_range-apply some random zooms

#horizontal\_flip-images will be flipped horizontally

**test\_datagen = ImageDataGenerator(rescale=1./255)**

**training\_set = train\_datagen.flow\_from\_directory(**

**&#39;C://Users//Shriyashish Mishra//Desktop//ml//dataset//training\_set&#39;,**

**target\_size=(64, 64),**

**batch\_size=32,**

**class\_mode=&#39;binary&#39;)**

#flow\_from\_directory is used because the dataset with the two folders-training set and test set- is put in the working directory folder.

#target\_size is the size of the images that is expected by CNN model

#batch\_size is the size of the batch of the sample images will go through the CNN model after which the weight will be updated

#class\_mode is binary because we have two classes here-cats and dogs

**test\_set= test\_datagen.flow\_from\_directory(**

**&#39;C://Users//Shriyashish Mishra//Desktop//ml//dataset//test\_set&#39;,**

**target\_size=(64, 64),**

**batch\_size=32,**

**class\_mode=&#39;binary&#39;)**

#creating the test\_set just like we created the training\_set

**classifier.fit\_generator(**

**training\_set,**

**samples\_per\_epoch=8000,**

**nb\_epoch=25,**

**validation\_data=test\_set,**

**nb\_val\_samples=2000)**

#Here, we fit our CNN to the training set, while also testing its performance on the test set.

#We put the training\_set as our input.

#samples\_per\_epoch is the no. of images in the training\_set.

#nb\_epoch is the no. of epochs that we want to choose to train our CNN.

#test\_set is the validation\_data because it is on which we want to evaluate the performance on CNN

#nb\_val\_samples is the no. of images on our test set

#Run it till the no. of epochs reach 25/25
