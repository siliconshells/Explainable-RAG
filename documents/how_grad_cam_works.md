Artificial intelligence is no longer a futuristic concept, it’s here now. AI systems are used everywhere, from helping to identify cats and dogs in your photo gallery to analyzing medical imagery for diagnoses, often faster and sometimes better than humans.
We’ve all wondered how they do it. Behind every accurate (or inaccurate) classification is a wealth of mathematics and statistics. But the real magic, and the biggest mystery, lies in the AI’s “brain.” How does an AI system decide to label an image a particular way?
If we knew what specific features an AI focuses on, we could do two critical things:
1. Improve its performance by making those key features more prominent in the data.
2. Increase trust by explaining its decisions, especially when it makes a mistake.
This pursuit of transparency is vital for improving human-AI interaction, which is becoming an increasingly large part of our lives. This is where Grad-CAM (Gradient-weighted Class Activation Mapping) comes in. Presented at the IEEE International Conference on Computer Vision (ICCV) in 2017, Grad-CAM was designed specifically to work with Convolutional Neural Networks (CNNs). While it has several variants, the base version is the key to unlocking AI transparency.
What is a CNN?
Before diving into Grad-CAM, let’s understand its target. CNNs are a type of Deep Neural Network, mimicking how the human brain’s visual cortex works. They are the bedrock of modern computer vision, primarily used for image classification, and are critical across diverse industries like medicine and autonomous vehicles.
These networks, such as the popular ResNet50 (meaning it has 50 layers), are often described as black boxes. Their complexity makes it incredibly difficult to understand exactly what they do, layer by layer, to arrive at a final object identification.
Grad-CAM provides a solution, giving us a clear idea of what parts of an input image the CNN considered most important to make its determination.
How Grad-CAM Works: Targeting the Expert Layer
A CNN processes an image through multiple layers, with each layer extracting different kinds of information, from simple edges and shapes to complex object parts. By the time the image reaches the final layer, the network has gathered everything it needs to make a classification. This is why Grad-CAM targets the last convolutional layer. This layer is rich with high-level semantic information. In a network like ResNet50, this last layer can contain over 2,048 sub-units, which we call feature maps.
Think of each feature map as an expert focused on a very specific feature: one might be an expert on faces, another on sharp edges, and yet another on a specific texture.
Creating the Heatmap
The final decision is made by aggregating the results of these feature maps with different levels of importance. To create the visual explanation, Grad-CAM performs a clever trick:
1. It measures the importance (gradient) of each feature map concerning the final classification decision.
2. It aggregates the results, shading them from dull to bright in a representative image.
3. Because the feature maps are significantly smaller than the original image (often), the resulting visualization must be resized and superimposed over the original image.
The resulting image is a heatmap that clearly shows which pixels the network focused on.
A quick technical note: A function called ReLU (Rectified Linear Unit) is applied to the aggregated results. This ensures that only the parts of the image that positively contributed to the final decision are highlighted, simplifying the interpretation of the resulting heatmap. Parts that may have negatively contributed to the decision are dropped.
As the colors in the heatmap move from blue (low importance) to red (high importance), we see an increase in the influence of those parts of the image on the CNN’s final classification.

Figure 1. An original input image we will pass through the CNN. We will use Grad-CAM to understand which parts are important for specific classifications.

Figure 2. A heatmap showing what a model is focusing on to classify this image as an image of a cat. The red areas show the highest importance.

Figure 3. A heatmap showing what a model is focusing on to classify this image as a dog. Notice the model focuses on different features than for a cat.
Press enter or click to view image in full size

Figure 4. An original input image and a heatmap showing what ResNet50 is focusing on to classify an image as that of a Lycaenid butterfly. Notice the model focuses on the head and patterns in the wings.
The immense value of Grad-CAM is that it shows you what the model is thinking, whether it correctly says an image is a butterfly or mistakenly says it is a bird.
Grad-CAM vs. The Alternatives: Speed and Efficiency
Grad-CAM is not the only way to get a peek inside the black box. One close alternative is the Occlusion Map.
The Occlusion Map Approach
Occlusion mapping works by repeatedly removing (occluding) parts of the input image, passing the modified image through the CNN, and checking how much the classification accuracy changes. The part that causes the biggest drop in accuracy is deemed the most important.
It is more model-agnostic (works with a wider range of models) and can be very detailed. However, a challenge it has is that it is very slow and computationally expensive. You need hundreds or even thousands of passes through the CNN to test every part of the image.
The Grad-CAM Advantage
Grad-CAM, on the other hand, is extremely efficient. Since it picks its information directly from the model’s internal layers and weights, it requires only a single forward and backward pass (which is already part of the classification process) to get all the information it needs.
This makes it far more computationally cheaper. The trade-off is that it requires full access to the model’s internal structure, making it slightly more restricted than a truly model-agnostic technique.
In conclusion, Grad-CAM gives us a crucial view into the Convolutional Neural Network’s mind. It doesn’t tell us whether the classification was right or wrong, it only tells us what the CNN was focusing on when it was thinking, allowing us to build, refine, and trust our AI systems with greater confidence.
Reference
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2019). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. International Journal of Computer Vision, 128, 336–359.