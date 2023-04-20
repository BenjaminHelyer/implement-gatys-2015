# implement-gatys-2015
Implementation of "A Neural Algorithm of Artistic Style" by (Gatys et al, 2015).

# Content Extraction
This implementation provides an interface for content extraction, the ContentExtractor class.
The class primarily allows one to extract content from a given image. By itself, content
generation can still yield some interesting results.  

For instance, one can generate content on a modified version of the original image with 
a portion of the original image blocked out. As the generated output shows, the content 
generator "fills in" the content of the blocked out portion while keeping the "style" 
(in this case, color) the same.  
![Example of Generated Content](example_generated_content.jpg)

We can visualize the content layer that we specify as well.  
![Example of Content Layer Activation](example_content_layer.jpg)

# VGG Wrapper
A wrapper is provided to the VGG model, which allows a higher level interaction
with the model than PyTorch gives by default. For example, the wrapper allows 
a visualization of various activations of the VGG model.  

![Example of Activation Visualizations Using the VGG Wraper](example_visualization.jpg)

# References
Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.

Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2414-2423).
