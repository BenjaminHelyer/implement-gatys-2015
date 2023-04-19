# implement-gatys-2015
Implementation of "A Neural Algorithm of Artistic Style" by (Gatys et al, 2015).

# Content Extraction
This implementation provides an interface for content extraction, the ContentExtractor class.
The class primarily allows one to extract content from a given image by calling the 
generate_content_image() function. 

Additionally, the class provides an interface to
the VGG model in general, allowing one to visualize certain layers via functions such
as visualize_activations() and visualize_original_content(). An interface to the basic
classification capabilities of the VGG model is also provided with the function
get_top_k_predictions().

# References
Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.

Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2414-2423).
