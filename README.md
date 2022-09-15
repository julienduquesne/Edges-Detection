# Edges-Detection
This project aims at being able to detect edges on a picture. It was coded in python, I used Spyder as an editing interface.

# Method of Canny
I aimed at detecting edges on an image. I followed the following steps to do so:
## Smoothing image
I used a gaussian filter to remove noise from the image. It consists in a convolution of the image with a gaussian function in two dimensions
## Computing gradient 
We can easily compute the gradient of an image by using the Sobel filter.
## Thinning gradient
After computing it, I needed to thin it. To do so, I only kept the maximas of gradient, using the direction of gradient and keeping only pixels that has a higher gradient than his neighbors
## Separating gradient between strong and weak edges
I fixed a threshold to separate gradient and keep strong gradient points, i.e points that have a gradient over the threshold
## Finding the final edges
I started from the strong edges and kept only the weak edges that were linked to a strong edge.
