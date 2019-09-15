
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <string.h>

using namespace cv;
using namespace std;

// Fonction pour modifier le contraste avec la fonction lineaire de saturation
Mat fonctionLineaireSaturation(Mat image, Point P1, Point P2);
Mat CannyDetector(Mat imgOriginale, int seuilMin, int seuilR);
Mat transformationLinsaturation(Mat image, int smin, int smax);
Mat segmentation(Mat imgSource);
Mat postSegmentation(Mat imgSource);

