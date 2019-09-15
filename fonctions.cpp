#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <string.h>
#include "fonctions.h"


Mat fonctionLineaireSaturation(Mat image, Point point1, Point point2){
    Mat imageFinale(image.rows, image.cols, CV_8UC3, Scalar( 255,255,255));

    for(int i = 0; i < image.rows; i++){

        for(int j = 0; j < image.cols; j++){

            for(int c = 0; c < 3; c++){

					int val = image.at<Vec3b>(i,j)[c];

					if(val>=0 && val <= point1.x) val = 0;
    				else if(point1.x < val && val <= point2.x)
						val =(int) (255/(point2.x - point1.x)) * (val - point1.x);
    				else if(point2.x < val && val <= 255) val = 255;
               imageFinale.at<Vec3b>(i,j)[c] = saturate_cast<uchar>(val);
            }
        }
    }

    return imageFinale;
}


//------------Détection de contours avec Canny------------//
Mat CannyDetector(Mat imgOriginale, int seuilMin, int seuilR){
/*Ce code est inspiré de: http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html */
	Mat imgContour;
	Mat imgDestination;
	Mat imgGray;
//Creation d'une matrice de meme taille que l'image originale
	imgDestination.create( imgOriginale.size(), imgOriginale.type() );
//Conversion de l'image en entrée en une image en niveau de gris
	cvtColor( imgOriginale, imgGray, CV_BGR2GRAY );
//reduction du bruit avec un noyau 3x3
	blur(imgGray, imgContour, Size(3,3));
//détecteur canny
	Canny(imgContour, imgContour, seuilMin, seuilMin*seuilR, 3 );
//on utilise Canny comme un masque pour afficher notre résultat
	imgDestination = Scalar::all(0);

	imgOriginale.copyTo( imgDestination, imgContour);

	return imgDestination;
}//fin CannyDetector


Mat segmentation(Mat imgSource){
    Mat imageResult;
    // conversion de l'image couleur en image en niveaux de gris
    cvtColor(imgSource, imageResult, COLOR_RGB2GRAY);
     // application de l'algorithme d'OTSU
    threshold(imageResult, imageResult, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

    return imageResult;

}

Mat postSegmentation(Mat imgSource){
    // définition des éléments structurants
    Mat struct_element_1 = getStructuringElement(MORPH_RECT, Size(16, 16));
    Mat struct_element_2 = getStructuringElement(MORPH_RECT, Size(7, 7));
    Mat transform_Img;

    /// dilatation et erosion
    dilate(imgSource, transform_Img, struct_element_1);
    dilate(imgSource, transform_Img, struct_element_1);
    erode(transform_Img, transform_Img, struct_element_2);
    erode(transform_Img, transform_Img, struct_element_2);

    // déclaration d'un vecteur pour contenir les contours des régions détectées
    vector<vector<Point> > contours;

    // recherche des contours des régions détectées
    findContours(transform_Img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // etiquetage des régions
    Mat markers = Mat::zeros(transform_Img.size(), CV_32SC1);
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i) + 1), -1);

    /// génération aléatoire de couleurs
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++) {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
    }

    // coloration des régions avec des couleurs différentes
    Mat post_seg_img = Mat::zeros(markers.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                post_seg_img.at<Vec3b>(i, j) = colors[index - 1];
            else
                post_seg_img.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        }
    }


    return post_seg_img;


}
