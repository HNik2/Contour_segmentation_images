#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <ctype.h>

#include <stdlib.h>
#include <stdio.h>
#include "fonctions.h"

using namespace cv;
using namespace std;

int main()
{

    Mat imageOriginal1,imageOriginal2, imageOriginal3, imageOriginal4, imageContrasteMod1, imageContrasteMod2, imageContrasteMod3, imageContrasteMod4;
    Point p1, p2;
    Mat contraste ;

    imageOriginal1 = imread("images/objets1.jpg", 1 );
    imageOriginal2 = imread("images/objets2.jpg", 1 );
    imageOriginal3 = imread("images/objets3.jpg", 1 );
    imageOriginal4 = imread("images/objets4.jpg", 1 );
    p1.x = 10;
    p1.y = 0;
    p2.x = 120;
    p2.y = 255;

    imageContrasteMod1 = fonctionLineaireSaturation(imageOriginal1,p1, p2);
    imageContrasteMod2 = fonctionLineaireSaturation(imageOriginal2,p1, p2);
    imageContrasteMod3 = fonctionLineaireSaturation(imageOriginal3,p1, p2);
    imageContrasteMod4 = fonctionLineaireSaturation(imageOriginal4,p1, p2);


    if(!imwrite("images_modifies/objets1.jpg", imageContrasteMod1))
        cout<<"Erreur d enregistrement de l'image"<<endl;

    if(!imwrite("images_modifies/objets2.jpg", imageContrasteMod2))
        cout<<"Erreur d enregistrement de l'image"<<endl;
    if(!imwrite("images_modifies/objets3.jpg", imageContrasteMod3))
        cout<<"Erreur d enregistrement de l'image"<<endl;
    if(!imwrite("images_modifies/objets4.jpg", imageContrasteMod4))
        cout<<"Erreur d enregistrement de l'image"<<endl;


    Mat contour1 = CannyDetector(imageContrasteMod1,55, 2);
    Mat contour2 = CannyDetector(imageContrasteMod2,55, 2);
    Mat contour3 = CannyDetector(imageContrasteMod3,55, 2);
    Mat contour4 = CannyDetector(imageContrasteMod4,55, 2);
    Mat contour5 = CannyDetector(imageOriginal4,55, 2);

    imwrite("images_contour/objet1_contour.jpg" , contour1);
    imwrite("images_contour/objet2_contour.jpg" , contour2);
    imwrite("images_contour/objet3_contour.jpg" , contour3);
    imwrite("images_contour/objet4_contour.jpg" , contour4);
    imwrite("images_contour/objet4_contour2.jpg" , contour5);

    Mat segment1 = segmentation(imageContrasteMod1);
    Mat segment2 = segmentation(imageContrasteMod2);
    Mat segment3 = segmentation(imageContrasteMod3);
    Mat segment4 = segmentation(imageContrasteMod4);

    imwrite("output/segmente1.jpg" , segment1);
    imwrite("output/segmente2.jpg" , segment2);
    imwrite("output/segmente3.jpg" , segment3);
    imwrite("output/segmente4.jpg" , segment4);

    Mat seg_img1, seg_img2, seg_img3, seg_img4;
    Mat post_seg_img1 = postSegmentation(segment1);
    Mat post_seg_img2 = postSegmentation(segment2);
    Mat post_seg_img3 = postSegmentation(segment3);
    Mat post_seg_img4 = postSegmentation(segment4);
    imageOriginal1.copyTo(seg_img1, post_seg_img1);
    imageOriginal2.copyTo(seg_img2, post_seg_img2);
    imageOriginal3.copyTo(seg_img3, post_seg_img3);
    imageOriginal4.copyTo(seg_img4, post_seg_img4);
    imwrite("output/seg_img1.jpg" , seg_img1);
    imwrite("output/post_seg_img1.jpg", post_seg_img1);
    imwrite("output/seg_img2.jpg" , seg_img2);
    imwrite("output/post_seg_img2.jpg", post_seg_img2);
    imwrite("output/seg_img3.jpg" , seg_img3);
    imwrite("output/post_seg_img3.jpg", post_seg_img3);
    imwrite("output/seg_img4.jpg" , seg_img4);
    imwrite("output/post_seg_img4.jpg", post_seg_img4);
    return 0;

}
