/**
 * @file main.cpp
 * @brief Main Function of the project
 * @author Raingo (raingomm@gmail.com)
 * @version 1.0
 * @date 2011-03-23
 */

/* Copyleft(C)
 * For Raingo
 * Some right reserved
 *
 */


/*! \mainpage Part of the code of the object track project
 *
 * \section intro_sec Introduction
 * This is part of the code of my object track project. \n
 * Just this part consists only the topmost level the utility, and the extra function and class is not included.\n
 * Basic thoughts of this object track project is that since the SIFT feature extracter achieve a good performance on identifying the object but the complexity of the SIFT algorithm is too high to get a real-time performance, and the lk optic flow algorithm achieve a near real-time performance on tracking specific keypoints on the video and utilize the continuity of the video flow, Combining these two algorithm to get trade-off. When the situation get worse, the algorithm choose the SIFT to refine the track process and choose the optic flow algorithm to maintain the track.
 *
 */


#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <ctype.h>
#include <algorithm>



#include "siftCorner.h"
#include "backGroundmodel.h"
#include "reSiftValidate.h"
#include "timerMacros.h"
using namespace cv;
using namespace std;


/* --------------------------------------------------------------------------*/
/**
 * @brief cmpPoint3f The Key Point has been denoted by three dimensional data, and the third dim denote the distance between the train image and the query image
 *
 * @param a The first Keypoint
 * @param b The second Keypoint
 *
 * @return wether the distance of first Keypoint<matched point> is small than the second matched Keypoint
 */
/* ----------------------------------------------------------------------------*/
bool cmpPoint3f(const Point3f &a,const Point3f &b)
{
    return a.z<b.z;
}
/* --------------------------------------------------------------------------*/
/**
 * @brief chooseBestOrder Choose good Matched keypoints based on there distance, noticing that the aimed keypoints are not always the "best" keypoints.
 *
 * @param keypoints the matched keypoints denoted by three dimensional
 * @param points the keypoints with shorter match distance
 * @param maxcount the maximum number of points to be chooseen in the ordered vector
 */
/* ----------------------------------------------------------------------------*/
void chooseBestOrder(vector<Point3f> &keypoints,vector<Point2f> &points,const int maxcount)
{
    //sorting the keypoints based on there matched distance
    sort(keypoints.begin(),keypoints.end(),cmpPoint3f);

    vector<Point3f>::iterator it;
    int count=0;
    points.clear();

    //choose the "Top maxcount" keypoints
    for(it=keypoints.begin();it!=keypoints.end()&&count<maxcount;++it)
    {
        Point2f temp;
        temp.x=(*it).x;
        temp.y=(*it).y;
        points.push_back(temp);
        ++count;
    }
}

/* --------------------------------------------------------------------------*/
/**
 * @brief distanceEqual As the criteria to judge whether the points and near enough to treat them in the same pixel block
 *
 * @param lhs The first point
 * @param rhs The second point
 *
 * @return if and only if they are near enough, return true.
 */
/* ----------------------------------------------------------------------------*/
bool distanceEqual(const Point2f &lhs,const Point2f &rhs)
{
    static double temp=0.0;
    static double distance=0.0;

    /* --------------------------------------------------------------------------*/
    /**
     * @brief distanceThre parameter to determine whether the two points are near enough
     */
    /* ----------------------------------------------------------------------------*/
    const double distanceThre=100;
    temp=lhs.x-rhs.x;
    distance=temp*temp;
    temp=lhs.y-lhs.y;
    distance+=temp*temp;
    return distance<=distanceThre;
}
/* --------------------------------------------------------------------------*/
/**
 * @brief chooseBestDistace choose the best group the pixel that are near enough, based on the assumption that that the group of pixel on the aimed object are neighborhood pixels
 *
 * @param src the points that are disorder in the terms of space neighborhood
 * @param results after partition the points, return the group of pixels that has most members
 */
/* ----------------------------------------------------------------------------*/
void chooseBestDistace(const vector<Point2f> &src,vector<Point2f> &results)
{
    vector<int> labels;


    //partition the source points based on the neighborhood quality
    int size=partition(src,labels,distanceEqual);

    //initialize the count array
    int *index_count=new int[size];
    for(int i=0;i<size;++i)
        index_count[i]=0;

    //count the group
    vector<int>::iterator it;
    for(it=labels.begin();it!=labels.end();++it)
        ++index_count[*it];

    //finding the group of pixels with mose members
    int most_index=0;
    int max=-1;
    for(int i=0;i<size;++i)
    {
        if(index_count[i]>max)
        {
            max=index_count[i];
            most_index=i;
        }
    }

    //return the most standingout group of pixels
    results.clear();
    vector<Point2f>::const_iterator srcIts=src.begin();
    for(it=labels.begin();it!=labels.end();++it,++srcIts)
        if(*it==most_index)
            results.push_back(*srcIts);

    cout<<"results size: "<<results.size()<<endl;
    delete [] index_count;
}

/* --------------------------------------------------------------------------*/
/**
 * @brief PrintPoint print a Point on the console
 *
 * @param aim the point to be printed
 */
/* ----------------------------------------------------------------------------*/
void PrintPoint(Point aim)
{
    cout<<'('<<aim.x<<' '<<aim.y<<')'<<endl;
}
/* --------------------------------------------------------------------------*/
/**
 * @brief FindBound finding the rectangle bound of a group of pixels
 *
 * @param aim the group of pixels
 * @param pt1 the lefttop point of the result rectangle
 * @param pt2 the rightbottom point of the result rectangle
 */
/* ----------------------------------------------------------------------------*/
void FindBound(vector<Point2f> aim, Point2f &pt1, Point2f &pt2)
{
    vector<Point2f>::iterator it=aim.begin(),end=aim.end();

    //traverse through the group of pixels, pt1 is the lefttop with minimum x and y; pt2 is the rightbottom point with maximum x and y
    pt1.x=10000;
    pt1.y=10000;
    pt2.x=0;
    pt2.y=0;
    for(;it!=end;++it)
    {
        pt1.x=min(pt1.x,(*it).x);
	    pt1.y=min(pt1.y,(*it).y);
	    pt2.x=max(pt2.x,(*it).x);
	    pt2.y=max(pt2.y,(*it).y);
    }
}
/* --------------------------------------------------------------------------*/
/**
 * @brief main main function
 *
 * @param argc count of the command line
 * @param argv[] command line
 *
 * @return main return
 */
/* ----------------------------------------------------------------------------*/
int main( int argc, char* argv[] )
{
    VideoCapture cap;
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
    Size winSize(10,10);


    /* --------------------------------------------------------------------------*/
    /**
     * @brief MAX_COUNT parameter to determine how many points should be choosen at the process of sorting the points based on the matched distance\n
     * The parameter should be choosen for the factor list below:\n
     * 1,Selectiveness. get rid of the matched points with small distance, and retain the points with enough distance.\n
     * 2,Stability. make sure that the aimed points that on the object are retained though they are not always with maximum distance.\n
     */
    /* ----------------------------------------------------------------------------*/
    const int MAX_COUNT = 20;



    bool needToInit =true;
    bool nightMode = false;
    bool reSift=true;

    //initialize VideoCapture
    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
    else if( argc == 2 )
        cap.open(argv[1]);
    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }

    Mat gray, prevGray, image;
    vector<Point2f> points[2];
    VideoWriter wri;


    //initialize the SIFT detection object
    const string siftConfigfile="track.config";
    siftCorner cornerFinder;
    if(!cornerFinder.Init(siftConfigfile))
    {
        cout<<"Can not Init cornerFinder"<<endl;
        return 0;
    }


    //initialize the background Model
    const string backGroundFilename="background.jpg";
    const float alpha=0.85;
    Mat backGround;
    backGround=imread(backGroundFilename,CV_LOAD_IMAGE_GRAYSCALE);
    backGroundModel bgModel;
    bgModel.Init(alpha,backGround);


    //initialize the criteria object to determine whether the lk optic flow algorithm has run into a worse condition, and wether we should refind the keypoints using the SIFT algorithm
    reSiftValidate validator;

    //initialize the timing macro
    DECLARE_TIMING(myTimer);
    START_TIMING(myTimer);
    DECLARE_TIMING(siftTimer);
    for(;;)
    {
        Mat frame;
        cap >> frame;
        if( frame.empty() )
            break;

        //convert the frame into grayscale image
        frame.copyTo(image);
        cvtColor(image, gray, CV_BGR2GRAY);

	    if( nightMode )
            image = Scalar::all(0);

        if( needToInit )
        {
            //initialize the VideoWriter
            char fileNameBuffer[30];
            time_t rawtime;
            struct tm * timeinfo;

            time ( &rawtime );
            timeinfo = localtime ( &rawtime );

            sprintf(fileNameBuffer
                    ,"%d_%d_%d_%d_%d_%d.avi"
                    ,timeinfo->tm_year+1900,timeinfo->tm_mon,timeinfo->tm_mday,timeinfo->tm_hour,timeinfo->tm_min,timeinfo->tm_sec);
            wri.open(fileNameBuffer,CV_FOURCC('X','V','I','D'),50,image.size(),true);
            if(!wri.isOpened())
            {
                cout<<"can not init the writer"<<endl;
                return 0;
            }
            needToInit = false;
        }


        //median filter to denoise
        medianBlur(gray,gray,3);

        //renew the background model
        bgModel.renewModel(gray);



        if(reSift)
        {
            cout<<"reSift"<<endl;

            START_TIMING(siftTimer);

            //minus the background to attain the foregound mask
            Mat Mask;
            bgModel.substractModel(gray,Mask);

            //get the keypoints of the object to track using algorithm
            vector<Point3f> keypoints;
            cornerFinder.goodFeatures(gray,keypoints,Mask);

            //get the "top MAX_COUNT" keypoints based on their matched distance
            vector<Point2f> pointstemp;
            chooseBestOrder(keypoints,pointstemp,MAX_COUNT);

            //Get tht group of the pixel on the object based on the neighborhood quality
            chooseBestDistace(pointstemp,points[1]);

            STOP_TIMING(siftTimer);


            cout<<"reSift Done"<<endl;
            reSift=false;
        }
        else
        {
            if( !points[0].empty()  )
            {
                //光流法跟踪关键点
                //Track the keypoints based on the lk optic flow algorithm
                vector<uchar> status;
                vector<float> err;
                if(prevGray.empty())
                    gray.copyTo(prevGray);
                calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0);
                size_t i, k;

                /* --------------------------------------------------------------------------*/
                /**
                 * @brief 如果发现某关键点跟踪前后变化太少，那么直接排除掉，变化的阈值errMin可变\n
                 * 选择的依据是采用记录所有的一般的变化，看看那些点处在较低的谷点，直接排出
                 */
                /* ----------------------------------------------------------------------------*/
                /* --------------------------------------------------------------------------*/
                /**
                 * @brief errMin in order to get rid of the Tracked points that change too small, the parameter errMin determine what is to small
                 */
                /* ----------------------------------------------------------------------------*/
                const float errMin=25;
                for( i = k = 0; i < points[1].size(); i++ )
                {
                    //两种情况需要排除关键点，跟踪状态不佳，跟踪前后区域变化太少
                    //criteria to get rid of Tracked points: status are not so good, or the changes are too small
                    if( !status[i] || err[i]<errMin)
                        continue;
                    points[1][k++] = points[1][i];
                    circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
                }
                points[1].resize(k);

                //Find the rectangle to bound the object keypoints
                Point2f pt1,pt2;
	            FindBound(points[1],pt1,pt2);
	            rectangle(image,pt1,pt2,Scalar(255,0,0),3);

                //determine whether the optic flow Track has run into a bad condition
                reSift=!validator.validate(pt1,pt2);
            }
            else
            {
                reSift=true;
                cout<<"reSift Type three"<<endl;
            }

        }
        imshow("Track", image);
        wri<<image;

        char c;
        c=(char)waitKey(1);
        if( c == 27 )
            break;
        switch( c )
        {
        case 'r':
        case 'c':
        case 'R':
        case 'C':
            //manully instruct the routine to find new keypoints based on sift
            points[1].clear();
            reSift=true;
            cout<<"reSift Type four"<<endl;
            break;
        case 'n':
            nightMode = !nightMode;
            break;
        case ' ':
            waitKey(-1);
            break;
        default:
            ;
        }

        //save current frame and the keypoints
        std::swap(points[1], points[0]);
        swap(prevGray, gray);
    }
    STOP_TIMING(myTimer);
    printf("Execution time: %f ms.\n", GET_TIMING(myTimer));
    printf("sift Execution time: %f ms.\n", GET_TIMING(siftTimer));
    printf("sift average Execution time: %f ms.\n", GET_AVERAGE_TIMING(siftTimer));

    return 0;
}
