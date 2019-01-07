//#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

#include <Eigen/Geometry> 
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h> 
#include <pcl/io/pcd_io.h> 
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace cv;
using namespace std;

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, Mat& structure, vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << structure.cols;
	
	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.cols; ++i)
	{
		Mat_<float> c = structure.col(i);
		c /= c(3);	// Koordinat homogen, yang perlu dibagi dengan elemen terakhir untuk menjadi nilai koordinat sebenarnya
		fs << Point3f(c(0), c(1), c(2));
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();
}

int main( int argc, char** argv )
{
	// STAGE 1
	vector<KeyPoint> key_points1, key_points2;
	Mat descriptor1;
	Mat descriptor2;
	
	VideoCapture cap("/home/ferdyan/sfm-two-8b10fb671b791e818d12e24157c3ce787849d0b9/build/urd.mp4");

  	Mat currImage_c, currImage, prevImage_c, prevImage;
  	int i = 0;
  	typedef pcl::PointXYZRGB PointT;									// CREATE POINT STRUCTURE PCL
	typedef pcl::PointCloud<PointT> PointCloud;							// BASE CLASS IN PCL FOR STORING COLECTIONS OF 3D POINTS
	PointCloud::Ptr pointcloud( new PointCloud );



  	while(1)
  	{
    	cap >> currImage;

    	cout << "i" << i << endl;

    	if (currImage.empty())
    	{
      	break;
    	}


    	if(prevImage.empty())
    	{
    		currImage.copyTo(prevImage);
    		cap >> currImage;
    	}

    	//imshow("current", currImage);
    	//imshow("previous", prevImage);


    	// ektrak fitur
		Ptr<AKAZE> feature = AKAZE::create();
    	assert(!prevImage.empty());
    	feature->detect(prevImage, key_points1);
    	feature->compute(prevImage, key_points1, descriptor1);
    	cout << "Jumlah KeyPoint 1 = " << key_points1.size() << endl;
    	Mat output_key1;
    	drawKeypoints(prevImage, key_points1, output_key1);

    	vector<Vec3b> colors_for_all1(key_points1.size());
		for (int i = 0; i < key_points1.size(); ++i)
		{
			Point2f& p = key_points1[i].pt;
			colors_for_all1[i] = prevImage.at<Vec3b>(p.y, p.x);
		}



    	assert(!currImage.empty());
    	feature->detect(currImage, key_points2);
    	feature->compute(currImage, key_points2, descriptor2);
    	cout << "Jumlah KeyPoint 2 = " << key_points2.size() << endl;
    	Mat output_key2;
    	drawKeypoints(currImage, key_points2, output_key2);

    	vector<Vec3b> colors_for_all2(key_points2.size());
		for (int i = 0; i < key_points2.size(); ++i)
		{
			Point2f& p = key_points2[i].pt;
			colors_for_all2[i] = currImage.at<Vec3b>(p.y, p.x);
		}


		// STAGE 2
		cout << "stage 2" << endl;
    	vector<vector<DMatch>> knn_matches;
    	vector<DMatch> matches;
		BFMatcher matcher(NORM_L2);
		matcher.knnMatch(descriptor1, descriptor2, knn_matches, 2);

		float min_dist = FLT_MAX;
		for (int r = 0; r < knn_matches.size(); ++r)
		{
			//Ratio Test
			if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
				continue;

			float dist = knn_matches[r][0].distance;
			if (dist < min_dist) min_dist = dist;
		}

		cout << "knn matches 1 " << knn_matches.size() << endl; // 3183
		matches.clear();
		cout << "knn matches 2 " << knn_matches.size() << endl; // 3183

		for (size_t r = 0; r < knn_matches.size(); ++r)
		{
			// Pengecualian jarak yg memenuhi rasio
			if (
				knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
				knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
				)
				continue;

			// Simpan keypoint
			//cout << "push = " << knn_matches[r][0].distance << endl; // 49
			matches.push_back(knn_matches[r][0]);
		}
		cout << "knn matches 3 = " << knn_matches.size() << endl; // 3183
		cout << "matches 3 = " << matches.size() << endl; // 49


		// STAGE 3
		vector<Vec3b> c1, c2;
		Mat R, T;	// Matriks rotasi dan vektor terjemahan
		Mat mask;	// Poin dalam topeng lebih besar dari nol mewakili poin yang cocok, sama dengan nol mewakili poin ketidakcocokan
		vector<Point2f> out_p1;
		vector<Point2f> out_p2;

		out_p1.clear();
		out_p2.clear();

		for (int i = 0; i < matches.size(); ++i)
		{
			out_p1.push_back(key_points1[matches[i].queryIdx].pt);
			out_p2.push_back(key_points2[matches[i].trainIdx].pt);
		}


		// STAGE 4
		vector<Vec3b> out_c1;
		vector<Vec3b> out_c2;

		out_c1.clear();
		out_c2.clear();

		for (int i = 0; i < matches.size(); ++i)
		{
			out_c1.push_back(colors_for_all1[matches[i].queryIdx]);
			out_c2.push_back(colors_for_all2[matches[i].trainIdx]);
		}


		// STAGE 5
		Mat K(Matx33d(
			2759.48, 0, 1520.69,
			0, 2764.16, 1006.81,
			0, 0, 1));

		double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
		Point2d principle_point(K.at<double>(2), K.at<double>(5));

		Mat E = findEssentialMat(out_p1, out_p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
		double feasible_count = countNonZero(mask);
		cout << (int)feasible_count << " -in- " << out_p1.size() << endl;

		recoverPose(E, out_p1, out_p2, R, T, focal_length, principle_point, mask);
		cout << "rotational = " << endl << R << endl;
    	cout << "translation = " << endl << T << endl;




    	// STAGE 6
    	Mat structure;	// matriks 4 baris dan kolom N, masing-masing kolom mewakili titik dalam ruang (koordinat homogen)
    	vector<Point2f> p1_copy = out_p1;
		out_p1.clear();

		// Maskout points
		for (int i = 0; i < mask.rows; ++i)
		{
			if (mask.at<uchar>(i) > 0)
			{
				out_p1.push_back(p1_copy[i]);
			}
		}

		vector<Point2f> p2_copy = out_p2;
		out_p2.clear();

		for (int i = 0; i < mask.rows; ++i)
		{
			if (mask.at<uchar>(i) > 0)
			{
				out_p2.push_back(p2_copy[i]);
			}
		}

		Mat proj1(3, 4, CV_32FC1);
		Mat proj2(3, 4, CV_32FC1);

		// RANDOM VALUE
		cout << "proj1 1 = " << endl << proj1 << endl;
		cout << "proj2 1 = " << endl << proj2 << endl;

		proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
		proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);

		R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
		T.convertTo(proj2.col(3), CV_32FC1);

		cout << "proj1 2 = " << endl << proj1 << endl;
		cout << "proj2 2 = " << endl << proj2 << endl;

		Mat fK;
		K.convertTo(fK, CV_32FC1);
		proj1 = fK*proj1;
		proj2 = fK*proj2;

		// Triangulate
		triangulatePoints(proj1, proj2, out_p1, out_p2, structure);

		Mat save_struct[100];
		save_struct[i] = structure;


		// STAGE 7
		// Simpan dan tampilkan
		vector<Mat> rotations = { Mat::eye(3, 3, CV_64FC1), R };
		vector<Mat> motions = { Mat::zeros(3, 1, CV_64FC1), T };

		vector<Vec3b> c1_copy = out_c1;
		out_c1.clear();

		for (int i = 0; i < mask.rows; ++i)
		{
			if (mask.at<uchar>(i) > 0)
			{
				out_c1.push_back(c1_copy[i]);
			}
		}


		//cout << "structure" << structure.size() << endl << endl;
  		// show point cloud model
		//typedef pcl::PointXYZRGB PointT;									// CREATE POINT STRUCTURE PCL
		//typedef pcl::PointCloud<PointT> PointCloud;							// BASE CLASS IN PCL FOR STORING COLECTIONS OF 3D POINTS
		//PointCloud::Ptr pointcloud( new PointCloud );

		for(int z = 0; z <= 10; z++)
		{
			cout << "structure = " << save_struct[z].cols << endl << endl;
			for (int i = 0; i < save_struct[z].cols; ++i)
			{
				Mat_<float> col = save_struct[z].col(i);
				col /= col(3);
				PointT p;
				p.x = col(0);
				p.y = col(1);
				p.z = col(2);
				p.b = out_c1[i][0];
				p.g = out_c1[i][1];
				p.r = out_c1[i][2];

				//cout << col(0) << " " << col(1) << " " << col(2) << " " << out_c1[i]<< endl;

				pointcloud->points.push_back( p );
			}
			pointcloud->is_dense = false;
			//cout << "pointcloud size" << pointcloud.size() << endl;
		}


		if (i == 9)
		{
			pcl::visualization::CloudViewer viewer("Cloud Viewer");
			viewer.showCloud(pointcloud);

			int user_data;
			while(!viewer.wasStopped())
			{
				user_data++;
			}
		}
		



  		prevImage = currImage.clone();
  		i++;

		waitKey(1);
  	}
  	//cout << "structure" << structure.size() << endl;
  	cout << "done" << endl;

	return 0;
}
