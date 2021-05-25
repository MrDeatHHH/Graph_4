#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <numeric>
#include <algorithm>

using namespace cv;
using namespace std;

const double infinity = 1000000.;

// Xor operation
bool xor_ab(int a, int b)
{
	return (((a > 0) && (b == 0)) || ((a == 0) && (b > 0)));
}

bool is_zero(double a, double epsilon = 0.00001)
{
	return ((a > -epsilon) && (a < epsilon));
}

// Sum logs of probs instead of Mult probs
double probability(const double noise, int pos, const int ch_h, int ch_w, int** alphabet_img, int** img)
{
	double prob = 0.;
	for (int x = pos - ch_w; x < pos; ++x)
		for (int y = 0; y < ch_h; ++y)
			prob += (xor_ab(img[x][y], alphabet_img[x - pos + ch_w][y]) ?
				(is_zero(noise) ? -infinity : log(noise)) :
				(is_zero(1. - noise) ? -infinity : log(1. - noise)));
	return prob;
}

void get_d_best(const int p,
	const int c,
	double*** f,
	int*** k,
	const int d,
	const int alphabet_size,
	int* alphabet_width,
	double* freq,
	const double noise,
	const int ch_height,
	int*** alphabet_img,
	int** img)
{
	// Form the array
	const int N = d * alphabet_size;
	double* A = new double[N];
	for (int c_ = 0; c_ < alphabet_size; ++c_)
	{
		int j = p - alphabet_width[c_];
		if (j >= 0)
		{
			for (int r = 0; r < d; ++r)
			{
				A[c_ * d + r] = 0.;
				A[c_ * d + r] += is_zero(freq[c_ * alphabet_size + c]) ? -infinity : log(freq[c_ * alphabet_size + c]);
				A[c_ * d + r] += probability(noise, p, ch_height, alphabet_width[c_], alphabet_img[c_], img);
				A[c_ * d + r] += f[j][c_][r];
			}
		}
		else
		{
			for (int r = 0; r < d; ++r)
				A[c_ * d + r] = -infinity;
		}
	}

	// Sort
	vector<int> V(N);
	int x = 0;
	std::iota(V.begin(),V.end(),x++);
	sort(V.begin(),V.end(), [&](int i,int j) {return A[i] > A[j]; });

	// Save max found
	for (int r = 0; r < d; ++r)
	{
		f[p][c][r] = A[V[r]];
		k[p][c][r] = int(V[r] / d);
	}

	delete[] A;
}

// Sum logs of probs instead of Mult probs
int** solve(int* &n,
	const int d,
	const double noise,
	int const alphabet_size,
	const int ch_height,
	const int ch_width,
	double* freq,
	int* alphabet_width,
	const int height,
	const int width,
	int*** alphabet_img,
	int** img)
{
	// Initialize f
	double*** f = new double** [width + 1];
	for (int p = 0; p < width + 1; ++p)
	{
		f[p] = new double* [alphabet_size]();
		for (int c = 0; c < alphabet_size; ++c)
			f[p][c] = new double[d];
	}

	// f[0][k_0] = p(k_0) = p(k_0 | " ")
	for (int c = 0; c < alphabet_size; ++c)
	{
		f[0][c][0] = is_zero(freq[(alphabet_size - 1) * alphabet_size + c]) ? -infinity : log(freq[(alphabet_size - 1) * alphabet_size + c]);
		for (int r = 1; r < d; ++r)
			f[0][c][r] = -infinity;
	}

	// Initialize k taken for f[i]
	int*** k = new int** [width + 1]();
	for (int p = 0; p < width + 1; ++p)
	{
		k[p] = new int* [alphabet_size]();
		for (int c = 0; c < alphabet_size; ++c)
			k[p][c] = new int[d];
	}

	// k[0][c] = -1 for safety
	for (int c = 0; c < alphabet_size; ++c)
		for (int r = 0; r < d; ++r)
			k[0][c][r] = -1;

	// Calculate all f for i in [1, width - 1]
	for (int p = 1; p < width; ++p)
	{
		for (int c = 0; c < alphabet_size; ++c)
		{
			get_d_best(p, c, f, k, d, alphabet_size, alphabet_width, freq, noise, ch_height, alphabet_img, img);
		}
	}

	// Calculate f[width][alphabet_size - 1]
	get_d_best(width, alphabet_size - 1, f, k, d, alphabet_size, alphabet_width, freq, noise, ch_height, alphabet_img, img);

	// Initialize res
	int** res = new int*[width + 1]();
	for (int p = 0; p < width + 1; ++p)
		res[p] = new int[d];

	n = new int[d];
	for (int r = 0; r < d; ++r)
		n[r] = 0;

	int* pos_cur = new int[d];
	for (int r = 0; r < d; ++r)
		pos_cur[r] = width;

	int* ch_cur = new int[d];
	for (int r = 0; r < d; ++r)
		ch_cur[r] = k[width][alphabet_size - 1][r];

	for (int r = 0; r < d; ++r)
	{
		while (pos_cur[r] > 0)
		{
			res[n[r]][r] = ch_cur[r];
			n[r] += 1;
			pos_cur[r] -= alphabet_width[ch_cur[r]];
			ch_cur[r] = k[pos_cur[r]][ch_cur[r]][r];
		}
	}

	for (int p = 0; p < width + 1; ++p)
	{
		for (int r = 0; r < d; ++r)
			delete[] k[p][r];
		delete[] k[p];
	}
	delete[] k;
	for (int p = 0; p < width + 1; ++p)
	{
		for (int r = 0; r < d; ++r)
			delete[] f[p][r];
		delete[] f[p];
	}
	delete[] f;
	delete[] pos_cur;
	delete[] ch_cur;

	return res;
}

int main()
{
	//srand(time(NULL));
	int const alphabet_size = 27;
	fstream file;
	file.open("freq.txt", ios::in);

	const int ch_height = 28;
	const int ch_width = 28;

	char alphabet[] = "abcdefghijklmnopqrstuvwxyz1";

	// Freq
	double* freq = new double[alphabet_size * alphabet_size];
	for (int i = 0; i < alphabet_size * alphabet_size; ++i)
		file >> freq[i];

	// Alphabet imgs
	string folder = "alphabet/";
	string suffix = ".png";
	int* alphabet_width = new int[alphabet_size];
	int*** alphabet_img = new int** [alphabet_size];
	for (int c = 0; c < alphabet_size; ++c)
	{
		Mat image;
		string name(1, alphabet[c]);
		image = imread(folder + name + suffix, IMREAD_UNCHANGED);
		alphabet_width[c] = image.size().width;
		alphabet_img[c] = new int* [ch_width];
		for (int x = 0; x < ch_width; ++x)
		{
			alphabet_img[c][x] = new int[ch_height];
			for (int y = 0; y < ch_height; ++y)
			{
				alphabet_img[c][x][y] = int(image.at<uchar>(y, x));
			}
		}
	}

	// Input img
	const double noise = 0.4;
	Mat image;
	image = imread("input/i am alone the villain of the earth and feel i am so most_0.4.png", IMREAD_UNCHANGED);

	const int height = image.size().height;
	const int width = image.size().width;

	// Get array from Mat
	int** img = new int* [width];
	for (int x = 0; x < width; ++x)
	{
		img[x] = new int[height];
		for (int y = 0; y < height; ++y)
			img[x][y] = int(image.at<uchar>(y, x));
	}
	int* n;
	const int d = 10;
	int** res = solve(n, d, noise, alphabet_size, ch_height, ch_width, freq, alphabet_width, height, width, alphabet_img, img);

	cout << "Results" << endl;

	for (int r = 0; r < d; ++r)
	{
		for (int c = n[r] - 1; c >= 0; --c)
		{
			cout << ((alphabet[res[c][r]] != '1') ? alphabet[res[c][r]] : ' ');
		}
		cout << endl;
	}

	waitKey(0);
	return 0;
}