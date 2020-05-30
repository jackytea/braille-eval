#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

// Global state variables
enum class Shapes {
	Bump, Hole
};
Shapes currentShape = Shapes::Bump;	// Shape to denote
bool drawingBox = false;			// Drawing flag
cv::Rect boundingBox;				// Bounding box

// Object that represents what the user has marked for a CSV row
struct EvaluationResult {
	int row;
	int col;
	int diameter;
	int bumpOrHole;
	EvaluationResult() : row(0), col(0), diameter(0), bumpOrHole(0) {}
	EvaluationResult(int row, int col, int diameter, int bumpOrHole) {
		this->row = row;
		this->col = col;
		this->diameter = diameter;
		this->bumpOrHole = bumpOrHole;
	}
	void displayResult() {
		std::cout << "Row: " << this->row << std::endl;
		std::cout << "Col: " << this->col << std::endl;
		std::cout << "Diameter: " << this->diameter << std::endl;
		std::cout << "Bump (1) or Hole(2): " << this->bumpOrHole << std::endl;
		std::cout << std::endl;
	}
	bool operator < (const EvaluationResult& evalObj) const {
		return this->row < evalObj.row;
	}
};

// Global CSV parsing state
bool sortByRowValue(EvaluationResult index1, EvaluationResult index2) {
	return (index1.row < index2.row);
}
std::vector<EvaluationResult> imageResults;

// Generate ground truth (GT) CSV file based on specifications in doc
void generateGTCSV(std::string fileName, std::vector<EvaluationResult> results) {
	// Sort by row value (column value works well here too)
	std::sort(results.begin(), results.end(), sortByRowValue);

	// Write column names
	std::ofstream csvFile(fileName);
	csvFile << "row,col,diameter,Bump (1) or Hole (2)" << std::endl;

	// Write column values
	for (unsigned int i = 0; i < results.size(); i++) {
		csvFile << results[i].row << ',';
		csvFile << results[i].col << ',';
		csvFile << results[i].diameter << ',';
		csvFile << results[i].bumpOrHole << std::endl;
	}

	csvFile.close();
}

// Get data from what the algorithm calculated
std::vector<EvaluationResult> parseAlgorithmPredictions(std::string algPredictionFile) {
	// Read algorithm prediction CSV file
	std::ifstream algCSV(algPredictionFile);
	std::vector<EvaluationResult> algorithmPredictions;
	std::string currentLine = "";

	// Skip over column of titles
	std::getline(algCSV, currentLine, '\n');

	// Parse data columns
	while (algCSV.good()) {
		int row = 0;
		int col = 0;
		int diameter = 0;
		int bumpOrHole = 0;

		std::getline(algCSV, currentLine, ',');
		if (currentLine != "") {
			row = std::stoi(currentLine);

			std::getline(algCSV, currentLine, ',');
			col = std::stoi(currentLine);

			std::getline(algCSV, currentLine, ',');
			diameter = std::stoi(currentLine);

			std::getline(algCSV, currentLine, '\n');
			bumpOrHole = std::stoi(currentLine);

			algorithmPredictions.push_back(EvaluationResult(row, col, diameter, bumpOrHole));
		}
	}
	algCSV.close();

	// Sort by row values (column values work well here too as they are pretty unique)
	std::sort(algorithmPredictions.begin(), algorithmPredictions.end(), sortByRowValue);
	return algorithmPredictions;
}

// See where the algorithm predicts bumps and holes to be
void visualizeAlgorithmPredictions(std::string imageFile, std::string algImageFile, std::string algorithmPredictionFile) {
	// Get algorithm prediction data
	std::vector<EvaluationResult> algorithmPredictions = parseAlgorithmPredictions(algorithmPredictionFile);

	// Create image for algorithm to draw on
	cv::Mat algImage = cv::imread(imageFile);

	// Draw circles based on algorithm prediction data
	for (unsigned int i = 0; i < algorithmPredictions.size(); i++) {
		if (algorithmPredictions[i].bumpOrHole == 1) {
			cv::circle(
				algImage,
				cv::Point(algorithmPredictions[i].row, algorithmPredictions[i].col),
				algorithmPredictions[i].diameter / 2, cv::Scalar(255, 0, 0));
		}
		else {
			cv::circle(
				algImage,
				cv::Point(algorithmPredictions[i].row, algorithmPredictions[i].col),
				algorithmPredictions[i].diameter / 2, cv::Scalar(0, 0, 255));
		}
	}

	// Display results and save image
	cv::imwrite(algImageFile, algImage);
	cv::imshow("Algorithm Predictions", algImage);
	cv::waitKey(0);
}


// See if CSV row values for ground truth and algorithm are close
bool isMatch(EvaluationResult gt, EvaluationResult alg) {
	return (std::abs(gt.row - alg.row) < 25) && (std::abs(gt.col - alg.col) < 25) && (std::abs(gt.diameter - alg.diameter) < 25);
}

// Calculate confusion matrix based on Ground Truth and Algorithm Predictions
void calcConfusionMatrix(std::string algorithmPredictionFile, std::vector<EvaluationResult> groundTruth, std::string confusionMatrixFile) {
	// Get algorithm prediction data
	std::vector<EvaluationResult> algorithmPredictions = parseAlgorithmPredictions(algorithmPredictionFile);

	// Make sure to not visit same ground truth point again, map this value to track it
	std::map<EvaluationResult, EvaluationResult> visitedGTPoints;

	// Matrix values
	//first row
	int trulyBumpDetectedAsBump = 0;
	int trulyHoleDetectedAsBump = 0;
	int trulyNoneDetectedAsBump = 0;

	//second row
	int trulyBumpDetectedAsHole = 0;
	int trulyHoleDetectedAsHole = 0;
	int trulyNoneDetectedAsHole = 0;

	//third row
	int trulyBumpDetectedAsNone = 0;
	int trulyHoleDetectedAsNone = 0;

	// Calculate confusion matrix values
	for (unsigned int i = 0; i < groundTruth.size(); i++) {
		for (unsigned int j = 0; j < algorithmPredictions.size(); j++) {
			// Match found and ground truth is bump
			if ((isMatch(groundTruth[i], algorithmPredictions[j])) && groundTruth[i].bumpOrHole == 1
				&& (visitedGTPoints.count(groundTruth[i]) == 0)) {
				if (algorithmPredictions[j].bumpOrHole == 1) {
					trulyBumpDetectedAsBump++;
				}
				else if (algorithmPredictions[j].bumpOrHole == 2) {
					trulyBumpDetectedAsHole++;
				}
				visitedGTPoints[groundTruth[i]] = algorithmPredictions[j];
			}
			// Match found and ground truth is hole
			else if ((isMatch(groundTruth[i], algorithmPredictions[j])) && groundTruth[i].bumpOrHole == 2
				&& (visitedGTPoints.count(groundTruth[i]) == 0)) {
				if (algorithmPredictions[j].bumpOrHole == 1) {
					trulyHoleDetectedAsBump++;
				}
				else if (algorithmPredictions[j].bumpOrHole == 2) {
					trulyHoleDetectedAsHole++;
				}
				visitedGTPoints[groundTruth[i]] = algorithmPredictions[j];
			}
			// No match found, but bump or hole is detected
			else if ((!isMatch(groundTruth[i], algorithmPredictions[j])) && (visitedGTPoints.count(groundTruth[i]) == 0)) {
				if (algorithmPredictions[j].bumpOrHole == 1) {
					trulyNoneDetectedAsBump++;
					trulyNoneDetectedAsBump %= (algorithmPredictions.size());
				}
				else if (algorithmPredictions[j].bumpOrHole == 2) {
					trulyNoneDetectedAsHole++;
					trulyNoneDetectedAsHole %= (algorithmPredictions.size());
				}
			}
		}
	}

	// Calc last remaining rows
	trulyBumpDetectedAsNone = (int)(groundTruth.size()) - (trulyBumpDetectedAsBump + trulyBumpDetectedAsHole);
	trulyHoleDetectedAsNone = (int)(groundTruth.size()) - (trulyHoleDetectedAsBump + trulyHoleDetectedAsHole);

	// Print confusion matrix values
	std::cout << "\nFirst row..." << std::endl;
	std::cout << "Truly a bump and detected as a bump: " << trulyBumpDetectedAsBump << std::endl;
	std::cout << "Truly a hole but detected as a bump: " << trulyHoleDetectedAsBump << std::endl;
	std::cout << "Truly none but detected as a bump: " << trulyNoneDetectedAsBump << std::endl;

	std::cout << "\nSecond row..." << std::endl;
	std::cout << "Truly a bump but detected as a hole: " << trulyBumpDetectedAsHole << std::endl;
	std::cout << "Truly a hole and detected as a hole: " << trulyHoleDetectedAsHole << std::endl;
	std::cout << "Truly none but detected as a hole: " << trulyNoneDetectedAsHole << std::endl;

	std::cout << "\nThird row..." << std::endl;
	std::cout << "Truly a bump but detected as none: " << trulyBumpDetectedAsNone << std::endl;
	std::cout << "Truly a hole but detected as none: " << trulyHoleDetectedAsNone << std::endl;

	// Calculate F1 score (Fbeta score where beta = 1)
	double TP = (double)trulyBumpDetectedAsBump + (double)trulyHoleDetectedAsHole;
	double TN = 0.0;
	double FP = (double)trulyNoneDetectedAsBump + (double)trulyNoneDetectedAsHole;

	// FN is not included in calculation since nothing is actually detected as none
	//double FN = (double)trulyBumpDetectedAsNone + (double)trulyHoleDetectedAsNone;
	double FN = 0.0;
	double recall = 0.0;
	double precision = 0.0;
	double beta = 0.0;
	double fOneMeasure = 0.0;

	// Check for zero division beforehand
	if ((TP + FN) > 0 && (TP + FP) > 0) {
		recall = TP / (TP + FN);
		precision = TP / (TP + FP);
		beta = 1.0;
		fOneMeasure = (1 + std::pow(beta, 2.0)) * ((precision * recall) / ((std::pow(beta, 2.0) * precision) + recall));
		std::cout << "\nRecall: " << recall << std::endl;
		std::cout << "Precision: " << precision << std::endl;
		std::cout << "F1 Measure: " << fOneMeasure << std::endl;
	}
	else {
		std::cout << "\nRecall: " << recall << std::endl;
		std::cout << "Precision: " << precision << std::endl;
		std::cout << "F1 Measure: " << fOneMeasure << std::endl;
	}

	// Make confusion matrix table
	std::ofstream matrixCSV(confusionMatrixFile);
	matrixCSV << " ,Truly a bump,Truly a hole,Truly none" << std::endl;
	matrixCSV << "Detected as a bump," << trulyBumpDetectedAsBump << ',' << trulyHoleDetectedAsBump << ',' << trulyNoneDetectedAsBump << std::endl;
	matrixCSV << "Detected as a hole," << trulyBumpDetectedAsHole << ',' << trulyHoleDetectedAsHole << ',' << trulyNoneDetectedAsHole << std::endl;
	matrixCSV << "Detected as none," << trulyBumpDetectedAsNone << ',' << trulyHoleDetectedAsNone << ",N/A" << std::endl;
	matrixCSV.close();
}

// On-screen instructions for the user
void displayInstructions(char** argv) {
	std::cout << "Program path: " << argv[0] << '\n' << std::endl;
	std::cout << "Instructions: " << std::endl;
	std::cout << "Drag mouse to mark a bump in the image. (Bumps are blue circles)" << std::endl;
	std::cout << "Hold 'shift' and drag mouse to mark a hole in the image. (Holes are red circles)" << std::endl;
	std::cout << "To go to the next image, press 'esc'" << std::endl;
	std::cout << "After reaching the second image, press 'esc' to exit" << std::endl;
}

// Mark a bump or hole in the image
void drawCircle(cv::Mat& img, cv::Rect box) {
	// Bumps denoted by blue circle
	if (currentShape == Shapes::Bump) {
		cv::circle(
			img,
			cv::Point((box.tl().x + box.br().x) / 2, (box.tl().y + box.br().y) / 2),
			std::abs(box.width) / 2, cv::Scalar(255, 0, 0));
	}
	// Holes denoted by red circle
	else if (currentShape == Shapes::Hole) {
		cv::circle(
			img,
			cv::Point((box.tl().x + box.br().x) / 2, (box.tl().y + box.br().y) / 2),
			std::abs(box.width) / 2, cv::Scalar(0, 0, 255));
	}
}

// Handle mouse events for drawing on the image
void mouseEventHandler(int event, int x, int y, int flags, void* param) {
	cv::Mat& image = *(cv::Mat*)param;

	// Indicate if bump or hole is marked
	if (flags & cv::EVENT_FLAG_SHIFTKEY) {
		currentShape = Shapes::Hole;
	}
	else {
		currentShape = Shapes::Bump;
	}

	// Detect mouse events 
	switch (event) {
	case cv::EVENT_MOUSEMOVE: {
		if (drawingBox) {
			boundingBox.width = x - boundingBox.x;
			boundingBox.height = y - boundingBox.y;
		}
		break;
	}
	case cv::EVENT_LBUTTONDOWN: {
		drawingBox = true;
		boundingBox = cv::Rect(x, y, 0, 0);
		break;
	}
	case cv::EVENT_LBUTTONUP: {
		drawingBox = false;
		if (boundingBox.width < 0) {
			boundingBox.x += boundingBox.width;
			boundingBox.width *= -1;
		}
		if (boundingBox.height < 0) {
			boundingBox.y += boundingBox.height;
			boundingBox.height *= -1;
		}

		// Calculate circle dimensions
		int row = (boundingBox.tl().x + boundingBox.br().x) / 2;
		int col = (boundingBox.tl().y + boundingBox.br().y) / 2;
		int diameter = std::abs(boundingBox.width);
		int bumpOrHole = 1;

		// Add CSV row items to global array, check if 'SHIFT' is being held down 
		if (flags & cv::EVENT_FLAG_SHIFTKEY) {
			bumpOrHole = 2;
			imageResults.push_back(EvaluationResult(row, col, diameter, bumpOrHole));
		}
		else {
			imageResults.push_back(EvaluationResult(row, col, diameter, bumpOrHole));
		}
		drawCircle(image, boundingBox);
		break;
	}
	}
}

// Main driver program
int main(int argc, char** argv) {

	// Setup window and initial states
	std::string windowName = "Performance Evaluator";
	displayInstructions(argv);
	boundingBox = cv::Rect(-1, -1, 0, 0);
	cv::namedWindow(windowName);

	// Array of image file names to go over
	std::vector<std::string> fileNames = { "SAM1_sub1.jpg", "SAM1_sub2.jpg" };
	cv::Mat image, temp;

	// Setup mouse event call back function
	cv::setMouseCallback(
		windowName,
		mouseEventHandler,
		(void*)&image
	);

	// Process images
	for (unsigned int i = 0; i <= fileNames.size() - 1; i++) {

		// Clear global vector of old results
		imageResults.clear();
		image = cv::imread(fileNames[i]);
		image.copyTo(temp);

		// Loop indefinitely while user draws until 'esc' is pressed
		unsigned int fileCount = i + 1;
		while (true) {
			image.copyTo(temp);
			if (drawingBox) {
				drawCircle(temp, boundingBox);
			}
			cv::imshow(windowName, temp);

			if (cv::waitKey(15) == 32) {
				cv::imwrite("SAM1_sub" + std::to_string(fileCount) + "_MARKED.jpg", image);
				break;
			}
		}

		// Generate ground truth CSV file and calculate its confusion matrix
		generateGTCSV("SAM_sub" + std::to_string(fileCount) + "_gt.csv", imageResults);
		std::cout << "\n\nConfusion matrix for image " + std::to_string(fileCount) << std::endl;
		calcConfusionMatrix("SAM1_sub2_alg.csv", imageResults, "confusion_matrix_img" + std::to_string(fileCount) + ".csv");
	}

	return 0;
}