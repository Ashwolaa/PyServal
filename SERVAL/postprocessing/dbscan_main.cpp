#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath> 
#include <chrono> //for measuring time
#include <iomanip> //for setprecision
#include <cstdio>
#include <cstring>


//struct for a point
struct Point {
    double shot; //shot id
    double x;
    double y;
    double z; //tof transformed to z coordinate
    double tof;
    double tot;
    double label; // Cluster label
    bool visited; // Flag to mark if the point has been visited during clustering
};

//struct for a cluster
struct Cluster {
    int id;          // Cluster ID
    double shot;     // Shot ID
    double avgX;     // Average x coordinate
    double avgY;     // Average y coordinate
    double avgTOF;   // Average tof
    double maxTOT;   // Maximum tot
    std::vector<Point> points;  // Points in the cluster
};



double interpolate(const std::vector<double>& xVals, const std::vector<double>& yVals, double x) {
    int i = 0;

    if (x >= xVals[xVals.size() - 2]) {
        i = xVals.size() - 2; //edge case for last element
    } else {
        // Find index i so that xVals[i] <= x < xVals[i+1]
        while (x > xVals[i + 1]) {
            i++;
        }
    }   

    //initialize variables for linear interpolation
    double xLow = xVals[i], yLow = yVals[i];
    double xHigh = xVals[i + 1], yHigh = yVals[i + 1];

    //throw error if x is outside of the range since no extrapolation handled currently
    /*
    if (x < xLow || x > xHigh) {
        std::cerr << "Error: x value outside of range for interpolation: " << x << std::endl;
        return 0;
    } 
    */

    // Catch if x is outside the found boundaries, we believe this is only for the very highest and lowest values in the dataset. 
    // If x is outside the range by being smaller, give the corresponding smaller y value
    if (x < xLow) {
        return yLow; 
        // if x is outside the range by being larger, give the correspondnig larger y value.
    } else if (x > xHigh) {
        return yHigh;

    }

    double dydx = (yHigh - yLow) / (xHigh - xLow); // compute slope

    return yLow + dydx * (x - xLow); // Linear interpolation
}


// Calculate Euclidean distance between two points
double calculateDistance(const Point& p1, const Point& p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                (p1.y - p2.y) * (p1.y - p2.y) +
                (p1.z - p2.z) * (p1.z - p2.z));
}

// Find neighboring points within the epsilon distance
std::vector<int> findNeighbors(const std::vector<Point>& points, int pointIndex, double epsilon) {
    std::vector<int> neighbors;
    for (int i = 0; i < points.size(); ++i) {
        if (i == pointIndex) continue;
        double distance = calculateDistance(points[pointIndex], points[i]);
        if (distance <= epsilon) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

// Expand a cluster starting from a seed point
void expandCluster(std::vector<Point>& points, int pointIndex, int cluster, double epsilon, int minPoints) {
    std::vector<int> seeds = findNeighbors(points, pointIndex, epsilon);
    if (seeds.size() < minPoints) {
        points[pointIndex].label = -1; // Mark as noise. Should already be marked by default, but just to be safe.
        return;
    }
    points[pointIndex].label = cluster;
    points[pointIndex].visited = true;

    
    for (int i = 0; i < seeds.size(); ++i) {
        int seedIndex = seeds[i];
        if (!points[seedIndex].visited) {
            expandCluster(points, seedIndex, cluster, epsilon, minPoints);
        }
    }
}


//dbscan algorithm
void dbscan(std::vector<Point>& points, double epsilon, int minPoints, int& cluster) {
    for (int i = 0; i < points.size(); ++i) {
        if (!points[i].visited) {
            expandCluster(points, i, cluster, epsilon, minPoints);
            if (points[i].label != -1) {
                cluster++;
            }
        }
    }
}



//########################################################################################
//########################################################################################
//########################################################################################

// Main function
int main(int argc, char* argv[]) {
    // Parse optional named arguments first
    double EPSILON = 2.0;       // Epsilon neighborhood distance threshold
    int MINPOINTS = 1;          // Min number of points required to form a cluster (SELF EXCLUDED)
    double TOFTHRESHOLD = 2e-4; // tof threshold for filtering points

    std::vector<const char*> positional_args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) {
            EPSILON = atof(argv[++i]);
        } else if (strcmp(argv[i], "--min-points") == 0 && i + 1 < argc) {
            MINPOINTS = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tof-threshold") == 0 && i + 1 < argc) {
            TOFTHRESHOLD = atof(argv[++i]);
        } else {
            positional_args.push_back(argv[i]);
        }
    }

    int npos = (int)positional_args.size();
    if (npos != 2 && npos != 3 && npos != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> [correction.txt] [labels_file]" << std::endl;
        std::cerr << "Options: --epsilon <val>  --min-points <val>  --tof-threshold <val>" << std::endl;
        return 1;
    }

    const char* inputFileName = positional_args[0];
    const char* outputFileName = positional_args[1];
    std::string correctionFile = "";
    std::string labelsFileName = "";

    if (npos == 3) {
        // 3 positional args: distinguish correction.txt vs labels file by extension
        std::string unknownFileName = positional_args[2];
        std::string fileEnding;
        std::string txt = (".txt");
        int delimPos = 0;

        delimPos = unknownFileName.find("."); // Locate the period in the file name
        fileEnding = unknownFileName.substr(delimPos, unknownFileName.npos);

        if (fileEnding.compare(txt) == 0) {
            correctionFile = unknownFileName;
        } else {
            labelsFileName = unknownFileName;
        }
    }

    if (npos == 4) {
        correctionFile = positional_args[2];
        labelsFileName = positional_args[3];
    }

    //tof parameters
    const double TOFTTRANSFORM = 81920*(25./4096)*1E-9; //weird chemistry science stuff magic number maybe??
    const double TOFTTOZ = EPSILON / TOFTTRANSFORM; //tof to z coordinate transformation factor

    const int DOUBLEPRECISSION = 17; //number of digits for double precission
    const int EXPECTEDNUMBERPOINTS = 5e5; //expected number of datapoints in input file

    
    //linear interpolation of correction file (if provided)
    std::vector<double> tofVals;
    std::vector<double> correctionVals;
    if (correctionFile.empty() == 0) { //.empty returns false if the string length is not 0
        std::ifstream file(correctionFile);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open the file: " << correctionFile << std::endl;
            return 1;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double tof;
            double correction;
            char delim; // Variable to store the delimiter (,) character
            //read correction file assuming format: tof,correction and no header
            if (iss >> tof >> delim >> correction) {
                tofVals.push_back(tof);
                correctionVals.push_back(correction);
            } else {
                std::cerr << "Error: Invalid data format in line: " << line << std::endl;
            }
        }
    }

    // Open the input file
    std::ifstream inputFile;
    inputFile.open(inputFileName, std::ios::in | std::ios::binary);

    //Check if file opened succesfully
    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open the input file: " << inputFileName << std::endl;
        return 1;
    }

    // Get file size; each record is exactly 5 x double = 40 bytes
    inputFile.seekg(0, std::ios::end);
    long long fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);
    int lastReadProgress = -1;

    long long nRecords = fileSize / 40;

    // Chunked read: process CHUNK_RECORDS records per I/O call.
    // Avoids per-record overhead while keeping memory use fixed (~40 MB/chunk).
    struct RawRecord { double shot, x, y, tof, tot; };
    const long long CHUNK_RECORDS = 1024 * 1024; // 1 M records = 40 MB
    std::vector<RawRecord> chunk(CHUNK_RECORDS);

    std::vector<Point> points;
    points.reserve(nRecords);

    std::cout << "PHASE: reading" << std::endl;
    std::cout.flush();

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    double shotOffset = 0.0;
    bool firstShot = true;
    long long recordsDone = 0;

    while (recordsDone < nRecords) {
        long long toRead = std::min(CHUNK_RECORDS, nRecords - recordsDone);
        inputFile.read(reinterpret_cast<char*>(chunk.data()), toRead * 40);

        for (long long i = 0; i < toRead; i++) {
            const RawRecord& r = chunk[i];

            // Emit reading progress (maps to 0-25%)
            int readPct = (int)((recordsDone + i) * 25 / nRecords);
            if (readPct != lastReadProgress) {
                std::cout << "PROGRESS: " << readPct << std::endl;
                std::cout.flush();
                lastReadProgress = readPct;
            }

            if (r.tof > TOFTHRESHOLD) continue;

            Point point;
            point.shot    = r.shot;
            point.x       = r.x;
            point.y       = r.y;
            point.tof     = r.tof;
            point.tot     = r.tot;
            point.z       = r.tof * TOFTTOZ;
            point.label   = -1;
            point.visited = false;

            if (firstShot) { shotOffset = r.shot; firstShot = false; }
            point.shot = r.shot - shotOffset + 1;

            if (correctionFile.empty() == 0) {
                double tofCorrection = interpolate(tofVals, correctionVals, point.tot);
                point.tof -= tofCorrection;
            }
            points.push_back(point);
        }
        recordsDone += toRead;
    }
    inputFile.close();


    auto finish = std::chrono::high_resolution_clock::now(); //measure time for reading input file
    std::chrono::duration<double> elapsed = finish - start; //measure time for reading input file
    std::cout << "Elapsed time for reading and processing input file: " << elapsed.count()*1000 << " ms\n"; //measure time for reading input file
    std::cout << "PHASE: grouping" << std::endl;
    std::cout << "PROGRESS: 25" << std::endl;
    std::cout.flush();


    start = std::chrono::high_resolution_clock::now(); //measure time for grouping data
    std::map<double, std::vector<Point>> shotToPoints;

    // Group points by 'shot'
    for (const Point& point : points) {
        shotToPoints[point.shot].push_back(point);
    }

    finish = std::chrono::high_resolution_clock::now(); //measure time for grouping data
    elapsed = finish - start; //measure time for grouping data
    std::cout << "Elapsed time for grouping points: " << elapsed.count()*1000 << " ms\n"; //measure time for grouping data
    std::cout << "PHASE: dbscan" << std::endl;
    std::cout << "PROGRESS: 30" << std::endl;
    std::cout.flush();

    start = std::chrono::high_resolution_clock::now(); //measure time for DBSCAN algorithm

    // Perform DBSCAN on each shot, emitting progress (30-90%)
    int clusterId = 0;
    int totalShots = (int)shotToPoints.size();
    int shotsDone = 0;
    int lastDbscanProgress = 30;
    for (auto& pair : shotToPoints) {
        dbscan(pair.second, EPSILON, MINPOINTS, clusterId);
        shotsDone++;
        int dbscanPct = 30 + (totalShots > 0 ? (shotsDone * 60) / totalShots : 60);
        if (dbscanPct >= lastDbscanProgress + 5) {
            std::cout << "PROGRESS: " << dbscanPct << std::endl;
            std::cout.flush();
            lastDbscanProgress = dbscanPct;
        }
    }

    finish = std::chrono::high_resolution_clock::now(); //measure time for DBSCAN algorithm
    elapsed = finish - start; //measure time for DBSCAN algorithm
    std::cout << "Elapsed time for DBSCAN: " << elapsed.count()*1000 << " ms\n"; //measure time for DBSCAN algorithm
    std::cout << "PHASE: clustering" << std::endl;
    std::cout << "PROGRESS: 90" << std::endl;
    std::cout.flush();

    start = std::chrono::high_resolution_clock::now(); //measure time for cluster processing

    // Create a map to accumulate the clusters
    std::map<int, Cluster> clusters; // Map cluster ID to Cluster structure
    // Loop through the points and assign them to clusters based on their labels
    for (const auto& pair : shotToPoints) {
        double shot = pair.first;
        for (const Point& point : pair.second) {
            if (point.label != -1) { // Ignore noise points
                int clusterLabel = point.label;
                // Check if the cluster exists in the map, and if not, create it
                if (clusters.find(clusterLabel) == clusters.end()) {
                    clusters[clusterLabel].id = clusterLabel; // Initialize cluster ID
                    clusters[clusterLabel].shot = shot; // Initialize shot
                    clusters[clusterLabel].maxTOT = point.tot; // Initialize maxTOT
                }
                clusters[point.label].points.push_back(point); // Add point to cluster
                // Update the maximum tot value for the cluster
                if (point.tot > clusters[point.label].maxTOT) {
                    clusters[point.label].maxTOT = point.tot;
                }
            }
        }
    }

    
    // Calculate averages and for each cluster
    for (auto& pair : clusters) {
        Cluster& cluster = pair.second;
        double sumX = 0.0, sumY = 0.0, sumTOF = 0.0, sumWeight = 0.0; //initialize variables for weighted average

        if (cluster.points.empty()) {continue;} //skip empty clusters

        for (const Point& point : cluster.points) {
            sumX += point.x * point.tot;
            sumY += point.y * point.tot;
            sumTOF += point.tof * point.tot;
            sumWeight += point.tot; //needed for weighted average
        }
        // Calculate weighted averages
        cluster.avgX = sumX / sumWeight;
        cluster.avgY = sumY / sumWeight;
        cluster.avgTOF = sumTOF / sumWeight;

    }
    finish = std::chrono::high_resolution_clock::now(); //measure time for cluster processing
    elapsed = finish - start; //measure time for cluster processing


    std::cout << "Elapsed time for cluster processing: " << elapsed.count()*1000 << " ms\n"; //measure time for cluster processing



    start = std::chrono::high_resolution_clock::now(); //measure time for writing output file


    // Check if the user has requested the .toflabels output
    if (labelsFileName.empty() == 0) { //.empty returns false if the string length is not 0
        
        // Open output file for writing
        std::ofstream labelsFile(labelsFileName, std::ios::out | std::ios::binary);

        if (!labelsFile.is_open()) {
        std::cerr << "Error: Could not open the labels output file: " << labelsFileName << std::endl;
        return 1;
        }

        // Write out labels in .toflabels format
        for (auto& pair : shotToPoints) {
            for (auto& point : pair.second) {
                labelsFile.write(reinterpret_cast<char*>(&point.shot), sizeof(point.shot));
                labelsFile.write(reinterpret_cast<char*>(&point.x), sizeof(point.x));
                labelsFile.write(reinterpret_cast<char*>(&point.y), sizeof(point.y));
                labelsFile.write(reinterpret_cast<char*>(&point.tof), sizeof(point.tof));
                labelsFile.write(reinterpret_cast<char*>(&point.tot), sizeof(point.tot));
                labelsFile.write(reinterpret_cast<char*>(&point.label), sizeof(point.label));
            }
        }

        finish = std::chrono::high_resolution_clock::now(); //measure time for writing output file
        elapsed = finish - start; //measure time for writing output file
        std::cout << "Elapsed time for writing labels file: " << elapsed.count() * 1000 << " ms\n"; //measure time for writing output file
        std::cout << "Labels output written to " << labelsFileName << std::endl;
        std::cout << "PROGRESS: 100" << std::endl;
        std::cout.flush();
    }
    else {
        // Open output file for writing
        std::ofstream outputFile(outputFileName, std::ios::out | std::ios::binary);

        if (!outputFile.is_open()) {
            std::cerr << "Error: Could not open the output file: " << outputFileName << std::endl;
            return 1;
        }
        // Write data to output file by looping over cluster shots
        for (auto& pair : clusters) {
            Cluster cluster = pair.second;
            outputFile.write(reinterpret_cast<char*>(&cluster.shot), sizeof(cluster.shot));
            outputFile.write(reinterpret_cast<char*>(&cluster.avgX), sizeof(cluster.avgX));
            outputFile.write(reinterpret_cast<char*>(&cluster.avgY), sizeof(cluster.avgY));
            outputFile.write(reinterpret_cast<char*>(&cluster.avgTOF), sizeof(cluster.avgTOF));
            outputFile.write(reinterpret_cast<char*>(&cluster.maxTOT), sizeof(cluster.maxTOT));
        }

        finish = std::chrono::high_resolution_clock::now(); //measure time for writing output file
        elapsed = finish - start; //measure time for writing output file
        std::cout << "Elapsed time for writing output file: " << elapsed.count() * 1000 << " ms\n"; //measure time for writing output file
        std::cout << "Output written to " << outputFileName << std::endl;
        std::cout << "PROGRESS: 100" << std::endl;
        std::cout.flush();
    }

    return 0;
}