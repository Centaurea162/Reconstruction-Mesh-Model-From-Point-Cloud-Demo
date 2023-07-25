#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
/*** insert any necessary libigl headers here ***/
#include <igl/copyleft/marching_cubes.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/per_face_normals.h>
#include <imgui/imgui.h>
#include <vector>

using namespace std;
using Viewer = igl::opengl::glfw::Viewer;

// Input: imported points, #P x3
Eigen::MatrixXd P;

// Input: imported normals, #P x3
Eigen::MatrixXd N;

// Intermediate result: constrained points, #C x3
Eigen::MatrixXd constrained_points;

// Intermediate result: implicit function values at constrained points, #C x1
Eigen::VectorXd constrained_values;

// Intermediate result: color at constrained points, #C x3
Eigen::MatrixXd constrained_color;

// Intermediate result: index of points bais on partition, #C x2
Eigen::MatrixXi points_index;

// Parameter: degree of the polynomial
unsigned int polyDegree = 1;

// Parameter: Wendland weight function radius (make this relative to the size of the mesh)
double wendlandRadius = 0.8;

// Parameter: grid resolution
unsigned int resolution = 20;

// Parameter: spatia resolution
unsigned int spatialResolution = 10;

// Intermediate result: grid points, at which the implicit function will be evaluated, #G x3
Eigen::MatrixXd grid_points;

// Intermediate result: implicit function values at the grid points, #G x1
Eigen::VectorXd grid_values;

// Intermediate result: grid point colors, for display, #G x3
Eigen::MatrixXd grid_colors;

// Intermediate result: grid lines, for display, #L x6 (each row contains
// starting and ending point of line segment)
Eigen::MatrixXd grid_lines;

// Output: vertex array, #V x3
Eigen::MatrixXd V;

// Output: face array, #F x3
Eigen::MatrixXi F;

// Output: face normals of the reconstructed mesh, #F x3
Eigen::MatrixXd FN;

// Functions
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);

void createConstrainedPoints();
void spatialIndexInit(int resolution1D);
int queryClosestPoint(int index);
vector<int> queryPoints(int index, double h);
void createGrid();
void evaluateImplicitFunc(double radius, int degree);
double wendlandFunc(int k,double r);
void getLines();
void alignModelWithAxes(double angle);
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);

void createConstrainedPoints() {
    // get epsilon
    Eigen::Vector3d m = P.colwise().minCoeff();
    Eigen::Vector3d M = P.colwise().maxCoeff();
    Eigen::Vector3d difference = M - m;
    double epsilon = difference.norm() * 0.01;

    // set constrained values
    constrained_values = Eigen::VectorXd(P.rows() * 3);
    for (int i = 0; i < P.rows(); i++) {
        constrained_values(i) = 0;
        constrained_values(i + P.rows()) = epsilon;
        constrained_values(i + P.rows() * 2) = -epsilon;
    }

    // set constrained points and their color
    constrained_points = Eigen::MatrixXd(P.rows() * 3, P.cols());
    constrained_color = Eigen::MatrixXd(P.rows() * 3, P.cols());
    for (int i = 0; i < P.rows(); i++) {
        // GREEN
        constrained_points.row(i)                   = P.row(i);
        constrained_color.row(i)                    = Eigen::Vector3d(0.0, 1.0, 0.0);
        // RED
        constrained_points.row(i + P.rows())        = P.row(i) + epsilon * N.row(i);
        constrained_color.row(i + P.rows())         = Eigen::Vector3d(1.0, 0.0, 0.0);
        // BLUE
        constrained_points.row(i + P.rows() * 2)    = P.row(i) - epsilon * N.row(i);
        constrained_color.row(i + P.rows() * 2)     = Eigen::Vector3d(0.0, 0.0, 1.0);
    }
};


void spatialIndexInit(int resolution1D = 10)
{
    int numOfPartition = resolution1D * resolution1D * resolution1D;

    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = P.colwise().minCoeff();
    bb_max = P.colwise().maxCoeff();

    // Bounding box dimensions
    Eigen::RowVector3d dim = bb_max - bb_min;

    // Grid spacing
    const double dx = dim[0] / (double)(resolution1D - 1);
    const double dy = dim[1] / (double)(resolution1D - 1);
    const double dz = dim[2] / (double)(resolution1D - 1);
    // 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    points_index.resize(constrained_points.rows(), 3);

    // set index
    for (int i = 0; i < constrained_points.rows(); i++) {
        int x = floor((constrained_points(i, 0) - bb_min[0]) / dx);
        int y = floor((constrained_points(i, 1) - bb_min[1]) / dy);
        int z = floor((constrained_points(i, 2) - bb_min[2]) / dz);
        if (x == -1)
            x = 0;
        if (y == -1)
            y = 0;
        if (z == -1)
            z = 0;
        points_index.row(i) = Eigen::Vector3i(x, y, z);
    }
    return;
}

// find the closest input point to q
int queryClosestPoint(int index){
    int res = -1;
    double minDis = 100000.0;
    int layer = 0;
    Eigen::Vector3i indexDiffer;
    while (res == -1) {
        for (int i = 0; i < points_index.rows(); i++) {
            //get the difference of spatial index
            indexDiffer = points_index.row(i) - points_index.row(index);
            //query
            if (abs(indexDiffer(0)) == layer && abs(indexDiffer(1)) == layer && abs(indexDiffer(2)) == layer) {
                Eigen::Vector3d differ = constrained_points.row(i) - constrained_points.row(index);
                double dis = differ.norm();
                if (dis < minDis) {
                    minDis = dis;
                    res = i;
                }
            }
        }
        layer++;
    }

    return res;
}

// find all input points within distance h of q
vector<int> queryPoints(int index, double h){
    vector<int> res;

    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = P.colwise().minCoeff();
    bb_max = P.colwise().maxCoeff();
    // Bounding box dimensions
    Eigen::RowVector3d dim = bb_max - bb_min;
    // Grid spacing
    const double dx = dim[0] / (double)(spatialResolution - 1);
    const double dy = dim[1] / (double)(spatialResolution - 1);
    const double dz = dim[2] / (double)(spatialResolution - 1);
    //bounding index
    int x = ceil(h / dx);
    int y = ceil(h / dy);
    int z = ceil(h / dz);;
    //query
    for (int i = 0; i < points_index.rows(); i++) {
        // get the difference of spatial index
        Eigen::Vector3i indexDiffer = points_index.row(i) - points_index.row(index);
        if (abs(indexDiffer(0)) <= x && abs(indexDiffer(1)) <= y && abs(indexDiffer(2)) <= z) {
            Eigen::Vector3d differ = constrained_points.row(i) - constrained_points.row(index);
            double dis = differ.norm();
            if (dis < h) {
                res.push_back(i);
            }
        }
    }

    return res;
}



// Creates a grid_points array for the simple sphere example. The points are
// stacked into a single matrix, ordered first in the x, then in the y and
// then in the z direction. If you find it necessary, replace this with your own
// function for creating the grid.
void createGrid()
{
    grid_points.resize(0, 3);
    grid_colors.resize(0, 3);
    grid_lines.resize(0, 6);
    grid_values.resize(0);
    V.resize(0, 3);
    F.resize(0, 3);
    FN.resize(0, 3);

    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = P.colwise().minCoeff();
    bb_max = P.colwise().maxCoeff();

    // Bounding box dimensions
    Eigen::RowVector3d dim = bb_max - bb_min;

    // Grid spacing
    const double dx = dim[0] / (double)(resolution - 1);
    const double dy = dim[1] / (double)(resolution - 1);
    const double dz = dim[2] / (double)(resolution - 1);
    // 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    grid_points.resize(resolution * resolution * resolution, 3);
    // Create each gridpoint
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                // 3D point at (x,y,z)
                grid_points.row(index) = bb_min + Eigen::RowVector3d(x * dx, y * dy, z * dz);
            }
        }
    }
}

// Function for explicitly evaluating the implicit function for a sphere of
// radius r centered at c : f(p) = ||p-c|| - r, where p = (x,y,z).
// This will NOT produce valid results for any mesh other than the given
// sphere.
// Replace this with your own function for evaluating the implicit function
// values at the grid points using MLS
void evaluateImplicitFunc(double wendRadius,int degree)
{
    // radius
    Eigen::RowVector3d bb_min = grid_points.colwise().minCoeff().eval();
    Eigen::RowVector3d bb_max = grid_points.colwise().maxCoeff().eval();
    double radius = wendRadius * (bb_max - bb_min).norm();

    // Scalar values of the grid points (the implicit function values)
    grid_values.resize(resolution * resolution * resolution);

    // get RBF
    Eigen::MatrixXd square = Eigen::MatrixXd::Zero(constrained_points.rows(), constrained_points.rows());
    vector<int> points;
    for (int i = 0; i < square.rows(); i++) {
        points = queryPoints(i, radius);
        for each (int var in points) {
            square(i, var) = wendlandFunc(degree, (constrained_points.row(i) - constrained_points.row(var)).norm());
        }
    };
    Eigen::VectorXd paramW = square.fullPivLu().solve(constrained_values);
    
    // Evaluate model's signed distance function at each gridpoint.
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                // get RBF terms
                Eigen::RowVectorXd row = Eigen::RowVectorXd(constrained_points.rows());
                for (int i = 0; i < constrained_points.rows(); i++) {
                    row(i) = wendlandFunc(degree, (grid_points.row(index) - constrained_points.row(i)).norm());
                }
                // Value at (x,y,z) = implicit function for the sphere
                grid_values[index] = row.dot(paramW);
            }
        }
    }
}


double wendlandFunc(int k, double r){
    switch (k) {
    case 0:
        return max(0.0, pow(1.0 - r, 2.0));
        break;
    case 1:
        return max(0.0, pow(1.0 - r, 4.0)) * (4.0 * r + 1.0) / 20.0;
        break;
    case 2:
        return max(0.0, pow(1.0 - r, 6.0)) * (35.0 * pow(r, 2.0) + 18.0 * r + 3.0) / 1680.0;
        break;
    case 3:
        return max(0.0, pow(1.0 - r, 8.0)) * (480.0 * pow(r, 3.0) + 375.0 * pow(r, 2.0) + 120.0 * r + 15.0) / 332640.0;
        break;
    case 4:
        return max(0.0, pow(1.0 - r, 10.0)) * (9009.0 * pow(r, 4.0) + 9450.0 * pow(r, 3.0) + 4410.0 * pow(r, 2.0) + 1050.0 * r + 105.0) / 121080960.0;
        break;
    default:
        return max(0.0, pow(1.0 - r, 2.0));
    }
}
// Code to display the grid lines given a grid structure of the given form.
// Assumes grid_points have been correctly assigned
// Replace with your own code for displaying lines if need be.
void getLines()
{
    int nnodes = grid_points.rows();
    grid_lines.resize(3 * nnodes, 6);
    int numLines = 0;

    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                int index = x + resolution * (y + resolution * z);
                if (x < resolution - 1) {
                    int index1 = (x + 1) + y * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (y < resolution - 1) {
                    int index1 = x + (y + 1) * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (z < resolution - 1) {
                    int index1 = x + y * resolution + (z + 1) * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
            }
        }
    }

    grid_lines.conservativeResize(numLines, Eigen::NoChange);
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers)
{
    if (key == '1') {
        // Show imported points
        viewer.data().clear();
        viewer.core().align_camera_center(P);
        viewer.data().point_size = 5;
        viewer.data().add_points(P, Eigen::RowVector3d(0, 0, 0));
    }

    if (key == '2') {
        // Show all constraints
        viewer.data().clear();
        viewer.core().align_camera_center(P);
        // Add your code for computing auxiliary constraint points here
        createConstrainedPoints();
        // Add code for displaying all points, as above
        viewer.data().point_size = 5;
        viewer.data().add_points(constrained_points, constrained_color);
    }

    if (key == '3') {
        // Show grid points with colored nodes and connected with lines
        viewer.data().clear();
        viewer.core().align_camera_center(P);
        // creating a grid
        createGrid();
        // computing auxiliary constraint points here
        createConstrainedPoints();

        spatialIndexInit(spatialResolution);
        // Evaluate implicit function
        evaluateImplicitFunc(wendlandRadius, polyDegree);
        // get grid lines
        getLines();

        // Code for coloring and displaying the grid points and lines
        // Assumes that grid_values and grid_points have been correctly assigned.
        grid_colors.setZero(grid_points.rows(), 3);

        // Build color map
        for (int i = 0; i < grid_points.rows(); ++i) {
            double value = grid_values(i);
            if (value < 0) {
                grid_colors(i, 1) = 1;
            } else {
                if (value > 0)
                    grid_colors(i, 0) = 1;
            }
        }

        // Draw lines and points
        viewer.data().point_size = 8;
        viewer.data().add_points(grid_points, grid_colors);
        viewer.data().add_edges(grid_lines.block(0, 0, grid_lines.rows(), 3),
            grid_lines.block(0, 3, grid_lines.rows(), 3),
            Eigen::RowVector3d(0.8, 0.8, 0.8));
        /*** end: sphere example ***/
        
    }

    if (key == '4') {
        // Show reconstructed mesh
        viewer.data().clear();
        // Code for computing the mesh (V,F) from grid_points and grid_values
        if ((grid_points.rows() == 0) || (grid_values.rows() == 0)) {
            cerr << "Not enough data for Marching Cubes !" << endl;
            return true;
        }
        // Run marching cubes
        igl::copyleft::marching_cubes(grid_values, grid_points, resolution, resolution, resolution, V, F);
        if (V.rows() == 0) {
            cerr << "Marching Cubes failed!" << endl;
            return true;
        }

        igl::per_face_normals(V, F, FN);
        viewer.data().set_mesh(V, F);
        viewer.data().show_lines = true;
        viewer.data().show_faces = true;
        viewer.data().set_normals(FN);
        igl::writeOFF("model.off", V, F);
    }

    if (key == '5') {
        // Show all constraints
        viewer.data().clear();
        viewer.core().align_camera_center(P);
        
        Eigen::MatrixXd axes = Eigen::MatrixXd(10 * 3, 3);
        Eigen::MatrixXd axesColors = Eigen::MatrixXd(10 * 3, 3);
        for (int i = 0; i < 10; i++) {
            axes.row(i)             = Eigen::Vector3d(1.0 + i, 0.0, 0.0);
            axesColors.row(i)       = Eigen::Vector3d(1.0, 0.0, 0.0);
            axes.row(i + 10)        = Eigen::Vector3d(0.0, 1.0 + i, 0.0);
            axesColors.row(i + 10)  = Eigen::Vector3d(0.0, 1.0, 0.0);
            axes.row(i + 20)        = Eigen::Vector3d(0.0, 0.0, 1.0 + i);
            axesColors.row(i + 20)  = Eigen::Vector3d(0.0, 0.0, 1.0);
        }

        // Add code for displaying all points, as above
        viewer.data().point_size = 5;
        viewer.data().add_points(axes, axesColors);
    }

    if (key == '6') {
        // Show imported points
        viewer.data().clear();
        viewer.core().align_camera_center(P);
        alignModelWithAxes(acos(-1) / 6.0);
        alignModelWithAxes(acos(-1) / 18.0);
        alignModelWithAxes(acos(-1) / 54.0);
        viewer.data().point_size = 5;
        viewer.data().add_points(P, Eigen::RowVector3d(0, 0, 0));
    }

    

    return true;
}

void alignModelWithAxes(double angle)
{
    // store the computing Intermediate points
    Eigen::MatrixXd tempP = Eigen::MatrixXd(P.rows(), 3);
    //final res
    double minVolume = std::numeric_limits<double>::max();
    Eigen::Matrix3d finalRotate;

    //enum all the combinataions
    for (int i = -2; i <= 2; i++) {
        Eigen::Matrix3d rotateX;
        if (i != 0) {
            rotateX << 
                1, 0,                  0,
                0, cos(angle * i), -sin(angle * i),
                0, sin(angle * i), cos(angle * i);
        } else {
            rotateX = Eigen::Matrix3d::Identity();
        }

        for (int j = -2; j <= 2; j++) {
            Eigen::Matrix3d rotateY;
            if (j != 0) {
                rotateY << 
                    cos(angle * j),  0, sin(angle * j),
                    0,               1, 0,
                    -sin(angle * j), 0, cos(angle * j);
            } else {
                rotateY = Eigen::Matrix3d::Identity();
            }

            for (int k = -2; k <= 2; k++) {
                Eigen::Matrix3d rotateZ;
                if (k != 0) {
                    rotateY << 
                        cos(angle * k), -sin(angle * k), 0,
                        sin(angle * k),  cos(angle * k), 0,
                        0,               0,              1;
                } else {
                    rotateZ = Eigen::Matrix3d::Identity();
                }

                Eigen::Matrix3d rotate = rotateX * rotateY * rotateZ;
                // rotate
                for (int i = 0; i < tempP.rows(); i++) {
                    tempP.row(i) = P.row(i) * rotate;
                }
                // get volume
                Eigen::Vector3d m = tempP.colwise().minCoeff();
                Eigen::Vector3d M = tempP.colwise().maxCoeff();
                Eigen::Vector3d difference = M - m;
                double volume = difference(0) * difference(1) * difference(2);

                if (volume < minVolume) {
                    minVolume = volume;
                    finalRotate = rotate;
                }


            }
        }
    }

    for (int i = 0; i < P.rows(); i++) {
        P.row(i) = P.row(i) * finalRotate;
        N.row(i) = N.row(i) * finalRotate;
    }

};

int main(int argc, char* argv[])
{
    if (argc != 2) {
        cout << "Usage ex2_bin mesh.off" << endl;
        exit(0);
    }

    // Read points and normals
    igl::readOFF(argv[1], P, F, N);

    // cout << P.rows() << "X" << P.cols() << endl;
    // cout << N.rows() << "X" << N.cols() << endl;

    Viewer viewer;
    viewer.callback_key_down = callback_key_down;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]() {
        menu.draw_viewer_menu();
        // Add widgets to the sidebar.
        if (ImGui::CollapsingHeader("Reconstruction Options", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputScalar("resolution", ImGuiDataType_U32, &resolution);
            ImGui::InputScalar("spatialResolution", ImGuiDataType_U32, &spatialResolution);
            ImGui::InputScalar("wendlandRadius", ImGuiDataType_Double, &wendlandRadius);
            ImGui::InputScalar("polyDegree", ImGuiDataType_U32, &polyDegree);
            if (ImGui::Button("Reset Grid", ImVec2(-1, 0))) {
                // Recreate the grid
                createGrid();
                // Switch view to show the grid
                callback_key_down(viewer, '3', 0);
            }

            if (ImGui::Button("Reset Spatial Index", ImVec2(-1, 0))) {
                spatialIndexInit(spatialResolution);
            }

            if (ImGui::Button("Reset wendlandRadius and polyDegree", ImVec2(-1, 0))) {
                // Switch view to show the grid
                callback_key_down(viewer, '3', 0);
            }



            // TODO: Add more parameters to tweak here...
        }
    };

    viewer.launch();
}
