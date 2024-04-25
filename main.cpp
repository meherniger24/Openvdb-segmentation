#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/GridTransformer.h>
#include <iostream>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/MeshToVolume.h>

#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/LevelsetFilter.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/ChangeBackground.h>
#include <limits>
#include <openvdb/tools/Morphology.h> 
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/ValueTransformer.h>
#include <cmath>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/Morphology.h> 
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/tools/PointIndexGrid.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

//template <class GridType>
//void makeNpyFile(GridType& grid, const std::string& npyFilePath)
//{
//    using ValueT = typename GridType::ValueType;
//
//    typename GridType::Accessor accessor = grid.getAccessor();
//
//    openvdb::Coord dim = grid.evalActiveVoxelDim();
//
//    openvdb::Coord ijk;
//    for (ijk[0] = 0; ijk[0] < dim[0]; ++ijk[0]) {
//        for (ijk[1] = 0; ijk[1] < dim[1]; ++ijk[1]) {
//            for (ijk[2] = 0; ijk[2] < dim[2]; ++ijk[2]) {
//                ValueT value = accessor.getValue(ijk);
//                accessor.setValue(ijk, value);
//            }
//        }
//    }
//
//    tira::volume<float> npyVolume;
//   // npyVolume.load(grid);  // Load grid data into tira::volume
//
//    npyVolume.save_npy(npyFilePath);
//
//    std::cout << "Numpy file saved: " << npyFilePath << "\n";
//}
using namespace openvdb;

// image to vdb conversion
template<class GridType> void img2vdb(GridType& grid, tira::image<float>& c, int t = 255) {
    using ValueT = typename GridType::ValueType;

    typename GridType::Accessor accessor = grid.getAccessor();

    openvdb::Coord ij;
    int& i = ij[0], & j = ij[1];
    for (i = 0; i < c.width(); ++i) {
        for (j = 0; j < c.height(); ++j) {
            float pixel = c(i, j);
            ValueT val = ValueT(pixel);

            if (pixel <= t) {
                accessor.setValue(ij, val);
            }
        }
    }

    openvdb::tools::signedFloodFill(grid.tree());
}


//vdb to image conversion
template<class GridType> void vdb2img(GridType& grid, tira::image<float>& c) {
    using ValueT = typename GridType::ValueType;

    typename GridType::Accessor accessor = grid.getAccessor();

    openvdb::Coord dim = grid.evalActiveVoxelDim();

    openvdb::Coord ij;
    int& i = ij[0], & j = ij[1];
    for (i = 0; i < dim[0]; ++i) {
        for (j = 0; j < dim[1]; ++j) {
            float pixel = (float)accessor.getValue(ij);
            c(i, j) = pixel;
        }
    }


}


//vdb to image conversion for 3D
template<class GridType> void vdb2img3D(GridType& grid, tira::volume<float>& img) {
    using ValueT = typename GridType::ValueType;

    typename GridType::Accessor accessor = grid.getAccessor();

    //openvdb::Coord dim = grid.evalActiveVoxelDim();
    openvdb::Coord ijk;
    int& i = ijk[0], & j = ijk[1], & k = ijk[2];
    for (i = 0; i < img.X(); i++) {
        for (j = 0; j < img.Y(); j++) {
            for (k = 0; k < img.Z(); k++) {
                float pixel = (float)accessor.getValue(ijk);
                img(i, j, k) = pixel;
            }
        }
    }
}


//vdb to npy image conversion
template<class GridType> void vdb2img22(GridType& grid, tira::image<float>& c) {
    using ValueT = typename GridType::ValueType;

    typename GridType::Accessor accessor = grid.getAccessor();

    openvdb::Coord dim = grid.evalActiveVoxelDim();

    openvdb::Coord ijk;
    int& i = ijk[0], & j = ijk[1], & k = ijk[2]; // Added z-index k if necessary
    for (i = 0; i < dim[0]; ++i) {
        for (j = 0; j < dim[1]; ++j) {
            k = 0; // Set k to the appropriate slice if the grid is 3D
            ValueT value = accessor.getValue(openvdb::Coord(i, j, k)); // Ensure proper Coord usage
            float pixel = static_cast<float>(value);
            c(i, j) = pixel;
            // Optionally add a check to print out values to ensure they are gradients
            std::cout << "Gradient at (" << i << ", " << j << "): " << pixel << std::endl;
        }
    }
}



template <class GridType>
void vdb2ims2(GridType& grid, tira::volume<float>& cs) {
    using ValueT = typename GridType::ValueType;
    using FloatT = float;  // Target type for the conversion

    typename GridType::Accessor accessor = grid.getAccessor();
    openvdb::Coord dim = grid.evalActiveVoxelDim();

    // Find the minimum and maximum values in the grid
    ValueT minValue = std::numeric_limits<ValueT>::max();
    ValueT maxValue = std::numeric_limits<ValueT>::lowest();
    for (typename GridType::ValueOnCIter iter = grid.cbeginValueOn(); iter; ++iter) {
        minValue = std::min(minValue, iter.getValue());
        maxValue = std::max(maxValue, iter.getValue());
    }

    // Perform value normalization and conversion
    FloatT minFloatValue = static_cast<FloatT>(minValue);
    FloatT maxFloatValue = static_cast<FloatT>(maxValue);
    FloatT range = maxFloatValue - minFloatValue;

    openvdb::Coord ijk;
    int& i = ijk[0], & j = ijk[1], & k = ijk[2];
    for (i = 0; i < dim[0]; ++i) {
        for (j = 0; j < dim[1]; ++j) {
            for (k = 0; k < dim[2]; ++k) {
                ValueT voxelValue = accessor.getValue(ijk);
                FloatT normalizedValue = (static_cast<FloatT>(voxelValue) - minFloatValue) / range;
                cs(i, j, k) = normalizedValue;
            }
        }
    }
}


// Heaviside function
template <class GridType>
void heaviside(GridType& grid)
{
    float epsilon = 0.2;
    // Get accessor for the grid
    typename GridType::Accessor accessor = grid.getAccessor();
    // Iterate over the grid's active values
    for (typename GridType::ValueOnIter iter = grid.beginValueOn(); iter; ++iter) {
        // Get the current value
        const typename GridType::ValueType value = iter.getValue();
        // Calculate the Heaviside function value
        const typename GridType::ValueType heaviside = (0.5 * (1 + (2 / 3.14159265358979323846) * atan(value / epsilon)));
        accessor.setValue(iter.getCoord(), heaviside);
    }
}



// derivative of Heaviside function
template <class GridType>
void deri_heaviside(GridType& grid)
{
    float epsilon = 0.2;

    typename GridType::Accessor accessor = grid.getAccessor();

    for (typename GridType::ValueOnIter iter = grid.beginValueOn(); iter; ++iter) {
        const typename GridType::ValueType value = iter.getValue();
        const typename GridType::ValueType deri_heaviside = (1 / 3.14159265358979323846) * (epsilon / ((epsilon * epsilon) + (value * value)));
        accessor.setValue(iter.getCoord(), deri_heaviside);
    }
}



// inverse a vdb file
template <class GridType>
void vdb_inverse(GridType& grid)
{
    typename GridType::Accessor accessor = grid.getAccessor();

    for (typename GridType::ValueOnIter iter = grid.beginValueOn(); iter; ++iter) {
        const typename GridType::ValueType value = iter.getValue();
        const typename GridType::ValueType inverse_vdb = (1 - value);
        accessor.setValue(iter.getCoord(), inverse_vdb);
    }
}


// Function to multiply two VDB grids
openvdb::FloatGrid::Ptr multiplyGrids(openvdb::FloatGrid::Ptr& grid1, openvdb::FloatGrid::Ptr& grid2, float& backgroundvalue) {
    // Create a new FloatGrid to store the result
    openvdb::FloatGrid::Ptr resultGrid = openvdb::FloatGrid::create(backgroundvalue);

    // Get accessors for the grids
    openvdb::FloatGrid::Accessor resultAccessor = resultGrid->getAccessor();
    openvdb::FloatGrid::Accessor grid1Accessor = grid1->getAccessor();
    openvdb::FloatGrid::Accessor grid2Accessor = grid2->getAccessor();

    // Iterate over the active values of the first grid
    for (openvdb::FloatGrid::ValueOnIter iter = grid1->beginValueOn(); iter; ++iter) {
        const openvdb::Coord& coord = iter.getCoord();

        // Multiply the values from grid1 and grid2, and set in resultGrid
        resultAccessor.setValue(coord, grid1Accessor.getValue(coord) * grid2Accessor.getValue(coord));
    }

    return resultGrid;
}

// Function to divide two VDB grids
openvdb::FloatGrid::Ptr divideGrids(openvdb::FloatGrid::Ptr& grid1, openvdb::FloatGrid::Ptr& grid2) {
    // Create a new FloatGrid to store the result
    openvdb::FloatGrid::Ptr resultGrid = openvdb::FloatGrid::create();
    resultGrid = grid1->deepCopy();

    // Get accessors for the grids
    openvdb::FloatGrid::Accessor resultAccessor = resultGrid->getAccessor();
    openvdb::FloatGrid::Accessor grid1Accessor = grid1->getAccessor();
    openvdb::FloatGrid::Accessor grid2Accessor = grid2->getAccessor();

    // Iterate over the active values of the first grid
    for (openvdb::FloatGrid::ValueOnIter iter = grid1->beginValueOn(); iter; ++iter) {
        const openvdb::Coord& coord = iter.getCoord();

        // Get the value from the second grid
        float grid2Value = grid2Accessor.getValue(coord);

        // Check if the divisor is not zero to avoid division by zero
        if (grid2Value != 0.0f) {
            // Divide the values from grid1 by grid2, and set in resultGrid
            resultAccessor.setValue(coord, grid1Accessor.getValue(coord) / grid2Value);
        }
        else {
            // Handle division by zero if necessary, e.g., set to a default value
            resultAccessor.setValue(coord, 0.0f); // or some other default value
        }
    }

    return resultGrid;
}


void computeSecondOrderGradients(openvdb::FloatGrid::Ptr& grid, openvdb::FloatGrid::Ptr gradMagnitude, openvdb::FloatGrid::Ptr& d2PHI_dx2, openvdb::FloatGrid::Ptr& d2PHI_dy2, openvdb::FloatGrid::Ptr& d2PHI_dz2) {
    // Assuming grid is your sdf_grid and gradMagnitude is precomputed
    // Compute gradients to get first-order derivatives
    auto gradX = openvdb::tools::gradient(*grid); // dPHI_dx
    // Convert Vec3fGrid (VectorGrid) to three FloatGrids for x, y, z components
    openvdb::FloatGrid::Ptr gradX_x = openvdb::FloatGrid::create(0.0);
    openvdb::FloatGrid::Ptr gradX_y = openvdb::FloatGrid::create(0.0);
    openvdb::FloatGrid::Ptr gradX_z = openvdb::FloatGrid::create(0.0);

    for (openvdb::VectorGrid::ValueOnCIter iter = gradX->cbeginValueOn(); iter; ++iter) {
        openvdb::Vec3f grad = iter.getValue();
        openvdb::Coord xyz = iter.getCoord();
        gradX_x->tree().setValue(xyz, grad.x());
        gradX_y->tree().setValue(xyz, grad.y());
        gradX_z->tree().setValue(xyz, grad.z());
    }

    // Compute second-order gradients
    auto gradGradX_x = openvdb::tools::gradient(*gradX_x);
    auto gradGradY_y = openvdb::tools::gradient(*gradX_y);
    auto gradGradZ_z = openvdb::tools::gradient(*gradX_z);

    // Initialize second-order derivative grids
    d2PHI_dx2 = openvdb::FloatGrid::create(0.0);
    d2PHI_dy2 = openvdb::FloatGrid::create(0.0);
    d2PHI_dz2 = openvdb::FloatGrid::create(0.0);

    // Extract and store second-order derivatives
    for (openvdb::VectorGrid::ValueOnCIter iter = gradGradX_x->cbeginValueOn(); iter; ++iter) {
        d2PHI_dx2->tree().setValue(iter.getCoord(), iter.getValue().x());
    }
    for (openvdb::VectorGrid::ValueOnCIter iter = gradGradY_y->cbeginValueOn(); iter; ++iter) {
        d2PHI_dy2->tree().setValue(iter.getCoord(), iter.getValue().y());
    }
    for (openvdb::VectorGrid::ValueOnCIter iter = gradGradZ_z->cbeginValueOn(); iter; ++iter) {
        d2PHI_dz2->tree().setValue(iter.getCoord(), iter.getValue().z());
    }
}


openvdb::FloatGrid::Ptr calculateGradientX(const openvdb::FloatGrid::Ptr& grid) {
   /* openvdb::FloatGrid::Ptr gradXGrid = openvdb::FloatGrid::create(0.0);
    gradXGrid->setTransform(grid->transform().copy());*/

    openvdb::FloatGrid::Ptr gradXGrid = openvdb::FloatGrid::create();
    gradXGrid = grid->deepCopy();

    openvdb::FloatGrid::ConstAccessor accessor = grid->getConstAccessor();

    for (openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter; ++iter) {
        const openvdb::Coord& xyz = iter.getCoord();
        float valueLeft = accessor.isValueOn(xyz.offsetBy(-1, 0, 0)) ? accessor.getValue(xyz.offsetBy(-1, 0, 0)) : iter.getValue();
        float valueRight = accessor.isValueOn(xyz.offsetBy(1, 0, 0)) ? accessor.getValue(xyz.offsetBy(1, 0, 0)) : iter.getValue();
        float gradX = (valueRight - valueLeft) / 2.0f;
        gradXGrid->tree().setValueOn(xyz, gradX);
    }

    return gradXGrid;
}

openvdb::FloatGrid::Ptr calculateGradientY(const openvdb::FloatGrid::Ptr& grid) {
    /*openvdb::FloatGrid::Ptr gradYGrid = openvdb::FloatGrid::create(0.0);
    gradYGrid->setTransform(grid->transform().copy());*/

    openvdb::FloatGrid::Ptr gradYGrid = openvdb::FloatGrid::create();
    gradYGrid = grid->deepCopy();

    openvdb::FloatGrid::ConstAccessor accessor = grid->getConstAccessor();

    for (openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter; ++iter) {
        const openvdb::Coord& xyz = iter.getCoord();
        float valueDown = accessor.isValueOn(xyz.offsetBy(0, -1, 0)) ? accessor.getValue(xyz.offsetBy(0, -1, 0)) : iter.getValue();
        float valueUp = accessor.isValueOn(xyz.offsetBy(0, 1, 0)) ? accessor.getValue(xyz.offsetBy(0, 1, 0)) : iter.getValue();
        float gradY = (valueUp - valueDown) / 2.0f;
        gradYGrid->tree().setValueOn(xyz, gradY);
    }

    return gradYGrid;
}

openvdb::FloatGrid::Ptr calculateGradientZ(const openvdb::FloatGrid::Ptr& grid) {
    /*openvdb::FloatGrid::Ptr gradZGrid = openvdb::FloatGrid::create(0.0);
    gradZGrid->setTransform(grid->transform().copy());*/
    openvdb::FloatGrid::Ptr gradZGrid = openvdb::FloatGrid::create();
    gradZGrid = grid->deepCopy();

    openvdb::FloatGrid::ConstAccessor accessor = grid->getConstAccessor();

    for (openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter; ++iter) {
        const openvdb::Coord& xyz = iter.getCoord();
        float valueBelow = accessor.isValueOn(xyz.offsetBy(0, 0, -1)) ? accessor.getValue(xyz.offsetBy(0, 0, -1)) : iter.getValue();
        float valueAbove = accessor.isValueOn(xyz.offsetBy(0, 0, 1)) ? accessor.getValue(xyz.offsetBy(0, 0, 1)) : iter.getValue();
        float gradZ = (valueAbove - valueBelow) / 2.0f;
        gradZGrid->tree().setValueOn(xyz, gradZ);
    }

    return gradZGrid;
}



openvdb::FloatGrid::Ptr calculateGradientMagnitude(
    const openvdb::FloatGrid::Ptr& gradXGrid,
    const openvdb::FloatGrid::Ptr& gradYGrid,
    const openvdb::FloatGrid::Ptr& gradZGrid) {

    /*openvdb::FloatGrid::Ptr gradMagGrid = openvdb::FloatGrid::create(0.0);
    gradMagGrid->setTransform(gradXGrid->transform().copy());*/

    openvdb::FloatGrid::Ptr gradMagGrid = openvdb::FloatGrid::create();
    gradMagGrid = gradXGrid->deepCopy();

    //  gradient grids
    openvdb::FloatGrid::ConstAccessor accessorX = gradXGrid->getConstAccessor();
    openvdb::FloatGrid::ConstAccessor accessorY = gradYGrid->getConstAccessor();
    openvdb::FloatGrid::ConstAccessor accessorZ = gradZGrid->getConstAccessor();

    
    for (openvdb::FloatGrid::ValueOnCIter iter = gradXGrid->cbeginValueOn(); iter; ++iter) {
        const openvdb::Coord& xyz = iter.getCoord();
        // for the current voxel
        float gradX = accessorX.getValue(xyz);
        float gradY = accessorY.getValue(xyz);
        float gradZ = accessorZ.getValue(xyz);
        //  magnitude 
        float magnitude = openvdb::math::Sqrt(gradX * gradX + gradY * gradY + gradZ * gradZ);
        // computed magnitude  in the output grid
        gradMagGrid->tree().setValueOn(xyz, magnitude);
    }

    return gradMagGrid;
}

void apply_threshold(openvdb::FloatGrid::Ptr grid, float minThreshold, float maxThreshold) {
    
    for (openvdb::FloatGrid::ValueOnIter iter = grid->beginValueOn(); iter; ++iter) {
        // if the voxel value is outside the threshold
        if (iter.getValue() < minThreshold || iter.getValue() > maxThreshold) {
            // Deactivate the voxel
            iter.setValueOff();
        }
    }
    grid->tree().prune(); // optimize grid structure after modifications
}


//void adjus_precision(openvdb::FloatGrid& grid) {
//    // lambda function 
//    auto adjust_value = [](const openvdb::FloatGrid::ValueAllIter& iter) {
//        const float scale = 1000.0f; // scaling 
//        // adjust the voxel value.
//        float new_value = std::round(iter.getValue() * scale) / scale;
//        const_cast<openvdb::FloatGrid::ValueAllIter&>(iter).setValue(new_value);
//    };
//
//    
//    openvdb::tools::foreach(grid.beginValueAll(), adjust_value);
//}

//to simulate lower precision
void adjust_precision(openvdb::FloatGrid::Ptr grid) {
    for (auto iter = grid->beginValueAll(); iter; ++iter) {
        float original_v = iter.getValue();
        
        const float scale_fac = 1000.0f; 
        float adjusted_v = std::round(original_v * scale_fac) / scale_fac;
        iter.setValue(adjusted_v);
    }
    grid->pruneGrid();
}

template <class GridType>
struct HeavisideFunctor {
    GridType& grid;
    float epsilon;

    HeavisideFunctor(GridType& grid, float epsilon) : grid(grid), epsilon(epsilon) {}

    void operator()(const tbb::blocked_range<size_t>& range) const {
        typename GridType::Accessor accessor = grid.getAccessor();
        for (size_t i = range.begin(); i != range.end(); ++i) {
            typename GridType::ValueType value = accessor.getValue(i);
            // Calculate the Heaviside function value
            typename GridType::ValueType heaviside =
                (0.5 * (1 + (2 / 3.14159265358979323846) * atan(value / epsilon)));
            accessor.setValue(i, heaviside);
        }
    }
};

template <class GridType>
void heaviside_p(GridType& grid) {
    float epsilon = 0.2;
    HeavisideFunctor<GridType> functor(grid, epsilon);
    // Define the range over which to parallelize
    tbb::parallel_for(tbb::blocked_range<size_t>(0, grid.activeVoxelCount()), functor);
}

int main()
{
    openvdb::initialize();

    // Load sdf grid from file
    openvdb::io::File file_sdf("3D_200_img.vdb");
    file_sdf.open();
    openvdb::GridBase::Ptr baseGrid_sdf;
    for (openvdb::io::File::NameIterator nameIter = file_sdf.beginName(); nameIter != file_sdf.endName(); ++nameIter) {
        if (nameIter.gridName() == "LevelSetSphere") {
            baseGrid_sdf = file_sdf.readGrid(nameIter.gridName());
            break;
        }
    }
    file_sdf.close();

    // Load input grid from file
    openvdb::io::File file_input("3D_200_img.vdb");
    file_input.open();
    openvdb::GridBase::Ptr baseGrid_input;
    for (openvdb::io::File::NameIterator nameIter = file_input.beginName(); nameIter != file_input.endName(); ++nameIter) {
        if (nameIter.gridName() == "LevelSetSphere") {
            baseGrid_input = file_input.readGrid(nameIter.gridName());
            break;
        }
    }
    file_input.close();


    // grids to FloatGrid
    openvdb::FloatGrid::Ptr sdf_grid1 = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid_sdf);
    openvdb::FloatGrid::Ptr input_grid1 = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid_input);

    /*size_t memUsage = sdf_grid1->memUsage();
    std::cout << "Memory usage: " << memUsage << " bytes" << std::endl;

    exit(1);*/

    float threshold = 110.0f;
    /*float upper_thresh = 255.0f;
    float lower_thresh = 0.0f;*/

    openvdb::FloatGrid::Ptr empty_grid = openvdb::FloatGrid::create();
    openvdb::FloatGrid::Ptr sdf_grid = openvdb::FloatGrid::create(1);
    openvdb::FloatGrid::Ptr input_grid = openvdb::FloatGrid::create(140);
    openvdb::FloatGrid::Ptr sdf_grid2 = openvdb::FloatGrid::create();
    
    for (openvdb::FloatGrid::ValueOnIter iter = input_grid1->beginValueOn(); iter; ++iter) {
        if (iter.getValue() <= threshold) {
            openvdb::Coord ijk = iter.getCoord();


            empty_grid->tree().setValueOn(ijk);
        }
    }

    /*for (openvdb::FloatGrid::ValueOnCIter iter = empty_grid->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord ijk = iter.getCoord();

        sdf_grid2->tree().setValueOn(ijk, 0.0f);
    }*/

    openvdb::tools::dilateActiveValues(empty_grid->tree(), 3);

    for (openvdb::FloatGrid::ValueOnCIter iter = input_grid1->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord ijk = iter.getCoord();

        if (empty_grid->tree().isValueOn(ijk)) {

            float value = input_grid1->tree().getValue(ijk);
            input_grid->tree().setValue(ijk, value);
        }

    }

    for (openvdb::FloatGrid::ValueOnCIter iter = input_grid->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord ijk = iter.getCoord();

        sdf_grid->tree().setValueOn(ijk);
    }


    for (openvdb::FloatGrid::ValueOnCIter iter = sdf_grid->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord ijk = iter.getCoord();

        if (sdf_grid1->tree().isValueOn(ijk)) {

            float value = sdf_grid1->tree().getValue(ijk);
            sdf_grid->tree().setValue(ijk, value);
        }

    }

   ///FOT HD IN

    for(openvdb::FloatGrid::ValueOnCIter iter = sdf_grid->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord ijk = iter.getCoord();

        if (sdf_grid->tree().isValueOn(ijk)) {

            float value = sdf_grid->tree().getValue(ijk);
            sdf_grid2->tree().setValue(ijk, value);
        }

    }


    /*for (openvdb::FloatGrid::ValueOnCIter iter = sdf_grid2->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord ijk = iter.getCoord();

        if (sdf_grid1->tree().isValueOn(ijk)) {

            float value = sdf_grid1->tree().getValue(ijk);
            sdf_grid2->tree().setValue(ijk, value);
        }

    }*/

    /*openvdb::FloatGrid::Ptr sdf_grid = openvdb::FloatGrid::create();
    openvdb::FloatGrid::Ptr input_grid = openvdb::FloatGrid::create();


    openvdb::FloatGrid::Ptr sdf_grid2 = openvdb::FloatGrid::create();


    for (openvdb::FloatGrid::ValueOnIter iter = input_grid1->beginValueOn(); iter; ++iter) {
        if (iter.getValue() <= threshold) {
            openvdb::Coord ijk = iter.getCoord();

            
            input_grid->tree().setValueOn(ijk, iter.getValue());
        }
    }

    for (openvdb::FloatGrid::ValueOnCIter iter = input_grid->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord ijk = iter.getCoord();

        sdf_grid2->tree().setValueOn(ijk, 0.0f);
    }


    for (openvdb::FloatGrid::ValueOnCIter iter = sdf_grid2->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord ijk = iter.getCoord();

        if (sdf_grid1->tree().isValueOn(ijk)) {

            float value = sdf_grid1->tree().getValue(ijk);
            sdf_grid2->tree().setValue(ijk, value);
        }

    }

    openvdb::tools::dilateActiveValues(input_grid->tree(), 3);

    for (openvdb::FloatGrid::ValueOnCIter iter = input_grid->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord ijk = iter.getCoord();
        
        sdf_grid->tree().setValueOn(ijk, 0.0f); 
    }

    
    for (openvdb::FloatGrid::ValueOnCIter iter = sdf_grid->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord ijk = iter.getCoord();
        
        if (sdf_grid1->tree().isValueOn(ijk)) {
            
            float value = sdf_grid1->tree().getValue(ijk);
            sdf_grid->tree().setValue(ijk, value);
        }
       
    }


    auto accessor = sdf_grid->getAccessor();

    for (openvdb::FloatGrid::ValueOnCIter iter = sdf_grid->cbeginValueOn(); iter; ++iter) {
        if (!sdf_grid2->tree().isValueOn(iter.getCoord())) {
            
            accessor.setValue(iter.getCoord(), 3.0f);
        }
    }*/
    /*auto accessor = sdf_grid->getAccessor();

    for (openvdb::FloatGrid::ValueOnCIter iter = sdf_grid->cbeginValueOn(); iter; ++iter) {
        if (!sdf_grid2->tree().isValueOn(iter.getCoord())) {

            accessor.setValue(iter.getCoord(), -3.0f);
        }
    }*/

    /*for (auto iter = input_grid->beginValueAll(); iter; ++iter) {
        if (!iter.isValueOn()) {
            iter.setValue(0.0f);
        }
    }

    for (auto iter = sdf_grid->beginValueAll(); iter; ++iter) {
        if (!iter.isValueOn()) {
            iter.setValue(140.0f);
        }
    }*/
    
    input_grid->tree().prune();
    sdf_grid->tree().prune();

    
    
    std::cout << "Total number of sdf voxel: " << sdf_grid1->activeVoxelCount() << std::endl;
    std::cout << "Total number of input voxel: " << input_grid->activeVoxelCount() << std::endl;

    /*float threshold_sdf = 13.0f;
    for (openvdb::FloatGrid::ValueOnIter iter = sdf_grid->beginValueOn(); iter; ++iter) {
        if (iter.getValue() >= threshold_sdf) {

            iter.setValueOff();
        }
    }*/
    
   

    /*tira::volume<float> I2(200, 200, 200);
    vdb2img3D(*input_grid, I2);
    I2.save_npy("C:/Users/meher/spyder/sdf.npy");
    exit(1);*/
    /*
    //empty destination grid.
    //openvdb::FloatGrid::Ptr sdf_grid = openvdb::FloatGrid::create();
    //openvdb::FloatGrid::Ptr input_grid = openvdb::FloatGrid::create();

    //
    ////// 1.0f to 2.0f, making the grid lower resolution.
    //input_grid->setTransform(openvdb::math::Transform::createLinearTransform(1.1f));
    //sdf_grid->setTransform(openvdb::math::Transform::createLinearTransform(1.1f));

    ////// rresample the original grid -- matching the destination grid's resolution -- box sampling.
    //openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*sdf_grid1, *sdf_grid);
    //openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*input_grid1, *input_grid);
    
    
    //adjust_precision(sdf_grid);
    //adjust_precision(input_grid);
    */
    /*float newVoxelSize = 5.0f; 
    
    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(newVoxelSize);

    openvdb::FloatGrid::Ptr resampledGrid = openvdb::FloatGrid::create();
    resampledGrid->setTransform(transform);
    openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*input_grid, *resampledGrid);

    openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*sdf_grid, *resampledGrid);*/
    /*
    float minThreshold = 10.0f; // Minimum value to keep
    float maxThreshold = 90.0f;  // Maximum value to keep

    // Accessor for the input grid
    openvdb::FloatGrid::Accessor accessor = input_grid->getAccessor();

    // Iterate over all voxels in the grid
    for (openvdb::FloatGrid::ValueOnIter iter = input_grid->beginValueOn(); iter; ++iter) {
        // Get the voxel value
        //const float voxelValue = iter.getValue();
        // Get the voxel value
        float voxelValue = iter.getValue();
        //openvdb::Coord voxelCoord = iter.getCoord();
        //if (voxelValue < 100) {
        //    // Print the voxel's coordinates and value
        //    std::cout << "Voxel at (" << voxelCoord.x() << ", " << voxelCoord.y() << ", " << voxelCoord.z() << ") has value: " << voxelValue << std::endl;
        //}
        // Check if the voxel value is outside the threshold
        if (voxelValue < minThreshold || voxelValue > maxThreshold) {
            // Set the voxel to be inactive or remove it based on your needs
            iter.setValueOff();
        }
    }
    */


    //exit(1);
    //openvdb::FloatGrid::Ptr highValueGrid = openvdb::FloatGrid::create(30.0);

    
    //highValueGrid->setTransform(sdf_grid->transform().copy());

    //// Get an accessor for the new grid.
    //openvdb::FloatGrid::Accessor accessor = highValueGrid->getAccessor();

    //// Iterate over all active voxels in the original grid and copy their values to the new grid.
    //for (openvdb::FloatGrid::ValueOnCIter iter = sdf_grid->cbeginValueOn(); iter; ++iter) {
    //    // Use the accessor to set the value of the voxel in the new grid.
    //    accessor.setValueOn(iter.getCoord(), *iter);
    //}
    
    //openvdb::Coord testCoord(60, 60, 60); 
    float backgroundvalue = 140.0;
    
    //float value = accessor.getValue(testCoord);
    //std::cout << "Value at " << testCoord << ": " << value << std::endl;
    openvdb::FloatGrid::Ptr Eout = openvdb::FloatGrid::create(backgroundvalue);
    openvdb::FloatGrid::Ptr Ein = openvdb::FloatGrid::create(backgroundvalue);
    openvdb::FloatGrid::Ptr DIVGrid = openvdb::FloatGrid::create(backgroundvalue);

    openvdb::FloatGrid::Ptr d2PHI_dx2_1 = openvdb::FloatGrid::create(backgroundvalue);
    openvdb::FloatGrid::Ptr d2PHI_dy2_1 = openvdb::FloatGrid::create(backgroundvalue);
    openvdb::FloatGrid::Ptr d2PHI_dz2_1 = openvdb::FloatGrid::create(backgroundvalue);

    openvdb::FloatGrid::Ptr norm_DIVgrid = openvdb::FloatGrid::create(backgroundvalue);


    int T = 13;
    for (int t = 0; t < T; t++) {
        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        ////Apply the heaviside function
        openvdb::FloatGrid::Ptr HD_out = sdf_grid->deepCopy();
        heaviside_p(*HD_out);

        auto finish = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

        exit(1);

        ////Apply the derivetive of heaviside function
        openvdb::FloatGrid::Ptr deri_heavi = sdf_grid->deepCopy();
        deri_heaviside(*deri_heavi);

        // multiplying  heaviside_image with input_image

        openvdb::FloatGrid::Ptr I_out = multiplyGrids(input_grid, HD_out, backgroundvalue);

        //inverse of heaviside image
        /*openvdb::FloatGrid::Ptr HD_in = HD_out->deepCopy();
        vdb_inverse(*HD_in);*/

        openvdb::FloatGrid::Ptr HD_in = sdf_grid2->deepCopy();
        heaviside(*HD_in);
        vdb_inverse(*HD_in);

        // multiplying  inverse_heaviside_image with input_image

        float b_hd_in = 0.0f;
        openvdb::FloatGrid::Ptr I_in = multiplyGrids(input_grid, HD_in, b_hd_in);

        

        // Apply Gaussian filter
        openvdb::FloatGrid::Ptr I_out_blurred = I_out->deepCopy();
        openvdb::FloatGrid::Ptr HD_out_blurred = HD_out->deepCopy();
        openvdb::tools::Filter<openvdb::FloatGrid> filter1(*I_out_blurred);
        filter1.setGrainSize(3);
        openvdb::tools::Filter<openvdb::FloatGrid> filter3(*HD_out_blurred);
        filter3.setGrainSize(3);

        double sigma = 3.0;
        //int kernelWidth = 1;
        filter1.gaussian( sigma);
        filter3.gaussian( sigma);

        // Apply Gaussian filter
        openvdb::FloatGrid::Ptr I_in_blurred = I_in->deepCopy();
        openvdb::FloatGrid::Ptr HD_in_blurred = HD_in->deepCopy();
        openvdb::tools::Filter<openvdb::FloatGrid> filter2(*I_in_blurred);
        filter2.setGrainSize(3);
        openvdb::tools::Filter<openvdb::FloatGrid> filter4(*HD_in_blurred);
        filter4.setGrainSize(3);

        filter2.gaussian(sigma);
        filter4.gaussian(sigma);

        //fout
        openvdb::FloatGrid::Ptr f_out = divideGrids(I_out_blurred, HD_out_blurred);

        //fin
        openvdb::FloatGrid::Ptr f_in = divideGrids(I_in_blurred, HD_in_blurred );
        openvdb::tools::changeBackground(f_in->tree(), backgroundvalue);
        
    /*
        // gradient of the sdf grid
        openvdb::VectorGrid::Ptr gradGrid = openvdb::tools::gradient(*sdf_grid);

        //  x and y and z of the gradient
        openvdb::FloatGrid::Ptr gradX = openvdb::FloatGrid::create(0.0);
        openvdb::FloatGrid::Ptr gradY = openvdb::FloatGrid::create(0.0);
        openvdb::FloatGrid::Ptr gradZ = openvdb::FloatGrid::create(0.0);

        const float scale = 1.0f; 

        for (openvdb::VectorGrid::ValueOnIter iter = gradGrid->beginValueOn(); iter; ++iter) {
            openvdb::Vec3f grad = iter.getValue();
            float length = grad.length();
            // in case the gradient vector is zero
            if (length > std::numeric_limits<float>::epsilon()) {
                grad /= length; //  normalize gradient 
            }
            else {
                grad = openvdb::Vec3f(0, 0, 0); // set to zero if the length is negligible
            }

            // Scale_normalized gradient
            grad *= scale;

            gradX->tree().setValue(iter.getCoord(), grad[0]); // x 
            gradY->tree().setValue(iter.getCoord(), grad[1]); // y 
            gradZ->tree().setValue(iter.getCoord(), grad[2]); //z
        }

        //magnitude of the gradient
        openvdb::FloatGrid::Ptr gradMagnitude = openvdb::FloatGrid::create(0.0);

        //  magnitude of the gradient
        for (openvdb::VectorGrid::ValueOnCIter iter = gradGrid->cbeginValueOn(); iter; ++iter) {
            // gradient vector from the current
            openvdb::Vec3f grad = iter.getValue();

            // magnitude of the gradient
            float magnitude = grad.length();

            //scaling 
            magnitude *= scale;

            // in the gradient magnitude grid
            gradMagnitude->tree().setValue(iter.getCoord(), magnitude);
        }

        // grid values to be between 0 and 1
        //for (openvdb::FloatGrid::ValueOnIter iter = gradMagnitude->beginValueOn(); iter; ++iter) {
        //    float val = iter.getValue();
        //    // value between 0 and 1
        //    val = std::max(0.0f, std::min(1.0f, val));
        //    iter.setValue(val);
        //}


        
        //Divergence Calculation 
        
        openvdb::FloatGrid::Ptr d2GradX_dx = openvdb::FloatGrid::create(0.0);
        openvdb::FloatGrid::Ptr d2GradY_dy = openvdb::FloatGrid::create(0.0);
        openvdb::FloatGrid::Ptr d2GradZ_dz = openvdb::FloatGrid::create(0.0);

        // gradient of gradX and gradY
        openvdb::VectorGrid::Ptr grad2X = openvdb::tools::gradient(*gradX);
        openvdb::VectorGrid::Ptr grad2Y = openvdb::tools::gradient(*gradY);
        openvdb::VectorGrid::Ptr grad2Z = openvdb::tools::gradient(*gradZ);

        // second derivatives
        for (openvdb::VectorGrid::ValueOnIter iter = grad2X->beginValueOn(); iter; ++iter) {
            openvdb::Vec3f grad = iter.getValue();
            d2GradX_dx->tree().setValue(iter.getCoord(), grad[0]); // x of gradX
        }

        for (openvdb::VectorGrid::ValueOnIter iter = grad2Y->beginValueOn(); iter; ++iter) {
            openvdb::Vec3f grad = iter.getValue();
            d2GradY_dy->tree().setValue(iter.getCoord(), grad[1]); // y of gradY
        }

        for (openvdb::VectorGrid::ValueOnIter iter = grad2Z->beginValueOn(); iter; ++iter) {
            openvdb::Vec3f grad = iter.getValue();
            d2GradZ_dz->tree().setValue(iter.getCoord(), grad[2]); // z of gradZ
        }

        // Sum 
        openvdb::FloatGrid::Ptr DIV = openvdb::FloatGrid::create(0.0);
        // sum the values
         // Loop through one of the derivative grids, assuming all have the same active voxels
        for (openvdb::FloatGrid::ValueOnIter iter = d2GradX_dx->beginValueOn(); iter; ++iter) {
            openvdb::Coord xyz = iter.getCoord();
            float second_derivative_x = iter.getValue();
            float second_derivative_y = d2GradY_dy->tree().isValueOn(xyz) ? d2GradY_dy->tree().getValue(xyz) : 0.0f;
            float second_derivative_z = d2GradZ_dz->tree().isValueOn(xyz) ? d2GradZ_dz->tree().getValue(xyz) : 0.0f;
            float sum = second_derivative_x + second_derivative_y + second_derivative_z;

            DIV->tree().setValue(xyz, sum); // Set divergence value
        }

        // prune the tree to clean up inactive voxels
        DIV->tree().prune();


        openvdb::FloatGrid::Ptr normalized_DIV = openvdb::FloatGrid::create(0.0);
        for (openvdb::FloatGrid::ValueOnIter iterX = d2GradX_dx->beginValueOn(), iterY = d2GradY_dy->beginValueOn(), iterZ = d2GradZ_dz->beginValueOn(), iterMag = gradMagnitude->beginValueOn(); iterX && iterY && iterZ && iterMag; ++iterX, ++iterY, ++iterZ, ++iterMag) {
            openvdb::Coord xyz = iterX.getCoord();
            float second_derivative_x = iterX.getValue();
            float second_derivative_y = iterY.getValue();
            float second_derivative_z = iterZ.getValue();
            float magnitude = iterMag.getValue();

            // Avoid division by zero
            if (magnitude > std::numeric_limits<float>::epsilon()) {
                float normalized_x = second_derivative_x / magnitude;
                float normalized_y = second_derivative_y / magnitude;
                float normalized_z = second_derivative_z / magnitude;
                float sum = normalized_x + normalized_y + normalized_z;

                normalized_DIV->tree().setValue(xyz, sum);
            }
        }
        

        //  prune the tree
        normalized_DIV->tree().prune();

      

       // openvdb::FloatGrid::Ptr divergenceGrid = openvdb::tools::divergence(*gradGrid);

        // Compute divergence of the gradient field
        openvdb::FloatGrid::Ptr divGrid = openvdb::tools::divergence(*gradGrid);

        openvdb::FloatGrid::Ptr d2PHI_dx2, d2PHI_dy2, d2PHI_dz2;
        computeSecondOrderGradients(sdf_grid, gradMagnitude, d2PHI_dx2, d2PHI_dy2, d2PHI_dz2);

        /*
        //// Lambda function to set negative values to zero
        //auto removeNegativeValues = [](const openvdb::FloatGrid::ValueOnIter& iter) {
        //    if (*iter < 0) {
        //        iter.setValue(0);
        //    }
        //};

        //// Apply the operation to all on-values in the grid
        //openvdb::tools::foreach(divergenceGrid->beginValueOn(), removeNegativeValues);

        //openvdb::FloatGrid::Ptr normalizedDIV = divergenceGrid->deepCopy();

        //for (openvdb::FloatGrid::ValueOnIter iter = normalizedDIV->beginValueOn(); iter; ++iter) {
        //    float val = iter.getValue();
        //    // Clamp the value between 0 and 1
        //    val = std::max(0.0f, std::min(1.0f, val));
        //    iter.setValue(val);
        //}

        //// Find the min and max values in the divergence grid
        //float minValue = std::numeric_limits<float>::max();
        //float maxValue = std::numeric_limits<float>::lowest();

        //for (openvdb::FloatGrid::ValueOnCIter iter = divergenceGrid->cbeginValueOn(); iter; ++iter) {
        //    float val = iter.getValue();
        //    minValue = std::min(minValue, val);
        //    maxValue = std::max(maxValue, val);
        //}

        //// Normalize the divergence grid
        //if (maxValue > minValue) {
        //    for (openvdb::FloatGrid::ValueOnIter iter = divergenceGrid->beginValueOn(); iter; ++iter) {
        //        float normalizedVal = (iter.getValue() - minValue) / (maxValue - minValue);
        //        iter.setValue(normalizedVal);
        //    }
        //}
        //else {
        //    // Handle the case where maxValue equals minValue
        //}

       
        openvdb::FloatGrid::Ptr DIVGrid = openvdb::FloatGrid::create(0.0);
        
        //// grid values to be between 0 and 1
        //for (openvdb::FloatGrid::ValueOnIter iter = d2PHI_dx2->beginValueOn(); iter; ++iter) {
        //    float val = iter.getValue();
        //    // value between 0 and 1
        //    val = std::max(0.0f, std::min(1.0f, val));
        //    iter.setValue(val);
        //}

        //// grid values to be between 0 and 1
        //for (openvdb::FloatGrid::ValueOnIter iter = d2PHI_dy2->beginValueOn(); iter; ++iter) {
        //    float val = iter.getValue();
        //    // value between 0 and 1
        //    val = std::max(0.0f, std::min(1.0f, val));
        //    iter.setValue(val);
        //}

        //// grid values to be between 0 and 1
        //for (openvdb::FloatGrid::ValueOnIter iter = d2PHI_dz2->beginValueOn(); iter; ++iter) {
        //    float val = iter.getValue();
        //    // value between 0 and 1
        //    val = std::max(0.0f, std::min(1.0f, val));
        //    iter.setValue(val);
        //}
        // Get accessors for the grids
        openvdb::FloatGrid::Accessor DIVAccessor = DIVGrid->getAccessor();
        openvdb::FloatGrid::Accessor d2PHI_dx2Accessor = d2PHI_dx2->getAccessor();
        openvdb::FloatGrid::Accessor d2PHI_dy2Accessor = d2PHI_dy2->getAccessor();
        openvdb::FloatGrid::Accessor d2PHI_dz2Accessor = d2PHI_dz2->getAccessor();

        // Iterate over the active values of the first grid
        for (openvdb::FloatGrid::ValueOnIter iter = d2PHI_dy2->beginValueOn(); iter; ++iter) {
            const openvdb::Coord& coord = iter.getCoord();

            // Multiply the values from grid1 and grid2, and set in resultGrid
            DIVAccessor.setValue(coord,  d2PHI_dy2Accessor.getValue(coord) + d2PHI_dz2Accessor.getValue(coord));
        }


        openvdb::FloatGrid::Ptr d3PHI_dy3 = openvdb::FloatGrid::create(0.0);
        openvdb::FloatGrid::Ptr d3PHI_dz3 = openvdb::FloatGrid::create(0.0);
        openvdb::FloatGrid::Ptr d3PHI_dx3 = openvdb::FloatGrid::create(0.0);

        openvdb::FloatGrid::Ptr norm_DIVgrid = openvdb::FloatGrid::create(0.0);

        d3PHI_dy3 = divideGrids(d2PHI_dy2, gradMagnitude);
        d3PHI_dx3 = divideGrids(d2PHI_dx2, gradMagnitude);
        d3PHI_dz3 = divideGrids(d2PHI_dz2, gradMagnitude);

        openvdb::FloatGrid::Accessor norm_DIVgridAccessor = norm_DIVgrid->getAccessor();
        openvdb::FloatGrid::Accessor d3PHI_dx3Accessor = d3PHI_dx3->getAccessor();
        openvdb::FloatGrid::Accessor d3PHI_dy3Accessor = d3PHI_dy3->getAccessor();
        openvdb::FloatGrid::Accessor d3PHI_dz3Accessor = d3PHI_dz3->getAccessor();

        for (openvdb::FloatGrid::ValueOnIter iter = d3PHI_dy3->beginValueOn(); iter; ++iter) {
            const openvdb::Coord& coord = iter.getCoord();

            // Multiply the values from grid1 and grid2, and set in resultGrid
            norm_DIVgridAccessor.setValue(coord,  d3PHI_dy3Accessor.getValue(coord) + d3PHI_dz3Accessor.getValue(coord));
        }

        openvdb::FloatGrid::Ptr gradXnew = calculateGradientZ(sdf_grid);

        */


         
         //DIVGrid and normDiv_Grid

        
        //  x and y and z of the gradient
        openvdb::FloatGrid::Ptr dPHI_dx = calculateGradientX(sdf_grid);
        openvdb::FloatGrid::Ptr dPHI_dy = calculateGradientY(sdf_grid);
        openvdb::FloatGrid::Ptr dPHI_dz = calculateGradientZ(sdf_grid);

        
        // Calculate the gradient magnitude grid
        openvdb::FloatGrid::Ptr gradMag = calculateGradientMagnitude(dPHI_dx, dPHI_dy, dPHI_dz);

        
        //  x and y and z of the gradient
        openvdb::FloatGrid::Ptr d2PHI_dx_n_n = calculateGradientX(dPHI_dx);
        openvdb::FloatGrid::Ptr d2PHI_dy_n_n = calculateGradientY(dPHI_dy);
        openvdb::FloatGrid::Ptr d2PHI_dz_n_n = calculateGradientZ(dPHI_dz);

        
        openvdb::FloatGrid::Accessor DIVAccessor = DIVGrid->getAccessor();
        openvdb::FloatGrid::Accessor d2PHI_dx_n_nAccessor = d2PHI_dx_n_n->getAccessor();
        openvdb::FloatGrid::Accessor d2PHI_dy_n_nAccessor = d2PHI_dy_n_n->getAccessor();
        openvdb::FloatGrid::Accessor d2PHI_dz_n_nAccessor = d2PHI_dz_n_n->getAccessor();

        // active values of the first grid
        for (openvdb::FloatGrid::ValueOnIter iter = d2PHI_dy_n_n->beginValueOn(); iter; ++iter) {
            const openvdb::Coord& coord = iter.getCoord();

            // Multiply 
            DIVAccessor.setValue(coord, d2PHI_dx_n_nAccessor.getValue(coord) + d2PHI_dy_n_nAccessor.getValue(coord) + d2PHI_dz_n_nAccessor.getValue(coord));
        }



        d2PHI_dx2_1 = divideGrids(dPHI_dx, gradMag);
        d2PHI_dy2_1 = divideGrids(dPHI_dy, gradMag);
        d2PHI_dz2_1 = divideGrids(dPHI_dz, gradMag);

        openvdb::FloatGrid::Ptr d2PHI_dx2 = calculateGradientX(d2PHI_dx2_1);
        openvdb::FloatGrid::Ptr d2PHI_dy2 = calculateGradientY(d2PHI_dy2_1);
        openvdb::FloatGrid::Ptr d2PHI_dz2 = calculateGradientZ(d2PHI_dz2_1);

        openvdb::FloatGrid::Accessor norm_DIVgridAccessor = norm_DIVgrid->getAccessor();
        openvdb::FloatGrid::Accessor d2PHI_dx2Accessor = d2PHI_dx2->getAccessor();
        openvdb::FloatGrid::Accessor d2PHI_dy2Accessor = d2PHI_dy2->getAccessor();
        openvdb::FloatGrid::Accessor d2PHI_dz2Accessor = d2PHI_dz2->getAccessor();

        for (openvdb::FloatGrid::ValueOnIter iter = d2PHI_dy2->beginValueOn(); iter; ++iter) {
            const openvdb::Coord& coord = iter.getCoord();

            // Multiply 
            norm_DIVgridAccessor.setValue(coord, d2PHI_dx2Accessor.getValue(coord) + d2PHI_dy2Accessor.getValue(coord) + d2PHI_dz2Accessor.getValue(coord));
        }

       

        
        
        //Eout and Ein Calculation
        
        //float sigma = 3.0f; 
        float sigma1 = 0.0f;
        float kernelSize = (2 * sigma1) + 1;
        float factor = 1 / std::pow(kernelSize, 3); 

        
        
        // iterate over active voxels 
        for (openvdb::FloatGrid::ValueOnCIter iter = input_grid->cbeginValueOn(); iter; ++iter) {
            openvdb::Coord xyz = iter.getCoord();
            float s1 = 0.0f, s2 = 0.0f;

            for (int u = -sigma1; u <= sigma1; ++u) {
                for (int v = -sigma1; v <= sigma1; ++v) {
                    for (int w = -sigma1; w <= sigma1; ++w) { 
                        openvdb::Coord neighbor = xyz.offsetBy(u, v, w);

                        //if neighbor is an active voxel
                        if (input_grid->tree().isValueOn(neighbor)) {
                            float inputValue = input_grid->tree().getValue(neighbor);

                            float foutValue = 0.0f, finValue = 0.0f;
                            if (f_out->tree().isValueOn(neighbor)) {
                                foutValue = f_out->tree().getValue(neighbor);
                            }
                            if (f_in->tree().isValueOn(neighbor)) {
                                finValue = f_in->tree().getValue(neighbor);
                            }

                            s1 += factor * std::pow(inputValue - foutValue, 2);
                            s2 += factor * std::pow(inputValue - finValue, 2);
                        }
                    }
                }
            }

            
            Eout->tree().setValueOn(xyz, s1);
            Ein->tree().setValueOn(xyz, s2);
        }

        

        
       

        float meu = 255 * 255 * 0.0001;
        float dt = 0.1; //0.5
        float f = 0.1; //1.5
        openvdb::FloatGrid::Accessor Eoutaccessor = Eout->getAccessor();
        openvdb::FloatGrid::Accessor Einaccessor = Ein->getAccessor();
        openvdb::FloatGrid::Accessor sdf_accessor = sdf_grid->getAccessor();
        openvdb::FloatGrid::Accessor deri_heavi_accessor = deri_heavi->getAccessor();
        openvdb::FloatGrid::Accessor DivGrid_accessor = DIVGrid->getAccessor();
        openvdb::FloatGrid::Accessor normDivGrid_accessor = norm_DIVgrid->getAccessor();

        float lambda_out = 1.3f;
        //for (openvdb::FloatGrid::ValueOnIter iter = Eout->beginValueOn(); iter; ++iter) {
        //    // Current voxel value
        //    const openvdb::Coord& coord = iter.getCoord();
        //    openvdb::Coord ijk = iter.getCoord();
        //    float value = Eout->tree().getValue(ijk);
        //    // Multiply the current voxel value by the scalar and update the voxel
        //    Eoutaccessor.setValue(iter.getCoord(), lambda_out* value);
        //}


        /*tira::volume<float> I2(200, 200, 200);
        vdb2img3D(*Eout, I2);
        I2.save_npy("C:/Users/meher/spyder/HD13.npy");
        exit(1);*/


        // openvdb::FloatGrid::Ptr sdf_grid_new = sdf_grid->deepCopy();

        for (openvdb::FloatGrid::ValueOnIter iter = input_grid->beginValueOn(); iter; ++iter) {
            const openvdb::Coord& coord = iter.getCoord();
           // sdf_accessor.setValue(coord, sdf_accessor.getValue(coord) - (dt * 1.3 * (deri_heavi_accessor.getValue(coord) * (Eoutaccessor.getValue(coord) - Einaccessor.getValue(coord)))) );
            sdf_accessor.setValue(coord, sdf_accessor.getValue(coord) - (dt * f * (deri_heavi_accessor.getValue(coord) * (Eoutaccessor.getValue(coord) - Einaccessor.getValue(coord)))) + (dt * meu * deri_heavi_accessor.getValue(coord) * normDivGrid_accessor.getValue(coord)) + (dt * DivGrid_accessor.getValue(coord) - dt * normDivGrid_accessor.getValue(coord)));
        }

    

       /* tira::volume<float> I2(50, 200, 100);
        vdb2img3D(*sdf_grid, I2);
        I2.save_npy("C:/Users/meher/spyder/HD13.npy");
        exit(1);*/
        //filter3.gaussian(kernelWidth, sigma);
        //std::cout << "done\n";
        // End timing
        //auto finish = std::chrono::high_resolution_clock::now();

        //// Calculate elapsed time
        //std::chrono::duration<double> elapsed = finish - start;
        //std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
        

    }
    for (openvdb::FloatGrid::ValueOnIter iter = sdf_grid->beginValueOn(); iter; ++iter) {
        if (*iter > 0.0f) {
            //if its value is greater than zero
            iter.setValueOff();
        }
    }
    //exit(1);
    std::string phi = "C:/openvdb_drop/bin/KESM_2000_863phi.vdb";
    openvdb::initialize();

    // Create a VDB file object.
    openvdb::io::File fileEout(phi);
    fileEout.write({ sdf_grid });
    //// Close the file. 
    fileEout.close();

    tira::volume<float> I2(1020, 1000, 1000);
    vdb2img3D(*sdf_grid, I2); 
    I2.save_npy("C:/Users/meher/spyder/KESM_2000_863phi.npy");

    return 0;
}
