/* Thermodynamic equilibrium properties
 *
 * Copyright (C) 2008  Peter Colberg
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MDSIM_ENERGY_HPP
#define MDSIM_ENERGY_HPP

#include <H5Cpp.h>
#include <algorithm>
#include <boost/foreach.hpp>
#include <string>
#include <vector>
#include "H5param.hpp"
#include "accumulator.hpp"
#include "ljfluid.hpp"
#include "log.hpp"
#include "statistics.hpp"


#define foreach BOOST_FOREACH

namespace mdsim
{

/**
 * Thermodynamic equilibrium properties
 */
template <unsigned dimension, typename T>
class energy
{
public:
    energy(block_param<dimension, T> const& param);
    /** create HDF5 thermodynamic equilibrium properties output file */
    void open(std::string const& filename);
    /** close HDF5 thermodynamic equilibrium properties output file */
    void close();

    /** dump global simulation parameters to HDF5 file */
    energy<dimension, T>& operator<<(H5param const& param);

    void sample(cuda::host::vector<T> const& v, cuda::host::vector<float> const& en, cuda::host::vector<float> const& virial, float const& density);
    void write();

private:
    /** block algorithm parameters */
    block_param<dimension, T> param;
    /** number of samples */
    unsigned int samples_;

    /** thermodynamic equilibrium properties */
    std::vector<float> en_pot_;
    std::vector<float> en_kin_;
    std::vector<float> en_tot_;
    std::vector<float> temp_;
    std::vector<float> press_;
    std::vector<T> v_cm_;

    /** HDF5 thermodynamic equilibrium properties output file */
    H5::H5File file_;
};


template <unsigned dimension, typename T>
energy<dimension, T>::energy(block_param<dimension, T> const& param) : param(param), samples_(0)
{
#ifdef NDEBUG
    // turns off the automatic error printing from the HDF5 library
    H5::Exception::dontPrint();
#endif

    // allocate thermodynamic equilibrium properties buffers
    try {
	en_pot_.reserve(param.max_samples());
	en_kin_.reserve(param.max_samples());
	en_tot_.reserve(param.max_samples());
	temp_.reserve(param.max_samples());
	press_.reserve(param.max_samples());
	v_cm_.reserve(param.max_samples());
    }
    catch (std::bad_alloc const& e) {
	throw exception("failed to allocate thermodynamic equilibrium properties buffers");
    }
}

/**
 * create HDF5 thermodynamic equilibrium properties output file
 */
template <unsigned dimension, typename T>
void energy<dimension, T>::open(std::string const& filename)
{
    LOG("write thermodynamic equilibrium properties to file: " << filename);
    try {
	// truncate existing file
	file_ = H5::H5File(filename, H5F_ACC_TRUNC);
    }
    catch (H5::FileIException const& e) {
	throw exception("failed to create thermodynamic equilibrium properties output file");
    }

}

/**
 * close HDF5 thermodynamic equilibrium properties output file
 */
template <unsigned dimension, typename T>
void energy<dimension, T>::close()
{
    try {
	file_.close();
    }
    catch (H5::Exception const& e) {
	throw exception("failed to close HDF5 correlations output file");
    }
}

/**
 * dump global simulation parameters to HDF5 file
 */
template <unsigned dimension, typename T>
energy<dimension, T>& energy<dimension, T>::operator<<(H5param const& param)
{
    param.write(file_.createGroup("/parameters"));
    return *this;
}

/**
 * sample thermodynamic equilibrium properties
 */
template <unsigned dimension, typename T>
void energy<dimension, T>::sample(cuda::host::vector<T> const& v, cuda::host::vector<float> const& en, cuda::host::vector<float> const& virial, float const& density)
{
    if (samples_ >= param.max_samples()) return;

    // mean squared velocity
    accumulator<float> vv;
    foreach (T const& v_, v) {
	vv += v_ * v_;
    }

    // mean potential energy per particle
    en_pot_.push_back(mean(en.begin(), en.end()));
    // mean kinetic energy per particle
    en_kin_.push_back(vv.mean() / 2);
    // mean total energy per particle
    en_tot_.push_back(en_pot_.back() + en_kin_.back());
    // temperature
    temp_.push_back(vv.mean() / dimension);
    // pressure
    press_.push_back(density * (vv.mean() + mean(virial.begin(), virial.end())));
    // velocity center of mass
    v_cm_.push_back(mean(v.begin(), v.end()));

    samples_++;
}


/**
 * write thermodynamic equilibrium properties buffer to file
 */
template <unsigned dimension, typename T>
void energy<dimension, T>::write()
{
    // create dataspaces for scalar and vector types
    hsize_t dim_scalar[2] = { param.max_samples(), 1 };
    hsize_t dim_vector[2] = { param.max_samples(), dimension };
    H5::DataSpace ds_scalar(2, dim_scalar);
    H5::DataSpace ds_vector(2, dim_vector);

    // HDF5 datasets for thermodynamic equilibrium properties
    boost::array<H5::DataSet, 6> dataset_;
    H5::DataType tid(H5::PredType::NATIVE_FLOAT);

    try {
	// mean potential energy per particle
	dataset_[0] = file_.createDataSet("EPOT", tid, ds_scalar);
	// mean kinetic energy per particle
	dataset_[1] = file_.createDataSet("EKIN", tid, ds_scalar);
	// mean total energy per particle
	dataset_[2] = file_.createDataSet("ETOT", tid, ds_scalar);
	// temperature
	dataset_[3] = file_.createDataSet("TEMP", tid, ds_scalar);
	// pressure
	dataset_[4] = file_.createDataSet("PRESS", tid, ds_scalar);
	// velocity center of mass
	dataset_[5] = file_.createDataSet("VCM", tid, ds_vector);
    }
    catch (H5::FileIException const& e) {
	throw exception("failed to create datasets in HDF5 energy file");
    }

    try {
	// mean potential energy per particle
	dataset_[0].write(en_pot_.data(), tid, ds_scalar, ds_scalar);
	// mean kinetic energy per particle
	dataset_[1].write(en_kin_.data(), tid, ds_scalar, ds_scalar);
	// mean total energy per particle
	dataset_[2].write(en_tot_.data(), tid, ds_scalar, ds_scalar);
	// temperature
	dataset_[3].write(temp_.data(), tid, ds_scalar, ds_scalar);
	// pressure
	dataset_[4].write(press_.data(), tid, ds_scalar, ds_scalar);
	// velocity center of mass
	dataset_[5].write(v_cm_.data(), tid, ds_vector, ds_vector);
    }
    catch (H5::FileIException const& e) {
	throw exception("failed to write thermodynamic equilibrium properties to HDF5 energy file");
    }
}

} // namespace mdsim

#undef foreach

#endif /* ! MDSIM_ENERGY_HPP */
