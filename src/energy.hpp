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

#include <algorithm>
#include <boost/foreach.hpp>
#include <hdf5.hpp>
#include <string>
#include <vector>
#include "accumulator.hpp"
#include "ljfluid.hpp"
#include "options.hpp"
#include "statistics.hpp"


#define foreach BOOST_FOREACH

namespace mdsim
{

/**
 * Thermodynamic equilibrium properties
 */
template <unsigned dimension, typename S>
class energy
{
public:
    energy(options const& opts);
    void sample(S const& s);
    void write();

    /** write parameters to HDF5 file */
    template <typename visitor> void visit_param(visitor const& v);

private:
    /** simulation timestep */
    float timestep_;
    /** particle density */
    float density_;
    /** sample count */
    unsigned int samples_;
    /** number of samples */
    unsigned int max_samples_;

    /** thermodynamic equilibrium properties */
    std::vector<float> en_pot_;
    std::vector<float> en_kin_;
    std::vector<float> en_tot_;
    std::vector<float> temp_;
    std::vector<float> press_;
    std::vector<typename S::vector_type::value_type> v_cm_;

    /** HDF5 output file */
    H5::H5File file_;
};


template <unsigned dimension, typename S>
energy<dimension, S>::energy(options const& opts) : timestep_(opts.timestep()), density_(opts.density()), samples_(0)
{
#ifdef NDEBUG
    // turns off the automatic error printing from the HDF5 library
    H5::Exception::dontPrint();
#endif

    // create thermodynamic equilibrium properties output file
    try {
	// truncate existing file
	file_ = H5::H5File(opts.energy_output_file(), H5F_ACC_TRUNC);
    }
    catch (H5::FileIException const& e) {
	throw exception("failed to create thermodynamic equilibrium properties output file");
    }

    // number of samples
    max_samples_ = std::min(opts.max_samples(), opts.steps());

    try {
	en_pot_.reserve(max_samples_);
	en_kin_.reserve(max_samples_);
	en_tot_.reserve(max_samples_);
	temp_.reserve(max_samples_);
	press_.reserve(max_samples_);
	v_cm_.reserve(max_samples_);
    }
    catch (std::bad_alloc const& e) {
	throw exception("failed to allocate thermodynamic equilibrium properties buffer");
    }
}

/**
 * write parameters to HDF5 file
 */
template <unsigned dimension, typename S>
template <typename visitor>
void energy<dimension, S>::visit_param(visitor const& v)
{
    v.write_param(file_.openGroup("/"));
}

/**
 * sample thermodynamic equilibrium properties
 */
template <unsigned dimension, typename S>
void energy<dimension, S>::sample(S const& s)
{
    if (samples_ >= max_samples_) return;

    // mean squared velocity
    accumulator<float> vv;
    foreach (typename S::vector_type::value_type const& v, s.v) {
	vv += v * v;
    }

    // mean potential energy per particle
    en_pot_.push_back(mean(s.en.begin(), s.en.end()));
    // mean kinetic energy per particle
    en_kin_.push_back(vv.mean() / 2);
    // mean total energy per particle
    en_tot_.push_back(en_pot_.back() + en_kin_.back());
    // temperature
    temp_.push_back(vv.mean() / dimension);
    // pressure
    press_.push_back(density_ * (vv.mean() + mean(s.virial.begin(), s.virial.end())));
    // velocity center of mass
    v_cm_.push_back(mean(s.v.begin(), s.v.end()));

    samples_++;
}


/**
 * write thermodynamic equilibrium properties buffer to file
 */
template <unsigned dimension, typename S>
void energy<dimension, S>::write()
{

    // write parameters
    H5ext::Group root(file_.openGroup("/"));
    H5::DataType dt(H5::PredType::NATIVE_FLOAT);

    try {
	root["timestep"] = timestep_;
    }
    catch (H5::FileIException const& e) {
	throw exception("failed to create attributes in HDF5 energy file");
    }

    // create dataspaces for scalar and vector types
    hsize_t dim_scalar[2] = { max_samples_, 1 };
    hsize_t dim_vector[2] = { max_samples_, dimension };
    H5::DataSpace ds_scalar(2, dim_scalar);
    H5::DataSpace ds_vector(2, dim_vector);

    // HDF5 datasets for thermodynamic equilibrium properties
    boost::array<H5::DataSet, 6> dataset_;

    try {
	// mean potential energy per particle
	dataset_[0] = file_.createDataSet("EPOT", dt, ds_scalar);
	// mean kinetic energy per particle
	dataset_[1] = file_.createDataSet("EKIN", dt, ds_scalar);
	// mean total energy per particle
	dataset_[2] = file_.createDataSet("ETOT", dt, ds_scalar);
	// temperature
	dataset_[3] = file_.createDataSet("TEMP", dt, ds_scalar);
	// pressure
	dataset_[4] = file_.createDataSet("PRESS", dt, ds_scalar);
	// velocity center of mass
	dataset_[5] = file_.createDataSet("VCM", dt, ds_vector);
    }
    catch (H5::FileIException const& e) {
	throw exception("failed to create datasets in HDF5 energy file");
    }

    try {
	// mean potential energy per particle
	dataset_[0].write(en_pot_.data(), dt, ds_scalar, ds_scalar);
	// mean kinetic energy per particle
	dataset_[1].write(en_kin_.data(), dt, ds_scalar, ds_scalar);
	// mean total energy per particle
	dataset_[2].write(en_tot_.data(), dt, ds_scalar, ds_scalar);
	// temperature
	dataset_[3].write(temp_.data(), dt, ds_scalar, ds_scalar);
	// pressure
	dataset_[4].write(press_.data(), dt, ds_scalar, ds_scalar);
	// velocity center of mass
	dataset_[5].write(v_cm_.data(), dt, ds_vector, ds_vector);
    }
    catch (H5::FileIException const& e) {
	throw exception("failed to write thermodynamic equilibrium properties to HDF5 energy file");
    }
}

} // namespace mdsim

#undef foreach

#endif /* ! MDSIM_ENERGY_HPP */
