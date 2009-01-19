/* Molecular Dynamics simulation
 *
 * Copyright © 2008-2009  Peter Colberg
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

#ifndef LJGPU_MDSIM_BASE_HPP
#define LJGPU_MDSIM_BASE_HPP

#include <boost/assign.hpp>
#include <cmath>
#include <limits>
#include <ljgpu/mdsim/impl.hpp>
#include <ljgpu/mdsim/traits.hpp>
#include <ljgpu/mdsim/variant.hpp>
#include <ljgpu/sample/H5param.hpp>
#include <ljgpu/sample/perf.hpp>
#include <ljgpu/util/exception.hpp>
#include <ljgpu/util/log.hpp>

#define foreach BOOST_FOREACH

namespace ljgpu
{

template <typename mdsim_impl>
class mdsim_base
{
public:
    typedef mdsim_impl impl_type;
    typedef mdsim_traits<impl_type> traits_type;
    typedef typename traits_type::float_type float_type;
    typedef typename traits_type::vector_type vector_type;
    typedef typename traits_type::sample_type sample_type;
    typedef typename sample_type::sample_visitor sample_visitor;
    enum { dimension = traits_type::dimension };

public:
    mdsim_base() :
	mixture_(UNARY) {}

    /** set number of particles */
    void particles(unsigned int value);
    /** set number of A and B particles in binary mixture */
    void particles(boost::array<unsigned int, 2> const& value);
    /** set particle density */
    void density(float_type value);
    /** set periodic box length */
    void box(float_type value);
    /** set system state from phase space sample */
    void sample(sample_visitor read);

    /** returns number of particles */
    unsigned int particles() const { return npart; }
    /** returns particle density */
    float_type density() const { return density_; }
    /** returns periodic box length */
    float_type box() const { return box_; }
    /** returns trajectory sample */
    sample_type const& sample() const { return m_sample; }
    /** returns and resets CPU or GPU time accumulators */
    perf::counters times();

    mixture_type mixture() const { return mixture_ ; }

protected:
    /** write parameters to HDF5 parameter group */
    void param(H5param& param) const;

protected:
    /** number of particles */
    unsigned int npart;
    /** number of A and B particles in binary mixture */
    boost::array<unsigned int, 2> mpart;
    /** particle density */
    float_type density_;
    /** periodic box length */
    float_type box_;
    /** trajectory sample in swappable host memory */
    sample_type m_sample;
    /** GPU time accumulators */
    perf::counters m_times;

    mixture_type mixture_;
};

template <typename mdsim_impl>
void mdsim_base<mdsim_impl>::particles(unsigned int value)
{
    if (value < 1) {
	throw exception("invalid number of particles");
    }
    npart = value;
    mpart = boost::assign::list_of(npart)(0);
    mixture_ = UNARY;
    LOG("number of particles: " << npart);
}

template <typename mdsim_impl>
void mdsim_base<mdsim_impl>::particles(boost::array<unsigned int, 2> const& value)
{
    if (*std::min_element(value.begin(), value.end()) < 1) {
	throw exception("invalid number of A or B particles");
    }
    mpart = value;
    npart = std::accumulate(mpart.begin(), mpart.end(), 0);
    mixture_ = BINARY;
    LOG("binary mixture with " << mpart[0] << " A particles and " << mpart[1] << " B particles");
}

template <typename mdsim_impl>
void mdsim_base<mdsim_impl>::density(float_type value)
{
    // set particle density
    density_ = value;
    LOG("particle density: " << density_);

    // compute periodic box length
    box_ = std::pow(npart / density_, (float_type) 1 / dimension);
    LOG("periodic simulation box length: " << box_);
}

template <typename mdsim_impl>
void mdsim_base<mdsim_impl>::box(float_type value)
{
    // set periodic box length
    box_ = value;
    LOG("periodic simulation box length: " << box_);

    // compute particle density
    density_ = npart / std::pow(box_, (float_type) dimension);
    LOG("particle density: " << density_);
}

template <typename mdsim_impl>
void mdsim_base<mdsim_impl>::sample(mdsim_base<mdsim_impl>::sample_visitor read)
{
    typedef typename sample_type::uniform_sample uniform_sample;
    typedef typename sample_type::position_vector position_vector;
    typedef typename position_vector::value_type position_value;

    read(m_sample);

    for (size_t i = 0; i < m_sample.size(); ++i) {
	if (m_sample[i].r.size() != mpart[i]) {
	    throw exception("mismatching number of particles in phase space sample");
	}
    }

    position_value const box = m_sample.box;
    foreach (uniform_sample& sample, m_sample) {
	foreach (position_vector &r, sample.r) {
	    // apply periodic boundary conditions to positions
	    r -= floor(r / box) * box;
	}
    }

    if (std::fabs(box - box_) > (box_ * std::numeric_limits<float>::epsilon())) {
	LOG("rescaling periodic simulation box length from " << box);
	position_value const scale = box_ / box;
	foreach (uniform_sample &sample, m_sample) {
	    foreach (position_vector &r, sample.r) {
		r *= scale;
	    }
	}
	m_sample.box = box_;
    }
}

template <typename mdsim_impl>
perf::counters mdsim_base<mdsim_impl>::times()
{
    perf::counters times(m_times);
    foreach (perf::counter& i, m_times) {
	// reset performance counter
	i.second.clear();
    }
    return times;
}

template <typename mdsim_impl>
void mdsim_base<mdsim_impl>::param(H5param& param) const
{
    H5xx::group node(param["mdsim"]);
    node["box_length"] = box_;
    node["density"] = density_;
    if (mixture_ == BINARY) {
	node["particles"] = mpart;
    }
    else {
	node["particles"] = npart;
    }
}

} // namespace ljgpu

#undef foreach

#endif /* ! LJGPU_MDSIM_BASE_HPP */
