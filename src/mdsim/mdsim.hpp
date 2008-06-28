/* Molecular Dynamics simulation of a Lennard-Jones fluid
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

#ifndef MDSIM_MDSIM_HPP
#define MDSIM_MDSIM_HPP

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <cuda_wrapper.hpp>
#include <signal.h>
#include <stdint.h>
#include "H5param.hpp"
#include "autocorrelation.hpp"
#include "perf.hpp"
#include "block.hpp"
#include "energy.hpp"
#include "ljfluid.hpp"
#include "log.hpp"
#include "options.hpp"
#include "rand48.hpp"
#include "trajectory.hpp"


#define foreach BOOST_FOREACH

namespace mdsim
{

/**
 * Molecular Dynamics simulation of a Lennard-Jones fluid
 */
template <unsigned dimension, typename T, typename U>
class mdsim
{
public:
    /** initialize MD simulation program */
    mdsim(options const& opts);
    /** run MD simulation program */
    void operator()();

private:
    /** signal handler */
    static void handle_signal(int signum);

private:
    /** program options */
    options const& opts;
    /** global simulation parameters */
    H5param param;
    /** Lennard-Jones fluid simulation */
    ljfluid<dimension, T, U> fluid;
    /** block algorithm parameters */
    block_param<dimension, T> block;

    /** signal number */
    static int signal_;
};

/**
 * initialize MD simulation program
 */
template <unsigned dimension, typename T, typename U>
mdsim<dimension, T, U>::mdsim(options const& opts) : opts(opts)
{
    // set positional coordinates dimension
    param.dimension(dimension);
    LOG("positional coordinates dimension: " << dimension);

    // initialize Lennard Jones fluid simulation
    if (!opts.trajectory_input_file().empty()) {
	trajectory<dimension, T, U, false> traj;
	// open trajectory input file
	traj.open(opts.trajectory_input_file().value());

	// read global simulation parameters
	traj.read(param);
	// set number of particles in system
	fluid.particles(param.particles());
	// set particle density
	fluid.density(!opts.density().defaulted() ? opts.density().value() : param.density());
#ifdef USE_CELL
	// compute cell parameters
	fluid.cell_occupancy(opts.cell_occupancy().value());
#endif
	// set number of CUDA execution threads
	fluid.threads(!opts.threads().defaulted() ? opts.threads().value() : param.threads());
	// read trajectory sample and restore system state
	fluid.restore(boost::bind(&trajectory<dimension, T, U, false>::read, boost::ref(traj), _1, _2, opts.trajectory_sample().value()));

	// close trajectory input file
	traj.close();

	// initialize random number generator with seed
	fluid.rng(opts.rng_seed().value());

	if (!opts.temperature().defaulted()) {
	    LOG_WARNING("discarding velocities from trajectory file");
	    // set system temperature according to Maxwell-Boltzmann distribution
	    fluid.temperature(opts.temperature().value());
	}

	// override parameters with non-default option values
	if (!opts.timestep().defaulted()) {
	    param.timestep(opts.timestep().value());
	}
#ifndef USE_BENCHMARK
	if (!opts.block_size().defaulted()) {
	    param.block_size(opts.block_size().value());
	}
	if (!opts.max_samples().defaulted()) {
	    param.max_samples(opts.max_samples().value());
	}
	if (!opts.q_values().defaulted()) {
	    param.q_values(opts.q_values().value());
	}
#endif
	if (!opts.steps().defaulted()) {
	    param.steps(opts.steps().value());
	}
    }
    else {
	// set number of particles in system
	fluid.particles(opts.particles().value());

	if (!opts.box_length().empty()) {
	    // set simulation box length
	    fluid.box(opts.box_length().value());
	}
	else {
	    // set particle density
	    fluid.density(opts.density().value());
	}
#ifdef USE_CELL
	// compute cell parameters
	fluid.cell_occupancy(opts.cell_occupancy().value());
#endif

	// set number of CUDA execution threads
	fluid.threads(opts.threads().value());
	// initialize random number generator with seed
	fluid.rng(opts.rng_seed().value());
	// arrange particles on a face-centered cubic (fcc) lattice
	fluid.lattice();
	// set system temperature according to Maxwell-Boltzmann distribution
	fluid.temperature(opts.temperature().value());

	// gather parameters from option values
	param.timestep(opts.timestep().value());
#ifndef USE_BENCHMARK
	param.block_size(opts.block_size().value());
	param.max_samples(opts.max_samples().value());
	param.q_values(opts.q_values().value());
#endif
	param.steps(opts.steps().value());
    }

    // gather number of particles
    param.particles(fluid.particles());
#ifdef USE_CELL
    // gather number of cells per dimension
    param.cells(fluid.cells());
    // gather total number of cell placeholders
    param.placeholders(fluid.placeholders());
#endif
    // gather number of CUDA execution blocks in grid
    param.blocks(fluid.blocks());
    // gather particle density
    param.density(fluid.density());
    // gather simulation box length
    param.box_length(fluid.box());
#ifdef USE_CELL
    // gather cell length
    param.cell_length(fluid.cell_length());
    // gather average cell occupancy
    param.cell_occupancy(fluid.cell_occupancy());
    // gather number of placeholders per cell
    param.cell_size(fluid.cell_size());
#endif
    // gather number of CUDA execution threads per block
    param.threads(fluid.threads());
    // gather potential cutoff distance
    param.cutoff_distance(fluid.cutoff_distance());

    // set simulation timestep
    fluid.timestep(param.timestep());

#ifndef USE_BENCHMARK
    if (!opts.time().empty()) {
	// set total simulation time
	block.time(opts.time().value(), param.timestep());
    }
    else {
	// set total number of simulation steps
	block.steps(param.steps(), param.timestep());
    }
    // gather total number of simulation steps
    param.steps(block.steps());
    // gather total simulation time
    param.time(block.time());

    // set block size
    block.block_size(param.block_size());
    // gather block shift
    param.block_shift(block.block_shift());
    // gather block count
    param.block_count(block.block_count());

    // set maximum number of samples per block
    block.max_samples(param.max_samples());
#endif
}

/**
 * run MD simulation program
 */
template <unsigned dimension, typename T, typename U>
void mdsim<dimension, T, U>::operator()()
{
#ifndef USE_BENCHMARK
    // time correlation functions
    autocorrelation<dimension, T, U> tcf(block, param.box_length(), opts.q_values().value());
    tcf.open(opts.output_file_prefix().value() + ".tcf");
    tcf << param;
    // trajectory file writer
    trajectory<dimension, T, U> traj(block);
    traj.open(opts.output_file_prefix().value() + ".trj", param.particles());
    traj << param;
    // thermodynamic equilibrium properties
    energy<dimension, T, U> tep(block);
    tep.open(opts.output_file_prefix().value() + ".tep");
    tep << param;
#endif
    // performance data
    perf<dimension, T, U> prf;
    prf.open(opts.output_file_prefix().value() + ".prf");
    prf << param;

    // install signal handlers
    boost::array<sighandler_t, 3> sigh;
    sigh[0] = signal(SIGHUP, handle_signal);
    sigh[1] = signal(SIGINT, handle_signal);
    sigh[2] = signal(SIGTERM, handle_signal);

    LOG("starting MD simulation");

    for (uint64_t step = 0; step < param.steps(); ++step) {
	// abort simulation on signal
	if (signal_) {
	    LOG_WARNING("caught signal at simulation step " << step);
	    break;
	}

	// copy previous MD simulation state from GPU to host
	fluid.sample();
	// stream next MD simulation program step on GPU
	fluid.mdstep();
#ifndef USE_BENCHMARK
	// sample time correlation functions
	fluid.sample(boost::bind(&autocorrelation<dimension, T, U>::sample, boost::ref(tcf), _2, _3));
	// sample thermodynamic equilibrium properties
	fluid.sample(boost::bind(&energy<dimension, T, U>::sample, boost::ref(tep), _3, _4, _5, param.density(), param.timestep()));
	// sample trajectory
	fluid.sample(boost::bind(&trajectory<dimension, T, U>::sample, boost::ref(traj), _1, _3, param.particles(), param.timestep()));
#endif
	// synchronize MD simulation program step on GPU
	fluid.synchronize();
    }

    LOG("finished MD simulation");

    // restore previous signal handlers
    signal(SIGHUP, sigh[0]);
    signal(SIGINT, sigh[1]);
    signal(SIGTERM, sigh[2]);

#ifndef USE_BENCHMARK
    // write time correlation function results to HDF5 file
    tcf.write();
    tcf.close();
    // write thermodynamic equilibrium properties to HDF5 file
    tep.write();
    tep.close();
    // close HDF5 trajectory output file
    traj.close();
#endif
    // write performance data to HDF5 file
    prf.write(fluid.times());
    prf.close();
}

/**
 * signal handler
 */
template <unsigned dimension, typename T, typename U>
void mdsim<dimension, T, U>::handle_signal(int signum)
{
    // store signal number in global variable
    signal_ = signum;
}

template <unsigned dimension, typename T, typename U>
int mdsim<dimension, T, U>::signal_(0);

#undef foreach

} // namespace mdsim

#endif /* ! MDSIM_MDSIM_HPP */
