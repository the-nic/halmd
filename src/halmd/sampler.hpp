/*
 * Copyright © 2010  Felix Höfling
 *
 * This file is part of HALMD.
 *
 * HALMD is free software: you can redistribute it and/or modify
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

#ifndef HALMD_SAMPLER_HPP
#define HALMD_SAMPLER_HPP

#include <boost/shared_ptr.hpp>

#include <halmd/io/profile/writer.hpp>
#include <halmd/io/statevars/writer.hpp>
#include <halmd/io/trajectory/writer.hpp>
#include <halmd/main.hpp>
#include <halmd/mdsim/core.hpp>
#include <halmd/observables/observable.hpp>
#include <halmd/utility/module.hpp>
#include <halmd/options.hpp>
#include <halmd/utility/profiler.hpp>

namespace halmd
{

template <int dimension>
class sampler
  : public halmd::main
{
public:
    // module definitions
    typedef sampler _Self;
    typedef halmd::main _Base;
    static void options(po::options_description& desc);
    static void depends();
    static void select(po::variables_map const& vm) {};

    typedef mdsim::core<dimension> core_type;
    typedef observables::observable<dimension> observable_type;
    typedef io::statevars::writer<dimension> statevars_writer_type;
    typedef io::trajectory::writer<dimension> trajectory_writer_type;
    typedef io::profile::writer profile_writer_type;
    typedef utility::profiler profiler_type;

    sampler(modules::factory& factory, po::variables_map const& vm);
    virtual void run();
    void sample(bool force=false);
    void register_runtimes(profiler_type& profiler);
    uint64_t steps() { return steps_; }
    double time() { return time_; }

    shared_ptr<core_type> core;
    std::vector<shared_ptr<observable_type> > observables;
    shared_ptr<statevars_writer_type> statevars_writer;
    shared_ptr<trajectory_writer_type> trajectory_writer;
    std::vector<shared_ptr<profile_writer_type> > profile_writers;

    // module runtime accumulator descriptions
    HALMD_PROFILE_TAG( msv_output_, "output of macroscopic state variables" );

private:
    /** number of integration steps */
    uint64_t steps_;
    /** integration time in MD units */
    double time_;
    // value from option --sampling-state-vars
    unsigned statevars_interval_;
    // value from option --sampling-trajectory
    unsigned trajectory_interval_;

    // list of profiling timers
    boost::fusion::map<
        boost::fusion::pair<msv_output_, accumulator<double> >
    > runtime_;
};

} // namespace halmd

#endif /* ! HALMD_SAMPLER_HPP */
