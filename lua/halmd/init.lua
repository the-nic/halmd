--
-- Copyright © 2010  Peter Colberg
--
-- This file is part of HALMD.
--
-- HALMD is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU General Public License for more details.
--
-- You should have received a copy of the GNU General Public License
-- along with this program.  If not, see <http://www.gnu.org/licenses/>.
--

--
-- Load HAL’s MD package
--
require("halmd.io.statevars.writers")
require("halmd.io.trajectory.writers")
require("halmd.mdsim.box")
require("halmd.mdsim.core")
require("halmd.mdsim.force")
require("halmd.mdsim.integrator")
require("halmd.mdsim.neighbour")
require("halmd.mdsim.particle")
require("halmd.mdsim.position")
require("halmd.mdsim.sort")
require("halmd.mdsim.velocity")
require("halmd.observables.thermodynamics")
require("halmd.profiler")
require("halmd.sampler")

--
-- Run simulation
--
function run()
    local core = halmd.mdsim.core() -- singleton
    core.particle = halmd.mdsim.particle()
    core.box = halmd.mdsim.box()
    core.integrator = halmd.mdsim.integrator()
    core.force = halmd.mdsim.force()
    core.neighbour = halmd.mdsim.neighbour()
    core.sort = halmd.mdsim.sort()
    core.position = halmd.mdsim.position()
    core.velocity = halmd.mdsim.velocity()

    local sampler = halmd.sampler() -- singleton
    local profiler = halmd.profiler()
    sampler.profile_writers = profiler.profile_writers
    sampler.trajectory_writer = halmd.io.trajectory.writers()
    sampler.observables = { halmd.observables.thermodynamics() }
    sampler.statevars_writer = halmd.io.statevars.writers()

    sampler:run()
end