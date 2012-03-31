#!/usr/bin/env halmd
--
-- Copyright © 2010-2012  Peter Colberg and Felix Höfling
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

local halmd = require("halmd")

-- grab modules
local log = halmd.io.log
local mdsim = halmd.mdsim
local observables = halmd.observables
local readers = halmd.io.readers
local writers = halmd.io.writers

--
-- Setup and run simulation
--
local function liquid(args)
    -- FIXME support reading multiple species groups into single particle
    local reader = readers.trajectory{group = "A"}

    -- label particles A, B, …

    -- create system state
    local particle = mdsim.particle{
        particles = assert(args.particles)
      , masses = assert(args.masses)
      , dimension = assert(args.dimension)
      , label = "all" -- FIXME make optional
    }
    -- create simulation box
    mdsim.box{particles = {particle}}
    -- add integrator
    mdsim.integrator{particle = particle}
    -- add force
    local force = mdsim.force{particle = particle}
    -- set initial particle positions (optionally from reader)
    mdsim.position{reader = reader, particle = particle}
    -- set initial particle velocities (optionally from reader)
    mdsim.velocity{reader = reader, particle = particle}

    -- Construct sampler.
    local sampler = observables.sampler{}

    -- Construct particle groups and phase space samplers by species (species are numbered 0, 1, 2, ...)
    local species = {} for i = 1, #args.particles do species[i] = i - 1 end -- FIXME avoid explicit for-loop!?
    local particle_group = observables.samples.particle_group{
        particle = particle, species = species
    }
    local phase_space = observables.phase_space{particle = particle_group}

    -- Sample macroscopic state variables.
    observables.thermodynamics{particle_group = { particle }, force = { force }, every = args.sampling.state_vars}
    observables.thermodynamics{particle_group = { particle_group[1] }, force = { force }, every = args.sampling.state_vars}

    -- Write trajectory to H5MD file.
    writers.trajectory{particle_group = particle_group, every = args.sampling.trajectory}

    -- Sample static structure factors, construct density modes before.
    local density_mode = observables.density_mode{
        phase_space = phase_space, max_wavevector = 15
    }
    observables.ssf{density_mode = density_mode, every = args.sampling.structure}

    -- compute mean-square displacement
    observables.dynamics.correlation{sampler = phase_space, correlation = "mean_square_displacement"}
    -- compute mean-quartic displacement
    observables.dynamics.correlation{sampler = phase_space, correlation = "mean_quartic_displacement"}
    -- compute velocity autocorrelation function
    observables.dynamics.correlation{sampler = phase_space, correlation = "velocity_autocorrelation"}
    -- compute intermediate scattering function from density modes different than those used for ssf computation
    density_mode = observables.density_mode{
        phase_space = phase_space, max_wavevector = 12, decimation = 2
    }
--     observables.dynamics.correlation{sampler = density_mode, correlation = "intermediate_scattering_function"}

    -- yield sampler.setup slot from Lua to C++ to setup simulation box
    coroutine.yield(sampler:setup())

    -- yield sampler.run slot from Lua to C++ to run simulation
    coroutine.yield(sampler:run())
end

--
-- Parse command-line arguments.
--
local function parse_args()
    local parser = halmd.utility.program_options.argument_parser()

    parser:add_argument("output,o", {type = "string", action = function(args, key, value)
        -- substitute current time
        args[key] = os.date(value)
    end, default = "liquid_%Y%m%d_%H%M%S", help = "prefix of output files"})

    parser:add_argument("verbose,v", {type = "accumulate", action = function(args, key, value)
        local level = {
            -- console, file
            {"warning", "info" },
            {"info"   , "info" },
            {"debug"  , "debug"},
            {"trace"  , "trace"},
        }
        args[key] = level[value] or level[#level]
    end, default = 1, help = "increase logging verbosity"})

    parser:add_argument("disable-gpu", {type = "boolean", help = "disable GPU acceleration"})
    parser:add_argument("devices", {type = "vector", dtype = "integer", help = "CUDA device(s)"})

    parser:add_argument("particles", {type = "vector", dtype = "integer", default = {1000}, help = "number of particles"})
    parser:add_argument("masses", {type = "vector", dtype = "number", default = {1}, help = "particle masses"})
    parser:add_argument("dimension", {type = "integer", default = 3, action = function(args, key, value)
        if value ~= 2 and value ~= 3 then
            error(("invalid dimension '%d'"):format(value), 0)
        end
        args[key] = value
    end, help = "dimension of positional coordinates"})

    parser:add_argument("ensemble", {type = "string", choices = {
        nve = "Constant NVE",
        nvt = "Constant NVT",
    }, default = "nve", help = "statistical ensemble"})

    local sampling = parser:add_argument_group("sampling", {help = "sampling intervals"})
    sampling:add_argument("trajectory", {type = "integer", default = 10, help = "sampling interval for trajectory"})
    sampling:add_argument("structure", {type = "integer", default = 10, help = "sampling interval for structural properties"})
    sampling:add_argument("state-vars", {type = "integer", default = 10, help = "sampling interval for state variables"})

    return parser:parse_args()
end

local args = parse_args()

-- log to console
halmd.io.log.open_console({severity = args.verbose[1]})
-- log to file
halmd.io.log.open_file(("%s.log"):format(args.output), {severity = args.verbose[2]})
-- log version
halmd.utility.version.prologue()
-- initialise or disable GPU
halmd.utility.device({disable_gpu = args.disable_gpu, devices = args.devices})

-- run simulation
liquid(args)