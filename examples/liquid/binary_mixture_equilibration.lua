#!/usr/bin/env halmd
--
-- Copyright © 2010-2012 Peter Colberg
-- Copyright © 2011-2012 Felix Höfling
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
local utility = halmd.utility

--
-- Setup and run simulation
--
local function liquid(args)
    -- total number of particles from sum of particles per species
    local nspecies = #args.particles
    local nparticle = 0
    for i = 1, nspecies do
        nparticle = nparticle + args.particles[i]
    end
    -- derive edge lengths from number of particles, density and edge ratios
    local volume = nparticle / args.density
    local dimension = #args.ratios
    local det = 1
    for i = 1, #args.ratios do
        det = det * args.ratios[i]
    end
    local length = {}
    for i = 1, #args.ratios do
        length[i] = args.ratios[i] * math.pow(volume / det, 1 / dimension)
    end
    -- create simulation domain with periodic boundary conditions
    local box = mdsim.box({length = length})

    -- create system state
    local particle = mdsim.particle({box = box, particles = nparticle, species = nspecies})
    -- set particle species, with continuous range of tags per species
    local species = {}
    local groups = {}
    local offset = 0
    for s = 0, nspecies - 1 do
        local nparticle = assert(args.particles[s + 1])
        for i = 1, nparticle do
            table.insert(species, s)
        end
        local label = string.char(string.byte("A") + s)
        groups[label] = mdsim.particle_groups.from_range({
            particle = particle
          , range = {offset + 1, offset + nparticle}
          , label = label
        })
        offset = offset + nparticle
    end
    particle:set_species(species)
    -- set initial particle positions
    local lattice = mdsim.positions.lattice({box = box, particle = particle})
    lattice:set()
    -- set initial particle velocities
    local boltzmann = mdsim.velocities.boltzmann({
        particle = particle
      , temperature = args.temperature
    })
    boltzmann:set()

    -- truncated Lennard-Jones potential
    local potential = mdsim.potentials.lennard_jones({
        particle = particle
      , epsilon = {
            {1  , 1.5} -- AA, AB
          , {1.5, 0.5} -- BA, BB
        }
      , sigma = {
            {1  , 0.8 } -- AA, AB
          , {0.8, 0.88} -- BA, BB
        }
      , cutoff = 2.5
    })
    -- smoothing at potential cutoff
    local trunc = mdsim.forces.trunc.local_r4({h = 0.005})
    -- compute forces
    local force = mdsim.forces.pair_trunc({
        box = box
      , particle = particle
      , potential = potential
      , trunc = trunc
    })

    -- H5MD file writer
    local writer = writers.h5md({path = ("%s.h5"):format(args.output)})
    -- write box specification to H5MD file
    box:writer(writer)

    -- select all particles
    local particle_group = mdsim.particle_groups.all({particle = particle})

    -- sample each particle group separately
    for label, group in utility.sorted(groups) do
        -- sample phase space
        local phase_space = observables.phase_space({box = box, group = group})
        -- write trajectory of particle groups to H5MD file
        phase_space:writer(writer, {every = args.sampling.trajectory})

        -- Sample macroscopic state variables.
        local msv = observables.thermodynamics({box = box, group = group, force = force})
        msv:writer(writer, {every = args.sampling.state_vars})
    end

    -- sample initial state
    observables.sampler:sample()

    -- add velocity-Verlet integrator with Boltzmann distribution
    local integrator = mdsim.integrators.verlet_boltzmann({
        box = box
      , group = particle_group
      , force = force
      , timestep = args.timestep
      , temperature = args.temperature
      , rate = args.rate
    })

    -- estimate remaining runtime
    local runtime = observables.runtime_estimate({
        steps = args.steps
      , first = 10
      , interval = 900
      , sample = 60
    })

    -- run simulation
    observables.sampler:run(args.steps)

    -- log profiler results
    halmd.utility.profiler:profile()
end

--
-- Parse command-line arguments.
--
local function parse_args()
    local parser = halmd.utility.program_options.argument_parser()

    parser:add_argument("output,o", {type = "string", action = function(args, key, value)
        -- substitute current time
        args[key] = os.date(value)
    end, default = "binary_mixture_equilibration_%Y%m%d_%H%M%S", help = "prefix of output files"})

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

    parser:add_argument("particles", {type = "vector", dtype = "integer", default = {4000, 1000}, help = "number of particles"})
    parser:add_argument("density", {type = "number", default = 1.2, help = "particle number density"})
    parser:add_argument("ratios", {type = "vector", dtype = "number", action = function(args, key, value)
        if #value ~= 2 and #value ~= 3 then
            error(("box ratios has invalid dimension '%d'"):format(#value), 0)
        end
        args[key] = value
    end, default = {1, 1, 1}, help = "relative aspect ratios of simulation box"})
    parser:add_argument("masses", {type = "vector", dtype = "number", default = {1}, help = "particle masses"})
    parser:add_argument("temperature", {type = "number", default = 0.7, help = "initial system temperature"})
    parser:add_argument("rate", {type = "number", default = 0.1, help = "heat bath collision rate"})
    parser:add_argument("steps", {type = "integer", default = 10000, help = "number of simulation steps"})
    parser:add_argument("timestep", {type = "number", default = 0.01, help = "integration time step"})

    local sampling = parser:add_argument_group("sampling", {help = "sampling intervals"})
    sampling:add_argument("trajectory", {type = "integer", default = 1000, help = "for trajectory"})
    sampling:add_argument("state-vars", {type = "integer", default = 1000, help = "for state variables"})

    return parser:parse_args()
end

local args = parse_args()

-- log to console
halmd.io.log.open_console({severity = args.verbose[1]})
-- log to file
halmd.io.log.open_file(("%s.log"):format(args.output), {severity = args.verbose[2]})
-- log version
halmd.utility.version.prologue()

-- run simulation
liquid(args)
