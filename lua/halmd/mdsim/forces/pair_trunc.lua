--
-- Copyright © 2010  Peter Colberg and Felix Höfling
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

require("halmd.modules")

-- grab environment
local pair_trunc_wrapper = {
    host = {
        [2] = halmd_wrapper.mdsim.host.forces.pair_trunc_2_
      , [3] = halmd_wrapper.mdsim.host.forces.pair_trunc_3_
    }
}
if halmd_wrapper.mdsim.gpu then
    pair_trunc_wrapper.gpu = {
        [2] = halmd_wrapper.mdsim.gpu.forces.pair_trunc_2_
      , [3] = halmd_wrapper.mdsim.gpu.forces.pair_trunc_3_
    }
end
local mdsim = {
    forces = {
        lennard_jones = require("halmd.mdsim.forces.lennard_jones")
      , morse = require("halmd.mdsim.forces.morse")
      , power_law = require("halmd.mdsim.forces.power_law")
    }
  , core = require("halmd.mdsim.core")
}
local device = require("halmd.device")
local options = require("halmd.options")
local assert = assert
local hooks = require("halmd.hooks")

module("halmd.mdsim.forces.pair_trunc", halmd.modules.register)

--
-- construct truncated pair force module
--
function new(args)
    local dimension = assert(options.dimension)
    local force = assert(args.force)

    -- dependency injection
    local core = mdsim.core()
    local particle = assert(core.particle)
    local box = assert(core.box)
    local potential = mdsim.forces[force]()

    local pair_trunc
    if device() then
        pair_trunc = assert(pair_trunc_wrapper.gpu[dimension][force])
    else
        pair_trunc = assert(pair_trunc_wrapper.host[dimension][force])
    end
    return pair_trunc(potential, particle, box)
end