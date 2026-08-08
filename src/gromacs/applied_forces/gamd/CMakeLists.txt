#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2026 The GROMACS Authors
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#

# Link to mdtypes (for GaMDParams)
target_link_libraries(applied_forces INTERFACE mdtypes)
gmx_add_libgromacs_sources(
    gamdmdmodule.cpp
    gamdcontroller.cpp
    gamdexecution.cpp
)
