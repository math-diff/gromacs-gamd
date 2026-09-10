/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief Strict legacy GaMD parameter parsing and validation.
 */
#include "gmxpre.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>

#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "gromacs/utility/fatalerror.h"

#include "gamdruntime.h"

namespace gmx
{
namespace
{

void requireParameter(const std::set<std::string>& providedParameters, const char* requiredKey)
{
    if (providedParameters.count(requiredKey) == 0)
    {
        gmx_fatal(FARGS, "Required GaMD parameter '%s' is missing from gamd.in.", requiredKey);
    }
}

} // namespace

void GaMDRuntime::loadParameters(int rank)
{
    if (parametersLoaded_)
    {
        return;
    }

    std::ifstream input("gamd.in");
    if (!input.is_open())
    {
        gmx_fatal(FARGS, "GaMD is enabled but required parameter file 'gamd.in' was not found.");
    }

    std::set<std::string> providedParameters;
    std::string           lineText;
    std::string           parameterName;
    int64_t               lineNumber = 0;

    while (std::getline(input, lineText))
    {
        lineNumber++;
        std::istringstream lineStream(lineText);
        if (!(lineStream >> parameterName) || parameterName[0] == '#' || parameterName[0] == ';')
        {
            continue;
        }

        // 每行先检查重复键，再读取目标字段，最后统一检查格式。
        if (!providedParameters.insert(parameterName).second)
        {
            gmx_fatal(FARGS,
                      "Duplicate GaMD parameter '%s' at gamd.in line %" PRId64 ".",
                      parameterName.c_str(),
                      lineNumber);
        }

        if (parameterName == "igamd")
        {
            lineStream >> parameters_.boostType;
        }
        else if (parameterName == "irest_gamd")
        {
            lineStream >> parameters_.restartMode;
        }
        else if (parameterName == "iE")
        {
            lineStream >> parameters_.thresholdMode;
        }
        else if (parameterName == "iEP")
        {
            lineStream >> parameters_.potentialMode;
        }
        else if (parameterName == "iED")
        {
            lineStream >> parameters_.dihedralMode;
        }
        else if (parameterName == "ntcmdprep")
        {
            lineStream >> parameters_.conventionalPrepEnd;
        }
        else if (parameterName == "ntcmd")
        {
            lineStream >> parameters_.conventionalEnd;
        }
        else if (parameterName == "ntebprep")
        {
            lineStream >> parameters_.boostPrepLength;
        }
        else if (parameterName == "nteb")
        {
            lineStream >> parameters_.boostLength;
        }
        else if (parameterName == "ntave")
        {
            lineStream >> parameters_.averagingWindow;
        }
        else if (parameterName == "reweight_nst")
        {
            lineStream >> parameters_.reweightInterval;
        }
        else if (parameterName == "para_nst")
        {
            lineStream >> parameters_.parameterInterval;
        }
        else if (parameterName == "sigma0P")
        {
            double sigmaLimitKcal = 0.0;
            lineStream >> sigmaLimitKcal;
            parameters_.potentialSigmaLimit = sigmaLimitKcal * 4.184;
        }
        else if (parameterName == "sigma0D")
        {
            double sigmaLimitKcal = 0.0;
            lineStream >> sigmaLimitKcal;
            parameters_.dihedralSigmaLimit = sigmaLimitKcal * 4.184;
        }
        else if (parameterName == "debug")
        {
            lineStream >> debug_;
        }
        else
        {
            gmx_fatal(FARGS,
                      "Unknown GaMD parameter '%s' at gamd.in line %" PRId64 ".",
                      parameterName.c_str(),
                      lineNumber);
        }
        if (!lineStream)
        {
            gmx_fatal(FARGS,
                      "Missing or invalid value for GaMD parameter '%s' at gamd.in line %" PRId64
                      ".",
                      parameterName.c_str(),
                      lineNumber);
        }
        std::string trailing;
        if ((lineStream >> trailing) && trailing[0] != '#' && trailing[0] != ';')
        {
            gmx_fatal(FARGS,
                      "Unexpected trailing token '%s' for GaMD parameter '%s' at gamd.in line "
                      "%" PRId64 ".",
                      trailing.c_str(),
                      parameterName.c_str(),
                      lineNumber);
        }
    }
    if (input.bad())
    {
        gmx_fatal(FARGS, "Failed while reading required parameter file 'gamd.in'.");
    }

    requireParameter(providedParameters, "igamd");
    requireParameter(providedParameters, "iE");
    requireParameter(providedParameters, "reweight_nst");

    if (parameters_.boostType < 1 || parameters_.boostType > 3)
    {
        gmx_fatal(FARGS, "GaMD parameter igamd must be 1, 2, or 3; got %d.", parameters_.boostType);
    }
    if (parameters_.restartMode != 0 && parameters_.restartMode != 1)
    {
        gmx_fatal(FARGS, "GaMD parameter irest_gamd must be 0 or 1; got %d.", parameters_.restartMode);
    }
    if (parameters_.thresholdMode < 1 || parameters_.thresholdMode > 2 || parameters_.potentialMode < 1
        || parameters_.potentialMode > 2 || parameters_.dihedralMode < 1 || parameters_.dihedralMode > 2)
    {
        gmx_fatal(FARGS, "GaMD parameters iE, iEP, and iED must each be 1 or 2.");
    }
    if (parameters_.reweightInterval <= 0)
    {
        gmx_fatal(FARGS,
                  "GaMD parameter reweight_nst must be positive; got %" PRId64 ".",
                  parameters_.reweightInterval);
    }
    if (parameters_.parameterInterval < 0)
    {
        gmx_fatal(FARGS,
                  "GaMD parameter para_nst must be non-negative; got %" PRId64 ".",
                  parameters_.parameterInterval);
    }
    if (debug_ != 0 && debug_ != 1)
    {
        gmx_fatal(FARGS, "GaMD parameter debug must be 0 or 1; got %d.", debug_);
    }

    const bool usesPotentialBoost = (parameters_.boostType == 1 || parameters_.boostType == 3);
    const bool usesDihedralBoost  = (parameters_.boostType == 2 || parameters_.boostType == 3);
    if (usesPotentialBoost)
    {
        requireParameter(providedParameters, "iEP");
        requireParameter(providedParameters, "sigma0P");
        if (!std::isfinite(parameters_.potentialSigmaLimit) || parameters_.potentialSigmaLimit <= 0.0)
        {
            gmx_fatal(FARGS, "GaMD parameter sigma0P must be finite and positive.");
        }
    }
    if (usesDihedralBoost)
    {
        requireParameter(providedParameters, "iED");
        requireParameter(providedParameters, "sigma0D");
        if (!std::isfinite(parameters_.dihedralSigmaLimit) || parameters_.dihedralSigmaLimit <= 0.0)
        {
            gmx_fatal(FARGS, "GaMD parameter sigma0D must be finite and positive.");
        }
    }

    if (parameters_.restartMode == 0)
    {
        requireParameter(providedParameters, "ntcmdprep");
        requireParameter(providedParameters, "ntcmd");
        requireParameter(providedParameters, "ntebprep");
        requireParameter(providedParameters, "nteb");
        requireParameter(providedParameters, "ntave");

        if (parameters_.averagingWindow <= 0 || parameters_.conventionalPrepEnd < 0
            || parameters_.conventionalEnd <= parameters_.conventionalPrepEnd
            || parameters_.boostPrepLength < 0 || parameters_.boostLength <= parameters_.boostPrepLength)
        {
            gmx_fatal(FARGS,
                      "Invalid GaMD stage parameters: require ntave>0, 0<=ntcmdprep<ntcmd, "
                      "and 0<=ntebprep<nteb; got %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64
                      ", %" PRId64 ".",
                      parameters_.averagingWindow,
                      parameters_.conventionalPrepEnd,
                      parameters_.conventionalEnd,
                      parameters_.boostPrepLength,
                      parameters_.boostLength);
        }

        if (parameters_.conventionalPrepEnd % parameters_.averagingWindow != 0
            || parameters_.conventionalEnd % parameters_.averagingWindow != 0
            || parameters_.boostPrepLength % parameters_.averagingWindow != 0
            || parameters_.boostLength % parameters_.averagingWindow != 0)
        {
            gmx_fatal(FARGS,
                      "GaMD requires ntcmdprep (%" PRId64 "), ntcmd (%" PRId64
                      "), ntebprep (%" PRId64
                      "), and nteb "
                      "(%" PRId64 ") to be multiples of ntave (%" PRId64
                      ") for aligned statistics windows.",
                      parameters_.conventionalPrepEnd,
                      parameters_.conventionalEnd,
                      parameters_.boostPrepLength,
                      parameters_.boostLength,
                      parameters_.averagingWindow);
        }
    }

    parametersLoaded_ = true;

    if (rank == 0)
    {
        std::fprintf(stderr, "\n=== GaMD Parameters Loaded from gamd.in ===\n");
        std::fprintf(stderr, "igamd        = %d\n", parameters_.boostType);
        std::fprintf(stderr, "irest_gamd   = %d\n", parameters_.restartMode);
        std::fprintf(stderr, "iE           = %d\n", parameters_.thresholdMode);
        std::fprintf(stderr, "iEP          = %d\n", parameters_.potentialMode);
        std::fprintf(stderr, "iED          = %d\n", parameters_.dihedralMode);
        std::fprintf(stderr, "ntcmdprep    = %" PRId64 "\n", parameters_.conventionalPrepEnd);
        std::fprintf(stderr, "ntcmd        = %" PRId64 "\n", parameters_.conventionalEnd);
        std::fprintf(stderr, "ntebprep     = %" PRId64 "\n", parameters_.boostPrepLength);
        std::fprintf(stderr, "nteb         = %" PRId64 "\n", parameters_.boostLength);
        std::fprintf(stderr, "ntave        = %" PRId64 "\n", parameters_.averagingWindow);
        std::fprintf(stderr, "reweight_nst = %" PRId64 "\n", parameters_.reweightInterval);
        std::fprintf(stderr, "para_nst     = %" PRId64 "\n", parameters_.parameterInterval);
        std::fprintf(stderr, "sigma0P      = %.4f kJ/mol\n", parameters_.potentialSigmaLimit);
        std::fprintf(stderr, "sigma0D      = %.4f kJ/mol\n", parameters_.dihedralSigmaLimit);
        std::fprintf(stderr, "DEBUG MODE   = %d\n", debug_);
        std::fprintf(stderr, "===========================================\n\n");
    }
}

} // namespace gmx
