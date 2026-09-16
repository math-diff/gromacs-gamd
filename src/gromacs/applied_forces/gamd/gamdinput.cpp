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
 * \brief Reads and validates legacy GaMD input with native text utilities.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include <cinttypes>
#include <cmath>

#include <limits>
#include <set>
#include <string>

#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/path.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textreader.h"

#include "gamddiagnostics.h"
#include "gamdruntime.h"

//! \cond internal
namespace gmx
{
namespace
{

/*! \internal \brief Rejects a missing mandatory legacy input key. */
void requireParameter(const std::set<std::string>& providedParameters, const char* requiredKey)
{
    if (providedParameters.count(requiredKey) == 0)
    {
        GMX_THROW(InvalidInputError(
                formatString("Required GaMD parameter '%s' is missing from gamd.in.", requiredKey)));
    }
}

} // namespace

std::vector<std::string> splitGaMDInputLine(const std::string& line)
{
    // substr 接受 npos，既支持无注释的整行，也支持紧跟数值的 # / ; 注释。
    return splitString(line.substr(0, line.find_first_of("#;")));
}

std::string gaMDStageParameterError(const GaMDParameters& parameters)
{
    // 先保证除数和各阶段长度合法，后面才能安全做取余和边界相加。
    if (parameters.averagingWindow <= 0 || parameters.conventionalPrepEnd < 0
        || parameters.conventionalEnd <= parameters.conventionalPrepEnd
        || parameters.boostPrepLength < 0 || parameters.boostLength <= parameters.boostPrepLength)
    {
        return "Invalid GaMD stage parameters: require ntave>0, 0<=ntcmdprep<ntcmd, "
               "and 0<=ntebprep<nteb.";
    }
    if (parameters.conventionalEnd > std::numeric_limits<int64_t>::max() - parameters.boostLength)
    {
        return "GaMD stage boundary ntcmd + nteb exceeds the signed 64-bit step range.";
    }
    if (parameters.conventionalPrepEnd % parameters.averagingWindow != 0
        || parameters.conventionalEnd % parameters.averagingWindow != 0
        || parameters.boostPrepLength % parameters.averagingWindow != 0
        || parameters.boostLength % parameters.averagingWindow != 0)
    {
        return "GaMD requires ntcmdprep, ntcmd, ntebprep, and nteb to be multiples of ntave "
               "for aligned statistics windows.";
    }
    return {};
}

void GaMDRuntime::loadParameters(int rank)
{
    if (parametersLoaded_)
    {
        return;
    }

    if (!File::exists("gamd.in", File::returnFalseOnError))
    {
        GMX_THROW(FileIOError(
                "GaMD is enabled but required parameter file 'gamd.in' was not found."));
    }
    TextReader            input("gamd.in");
    std::set<std::string> providedParameters;
    std::string           lineText;
    int64_t               lineNumber = 0;

    while (input.readLine(&lineText))
    {
        lineNumber++;
        const std::vector<std::string> fields = splitGaMDInputLine(lineText);
        if (fields.empty())
        {
            continue;
        }
        const std::string& parameterName = fields[0];
        if (!providedParameters.insert(parameterName).second)
        {
            GMX_THROW(InvalidInputError(
                    formatString("Duplicate GaMD parameter '%s' at gamd.in line %" PRId64 ".",
                                 parameterName.c_str(),
                                 lineNumber)));
        }

        // 统一使用项目的严格数值转换，失败时补上参数名和文件行号。
        const char* value = "";
        if (fields.size() >= 2)
        {
            value = fields[1].c_str();
        }
        bool recognized = true;
        try
        {
            if (parameterName == "igamd")
            {
                parameters_.boostType = intFromString(value);
            }
            else if (parameterName == "irest_gamd")
            {
                parameters_.restartMode = intFromString(value);
            }
            else if (parameterName == "iE")
            {
                parameters_.thresholdMode = intFromString(value);
            }
            else if (parameterName == "iEP")
            {
                parameters_.potentialMode = intFromString(value);
            }
            else if (parameterName == "iED")
            {
                parameters_.dihedralMode = intFromString(value);
            }
            else if (parameterName == "ntcmdprep")
            {
                parameters_.conventionalPrepEnd = int64FromString(value);
            }
            else if (parameterName == "ntcmd")
            {
                parameters_.conventionalEnd = int64FromString(value);
            }
            else if (parameterName == "ntebprep")
            {
                parameters_.boostPrepLength = int64FromString(value);
            }
            else if (parameterName == "nteb")
            {
                parameters_.boostLength = int64FromString(value);
            }
            else if (parameterName == "ntave")
            {
                parameters_.averagingWindow = int64FromString(value);
            }
            else if (parameterName == "reweight_nst")
            {
                parameters_.reweightInterval = int64FromString(value);
            }
            else if (parameterName == "para_nst")
            {
                parameters_.parameterInterval = int64FromString(value);
            }
            else if (parameterName == "sigma0P")
            {
                parameters_.potentialSigmaLimit = doubleFromString(value) * 4.184;
            }
            else if (parameterName == "sigma0D")
            {
                parameters_.dihedralSigmaLimit = doubleFromString(value) * 4.184;
            }
            else if (parameterName == "debug")
            {
                debug_ = intFromString(value);
            }
            else
            {
                recognized = false;
            }
        }
        catch (const InvalidInputError&)
        {
            GMX_THROW(InvalidInputError(formatString(
                    "Missing or invalid value for GaMD parameter '%s' at gamd.in line %" PRId64 ".",
                    parameterName.c_str(),
                    lineNumber)));
        }
        if (!recognized)
        {
            GMX_THROW(InvalidInputError(formatString(
                    "Unknown GaMD parameter '%s' at gamd.in line %" PRId64 ".", parameterName.c_str(), lineNumber)));
        }
        if (fields.size() > 2)
        {
            GMX_THROW(
                    InvalidInputError(formatString("Unexpected trailing token '%s' for GaMD "
                                                   "parameter '%s' at gamd.in line %" PRId64 ".",
                                                   fields[2].c_str(),
                                                   parameterName.c_str(),
                                                   lineNumber)));
        }
    }
    input.close();

    requireParameter(providedParameters, "igamd");
    requireParameter(providedParameters, "iE");
    requireParameter(providedParameters, "reweight_nst");

    if (parameters_.boostType < 1 || parameters_.boostType > 3)
    {
        GMX_THROW(InvalidInputError(formatString("GaMD parameter igamd must be 1, 2, or 3; got %d.",
                                                 parameters_.boostType)));
    }
    if (parameters_.restartMode != 0 && parameters_.restartMode != 1)
    {
        GMX_THROW(InvalidInputError(formatString(
                "GaMD parameter irest_gamd must be 0 or 1; got %d.", parameters_.restartMode)));
    }
    if (parameters_.thresholdMode < 1 || parameters_.thresholdMode > 2 || parameters_.potentialMode < 1
        || parameters_.potentialMode > 2 || parameters_.dihedralMode < 1 || parameters_.dihedralMode > 2)
    {
        GMX_THROW(InvalidInputError("GaMD parameters iE, iEP, and iED must each be 1 or 2."));
    }
    if (parameters_.reweightInterval <= 0)
    {
        GMX_THROW(InvalidInputError(
                formatString("GaMD parameter reweight_nst must be positive; got %" PRId64 ".",
                             parameters_.reweightInterval)));
    }
    if (parameters_.parameterInterval < 0)
    {
        GMX_THROW(InvalidInputError(
                formatString("GaMD parameter para_nst must be non-negative; got %" PRId64 ".",
                             parameters_.parameterInterval)));
    }
    if (debug_ != 0 && debug_ != 1)
    {
        GMX_THROW(InvalidInputError(formatString("GaMD parameter debug must be 0 or 1; got %d.", debug_)));
    }

    if (!std::isfinite(parameters_.potentialSigmaLimit) || !std::isfinite(parameters_.dihedralSigmaLimit))
    {
        GMX_THROW(InvalidInputError("GaMD sigma0P and sigma0D must be finite."));
    }

    const bool usesPotentialBoost = (parameters_.boostType == 1 || parameters_.boostType == 3);
    const bool usesDihedralBoost  = (parameters_.boostType == 2 || parameters_.boostType == 3);
    if (usesPotentialBoost)
    {
        requireParameter(providedParameters, "iEP");
        requireParameter(providedParameters, "sigma0P");
        if (!std::isfinite(parameters_.potentialSigmaLimit) || parameters_.potentialSigmaLimit <= 0.0)
        {
            GMX_THROW(InvalidInputError("GaMD parameter sigma0P must be finite and positive."));
        }
    }
    if (usesDihedralBoost)
    {
        requireParameter(providedParameters, "iED");
        requireParameter(providedParameters, "sigma0D");
        if (!std::isfinite(parameters_.dihedralSigmaLimit) || parameters_.dihedralSigmaLimit <= 0.0)
        {
            GMX_THROW(InvalidInputError("GaMD parameter sigma0D must be finite and positive."));
        }
    }

    if (parameters_.restartMode == 0)
    {
        requireParameter(providedParameters, "ntcmdprep");
        requireParameter(providedParameters, "ntcmd");
        requireParameter(providedParameters, "ntebprep");
        requireParameter(providedParameters, "nteb");
        requireParameter(providedParameters, "ntave");

        const std::string stageError = gaMDStageParameterError(parameters_);
        if (!stageError.empty())
        {
            GMX_THROW(InvalidInputError(stageError));
        }
    }

    parametersLoaded_ = true;

    if (rank == 0)
    {
        GMX_LOG(diagnostics_.logger().info)
                .asParagraph()
                .appendTextFormatted(
                        "GaMD parameters loaded from gamd.in:\n"
                        "igamd = %d, irest_gamd = %d, iE = %d, iEP = %d, iED = %d\n"
                        "ntcmdprep = %" PRId64 ", ntcmd = %" PRId64 ", ntebprep = %" PRId64
                        ", nteb = %" PRId64 ", ntave = %" PRId64
                        "\n"
                        "reweight_nst = %" PRId64 ", para_nst = %" PRId64
                        "\n"
                        "sigma0P = %.4f kJ/mol, sigma0D = %.4f kJ/mol, debug = %d",
                        parameters_.boostType,
                        parameters_.restartMode,
                        parameters_.thresholdMode,
                        parameters_.potentialMode,
                        parameters_.dihedralMode,
                        parameters_.conventionalPrepEnd,
                        parameters_.conventionalEnd,
                        parameters_.boostPrepLength,
                        parameters_.boostLength,
                        parameters_.averagingWindow,
                        parameters_.reweightInterval,
                        parameters_.parameterInterval,
                        parameters_.potentialSigmaLimit,
                        parameters_.dihedralSigmaLimit,
                        debug_);
    }
}

} // namespace gmx
//! \endcond
