/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2018- The GROMACS Authors
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
#include "gmxpre.h"

#include "gromacs/mdlib/ebin.h"

#include <cstdio>

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/textreader.h"
#include "gromacs/utility/unique_cptr.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{
namespace
{

//! Wraps fclose to discard the return value to use it as a deleter with gmx::unique_cptr.
void fcloseWrapper(FILE* fp)
{
    fclose(fp);
}

class PrEbinTest : public ::testing::Test
{
public:
    TestFileManager fileManager_;

    // TODO This will be more elegant (and run faster) when we
    // refactor the output routines to write to a stream
    // interface, which can already be handled in-memory when
    // running tests.
    std::string                      logFilename_;
    FILE*                            log_;
    unique_cptr<FILE, fcloseWrapper> logFileGuard_;

    TestReferenceData    refData_;
    TestReferenceChecker checker_;

    PrEbinTest() :
        logFilename_(fileManager_.getTemporaryFilePath(".log").string()),
        log_(std::fopen(logFilename_.c_str(), "w")),
        logFileGuard_(log_),
        checker_(refData_.rootChecker())
    {
    }
};

void expectEnergyBinsEqual(const t_ebin& reference, const t_ebin& candidate)
{
    ASSERT_EQ(reference.nener, candidate.nener);
    ASSERT_EQ(reference.nsum, candidate.nsum);
    ASSERT_EQ(reference.nsum_sim, candidate.nsum_sim);
    for (int i = 0; i < reference.nener; ++i)
    {
        EXPECT_DOUBLE_EQ(reference.e[i].e, candidate.e[i].e);
        EXPECT_DOUBLE_EQ(reference.e[i].eav, candidate.e[i].eav);
        EXPECT_DOUBLE_EQ(reference.e[i].esum, candidate.e[i].esum);
        EXPECT_DOUBLE_EQ(reference.e_sim[i].esum, candidate.e_sim[i].esum);
    }
}

TEST_F(PrEbinTest, HandlesAverages)
{
    ASSERT_NE(log_, nullptr);

    t_ebin*                        ebin = mk_ebin();
    unique_cptr<t_ebin, done_ebin> ebinGuard(ebin);

    // Set up the energy entries
    const char* firstName[]  = { "first" };
    const char* secondName[] = { "second" };
    int         first        = get_ebin_space(ebin, 1, firstName, nullptr);
    int         second       = get_ebin_space(ebin, 1, secondName, nullptr);

    // Put some data into the energy entries
    const real timevalues[2][2] = { { 1.0, 20.0 }, { 2.0, 40.0 } };
    gmx_bool   bSum             = true;
    for (const auto& values : timevalues)
    {
        add_ebin(ebin, first, 1, &values[0], bSum);
        add_ebin(ebin, second, 1, &values[1], bSum);
        ebin_increase_count(1, ebin, bSum);
    }

    // Test pr_ebin
    pr_ebin(log_, ebin, 0, 2, 5, eprAVER, true);

    // We need to close the file before the contents are available.
    logFileGuard_.reset(nullptr);

    checker_.checkInteger(ebin->nener, "Number of Energy Terms");
    checker_.checkString(TextReader::readFileToString(logFilename_), "log");
}

TEST_F(PrEbinTest, HandlesEmptyAverages)
{
    ASSERT_NE(log_, nullptr);

    t_ebin*                        ebin = mk_ebin();
    unique_cptr<t_ebin, done_ebin> ebinGuard(ebin);

    // Set up the energy entries
    const char* firstName[]  = { "first" };
    const char* secondName[] = { "second" };
    get_ebin_space(ebin, 1, firstName, nullptr);
    get_ebin_space(ebin, 1, secondName, nullptr);

    // Test pr_ebin
    pr_ebin(log_, ebin, 0, 2, 5, eprAVER, true);

    // We need to close the file before the contents are available.
    logFileGuard_.reset(nullptr);

    checker_.checkInteger(ebin->nener, "Number of Energy Terms");
    checker_.checkString(TextReader::readFileToString(logFilename_), "log");
}

TEST(EbinDeferredTest, MatchesSequentialContiguousUpdates)
{
    unique_cptr<t_ebin, done_ebin> reference(mk_ebin());
    unique_cptr<t_ebin, done_ebin> deferred(mk_ebin());
    const char*                    names[] = { "first", "second" };
    get_ebin_space(reference.get(), 2, names, nullptr);
    get_ebin_space(deferred.get(), 2, names, nullptr);

    constexpr std::array<std::array<real, 2>, 4> samples = {
        { { 1.0_real, 20.0_real }, { 2.0_real, 40.0_real }, { -3.0_real, 10.0_real }, { 8.0_real, -5.0_real } }
    };
    for (const auto& sample : samples)
    {
        add_ebin(reference.get(), 0, sample.size(), sample.data(), TRUE);
        ebin_increase_count(1, reference.get(), TRUE);
    }

    add_ebin(deferred.get(), 0, samples[0].size(), samples[0].data(), TRUE);
    ebin_increase_count(1, deferred.get(), TRUE);
    std::array<real, 6> deferredSamples;
    for (std::size_t sample = 1; sample < samples.size(); ++sample)
    {
        std::copy(samples[sample].begin(),
                  samples[sample].end(),
                  deferredSamples.begin() + (sample - 1) * samples[sample].size());
    }
    constexpr int numDeferredSamples = samples.size() - 1;
    ebin_increase_count(numDeferredSamples, deferred.get(), TRUE);
    add_ebin_deferred(deferred.get(), 0, 2, deferredSamples, numDeferredSamples);

    expectEnergyBinsEqual(*reference, *deferred);
}

TEST(EbinDeferredTest, MatchesSequentialIndexedUpdates)
{
    unique_cptr<t_ebin, done_ebin> reference(mk_ebin());
    unique_cptr<t_ebin, done_ebin> deferred(mk_ebin());
    const char*                    names[] = { "first", "second" };
    get_ebin_space(reference.get(), 2, names, nullptr);
    get_ebin_space(deferred.get(), 2, names, nullptr);

    std::array<bool, 4>                          shouldUse = { true, false, true, false };
    constexpr std::array<std::array<real, 4>, 4> samples   = {
        { { 1.0_real, 100.0_real, 20.0_real, 200.0_real },
            { 2.0_real, 101.0_real, 40.0_real, 201.0_real },
            { -3.0_real, 102.0_real, 10.0_real, 202.0_real },
            { 8.0_real, 103.0_real, -5.0_real, 203.0_real } }
    };
    for (const auto& sample : samples)
    {
        add_ebin_indexed(reference.get(), 0, shouldUse, sample, TRUE);
        ebin_increase_count(1, reference.get(), TRUE);
    }

    add_ebin_indexed(deferred.get(), 0, shouldUse, samples[0], TRUE);
    ebin_increase_count(1, deferred.get(), TRUE);
    std::array<real, 12> deferredSamples;
    for (std::size_t sample = 1; sample < samples.size(); ++sample)
    {
        std::copy(samples[sample].begin(),
                  samples[sample].end(),
                  deferredSamples.begin() + (sample - 1) * samples[sample].size());
    }
    constexpr int numDeferredSamples = samples.size() - 1;
    ebin_increase_count(numDeferredSamples, deferred.get(), TRUE);
    add_ebin_indexed_deferred(deferred.get(), 0, shouldUse, deferredSamples, numDeferredSamples);

    expectEnergyBinsEqual(*reference, *deferred);
}

} // namespace
} // namespace test
} // namespace gmx
