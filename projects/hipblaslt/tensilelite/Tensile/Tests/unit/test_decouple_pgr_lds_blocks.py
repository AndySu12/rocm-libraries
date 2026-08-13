################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Block-count map behind PrefetchGlobalReadA/B ("Decouple PGR", AIHPBLAS-4159).

``ldsBlocksForPgrLevel`` decides the whole LDS budget of a decoupled solution
and nothing else in-tree pins it down, so a silent change to the map is a
silent change to every decoupled kernel's LDS footprint.

The two guards below it are here for the same reason. Both were wrong on a
pushed branch, and both were wrong in the way a guard fails quietly: one let a
solution through validation to die on an emitter assertion, the other resolved
a pair away before the reject that contradicted it could see it. Neither shows
up in a build that only asks whether kernels came out.
"""
import pytest

from Tensile.Common.DecouplePgr import (
    decouplePgrBlocks,
    decoupledSingleBuffered,
    divergentPairUnsupportedReason,
    equalPairDegeneratesToScalar,
    ldsBlocksForPgrLevel,
)


# A block count, not a loop level: 0 and 1 are both a single block (1
# prefetches into it, 0 does not), 2 is the ping-pong pair.
@pytest.mark.parametrize(
    "level, blocks",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    ],
)
def test_lds_blocks_for_pgr_level(level, blocks):
    assert ldsBlocksForPgrLevel(level) == blocks


# (decoupled, blocksA, blocksB). An absent key falls back to the scalar, for
# that tensor only.
@pytest.mark.parametrize(
    "pgr, pgrA, pgrB, expected",
    [
        (2, None, None, (False, 2, 2)),
        (0, 0, 0, (True, 1, 1)),
        (1, 1, 1, (True, 1, 1)),
        (2, 2, 2, (True, 2, 2)),
        (2, 1, 2, (True, 1, 2)),
        (2, 2, 1, (True, 2, 1)),
        (2, 0, 2, (True, 1, 2)),
        (2, 2, 0, (True, 2, 1)),
        (2, 1, None, (True, 1, 2)),
        (2, None, 1, (True, 2, 1)),
    ],
)
def test_decouple_pgr_blocks(pgr, pgrA, pgrB, expected):
    ks = {"PrefetchGlobalRead": pgr}
    if pgrA is not None:
        ks["PrefetchGlobalReadA"] = pgrA
    if pgrB is not None:
        ks["PrefetchGlobalReadB"] = pgrB

    assert decouplePgrBlocks(ks) == expected


# Only a one-block tensor sharing a loop with a two-block one needs the
# write-after-read barriers. Equal counts do not: one block each is covered by
# decoupledOneBlockBoth, two blocks each ping-pong.
@pytest.mark.parametrize(
    "pgrA, pgrB, single",
    [
        (1, 2, True),
        (2, 1, True),
        (0, 2, True),
        (0, 0, False),
        (1, 1, False),
        (0, 1, False),
        (2, 2, False),
    ],
)
def test_decoupled_single_buffered(pgrA, pgrB, single):
    ks = {
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
    }

    assert decoupledSingleBuffered(ks) is single


def test_legacy_solution_is_not_decoupled():
    """No per-tensor key means the scalar owns the block count outright."""
    assert decoupledSingleBuffered({"PrefetchGlobalRead": 2}) is False


def _divergentSolution(**overrides):
    """A (1,2) pair with every precondition of the late fill satisfied."""
    ks = {
        "PrefetchGlobalRead": 1,
        "PrefetchGlobalReadA": 1,
        "PrefetchGlobalReadB": 2,
        "ScheduleIterAlg": 0,
        "PrefetchLocalRead": 1,
        "NumWaves": 4,
    }
    ks.update(overrides)
    return ks


# Each of these is a precondition of relocating the single-buffered tensor's
# fill, and each has to come back as a reject: a divergent pair that reaches
# KernelWriter._dcpScheduleSingleBufferedFillLate without one dies on a bare
# assertion, which takes down the whole Tensile invocation instead of dropping
# the one solution that is unbuildable.
@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, None),
        ({"PrefetchGlobalReadB": 3}, "more than two LDS blocks"),
        ({"ScheduleIterAlg": 3}, "ScheduleIterAlg=0"),
        ({"PrefetchLocalRead": 0}, "PrefetchLocalRead must be at least 1"),
        ({"NumWaves": 1}, "NumWaves > 1"),
    ],
)
def test_divergent_pair_unsupported_reason(overrides, expected):
    reason = divergentPairUnsupportedReason(_divergentSolution(**overrides))

    if expected is None:
        assert reason is None
    else:
        assert reason is not None
        assert expected in reason


def test_divergent_pair_at_one_wave_is_rejected_not_asserted():
    """NumWaves=1 is the precondition nothing used to check.

    Parity selects which tensor a wave fills, and parity means nothing with one
    wave -- KernelWriterAssembly.isTdmWaveSeparated is false there, which is
    exactly what the emitter asserts. Read this as: the guard exists at all, so
    the failure is a reject on one solution rather than an AssertionError on the
    sweep that contained it.
    """
    assert divergentPairUnsupportedReason(_divergentSolution(NumWaves=1)) is not None
    assert divergentPairUnsupportedReason(_divergentSolution(NumWaves=2)) is None


# 1LDSBuffer=1 gives every tensor one shared LDS block, which contradicts any
# pair asking for two. Degenerating an equal pair deletes both keys, so the
# reject that catches the contradiction never sees it and the solution silently
# builds the one block -- byte-identical to the legacy kernel and half the LDS
# that was asked for. Only an explicit 1 holds the pair back; -1 is the auto
# rule, which legacy PrefetchGlobalRead=k gets too.
@pytest.mark.parametrize(
    "pgrA, pgrB, oneLdsBuffer, degenerates",
    [
        (2, 2, 0, True),
        (2, 2, -1, True),
        (2, 2, None, True),
        (2, 2, 1, False),
        (0, 0, 0, True),
        (0, 0, 1, True),
        (1, 1, 0, False),
        (1, 1, 1, False),
        (1, 2, 0, False),
        (1, 2, 1, False),
    ],
)
def test_equal_pair_degenerates_to_scalar(pgrA, pgrB, oneLdsBuffer, degenerates):
    ks = {
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
    }
    if oneLdsBuffer is not None:
        ks["1LDSBuffer"] = oneLdsBuffer

    assert equalPairDegeneratesToScalar(ks) is degenerates


def test_legacy_solution_does_not_degenerate():
    """There is no pair to resolve away, whatever 1LDSBuffer says."""
    assert equalPairDegeneratesToScalar({"PrefetchGlobalRead": 2, "1LDSBuffer": 1}) is False
