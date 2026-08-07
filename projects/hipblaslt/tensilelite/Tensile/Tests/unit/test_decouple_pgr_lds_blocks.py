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

``ldsBlocksForPgrLevel`` decides the whole LDS budget of a decoupled solution,
and the direction of its level-1 rung has been reversed twice. Nothing else
pins it down: the byte totals it produces are only asserted by out-of-tree
build scripts, so a silent change of the map is a silent change of every
decoupled kernel's LDS footprint.
"""
import pytest

from Tensile.SolutionStructs.Solution import (
    decouplePgrBlocks,
    decoupledSingleBuffered,
    ldsBlocksForPgrLevel,
)


# The per-tensor value is a block count, not a loop level: 0 and 1 are both a
# single block (1 prefetches into it, 0 does not), 2 is the conventional
# ping-pong pair, and 3 and up are taken literally the way the scalar
# derivation takes them.
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


# (decoupled, blocksA, blocksB). Absence of both keys is the "not specified"
# sentinel and falls back to the scalar; a single absent key falls back to the
# scalar for that tensor only.
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


# The write-after-read barriers fire for a one-block tensor sharing a loop with
# a two-block one. Equal counts do not need them: one block each pins
# 1LDSBuffer, whose own barrier covers it, and two blocks each ping-pong.
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
