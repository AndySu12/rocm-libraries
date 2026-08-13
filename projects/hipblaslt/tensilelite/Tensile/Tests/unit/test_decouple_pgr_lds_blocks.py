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
"""
import pytest

from Tensile.Common.DecouplePgr import (
    decouplePgrBlocks,
    decoupledSingleBuffered,
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
