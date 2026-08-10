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
"""Semantics of PrefetchGlobalReadA/B, the per-tensor LDS block counts.

Pure functions of a solution mapping, deliberately in Common rather than beside
the derivation in SolutionStructs.Solution: Components.SIA needs them too, and
SolutionStructs imports Component which imports Components, so anything SIA
reaches for has to live below both. Common already does.
"""


def pgrLevelsForTensors(ks):
    """(decoupled, pgrA, pgrB) -- the per-tensor levels, scalar-filled.

    An absent key is the "not specified" sentinel and falls back to the scalar
    PrefetchGlobalRead, which is why 0 is free to be a real value.
    """
    pgr = ks.get("PrefetchGlobalRead", 0)
    pgrA = ks.get("PrefetchGlobalReadA")
    pgrB = ks.get("PrefetchGlobalReadB")
    if pgrA is None and pgrB is None:
        return False, pgr, pgr
    return True, pgr if pgrA is None else pgrA, pgr if pgrB is None else pgrB


def ldsBlocksForPgrLevel(pgr):
    """LDS blocks one per-tensor level allocates.

      0  no prefetch, one block
      1  prefetch,    one block
      2  prefetch,    two blocks (ping-pong)
      k  k blocks, as the scalar derivation does from level 3 up

    Level 1 is the rung that does not agree with the scalar, which allocates
    TWO blocks for PrefetchGlobalRead=1: the buffer_load path also holds a VGPR
    staging buffer and sizes LDS against a three-buffer pipeline. Under TDM and
    DirectToLds there is no VGPR buffer, so N blocks is what depth N needs.
    """
    if pgr <= 1:
        return 1
    return 2 if pgr == 2 else pgr


def decouplePgrBlocks(ks):
    """(decoupled, numLdsBlkA, numLdsBlkB).

    Derived on demand rather than stored in solution state, so the counts
    cannot drift from the levels that define them and nothing derived reaches
    the serialized library.
    """
    decoupled, pgrA, pgrB = pgrLevelsForTensors(ks)
    return decoupled, ldsBlocksForPgrLevel(pgrA), ldsBlocksForPgrLevel(pgrB)


def decoupledSingleBuffered(ks):
    """True when exactly one tensor is left on a single LDS block.

    That tensor has nowhere to put its next tile except on top of the copy the
    current iteration is still reading, so the write-after-read barriers have
    to fire even though the scalar PrefetchGlobalRead is nonzero. Keyed on the
    resolved block counts, not the levels: the hazard follows the block count.
    """
    decoupled, numLdsBlkA, numLdsBlkB = decouplePgrBlocks(ks)
    return decoupled and min(numLdsBlkA, numLdsBlkB) == 1 and max(numLdsBlkA, numLdsBlkB) > 1


def decoupledOneBlockBoth(ks):
    """True when both tensors are on a single LDS block inside a prefetch loop.

    Same emit shape as 1LDSBuffer=1 but reached from the per-tensor block
    counts, which is what lets it exist at every ScheduleIterAlg -- 1LDSBuffer=1
    is rejected outside SIA 2 and 3 -- and what avoids needing a per-tensor
    1LDSBufferA/B alongside PrefetchGlobalReadA/B.

    PrefetchGlobalRead must be nonzero: at level 0 the no-prefetch branch
    already allocates a single block and NumLdsBlk stays at the 2 that legacy
    PrefetchGlobalRead=0 also reports.
    """
    decoupled, numLdsBlkA, numLdsBlkB = decouplePgrBlocks(ks)
    return decoupled and max(numLdsBlkA, numLdsBlkB) == 1 and bool(ks["PrefetchGlobalRead"])
