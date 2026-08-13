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

In Common rather than beside the derivation in SolutionStructs.Solution because
Components.SIA needs these too, and SolutionStructs imports Component which
imports Components, so anything SIA reaches for has to live below both.
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

    Level 1 is the rung that does not agree with the scalar, which allocates TWO
    blocks for PrefetchGlobalRead=1: the buffer_load path also holds a VGPR
    staging buffer and sizes LDS against a three-buffer pipeline. Under TDM and
    DirectToLds there is no VGPR buffer, so N blocks is what depth N needs.
    """
    if pgr <= 1:
        return 1
    return 2 if pgr == 2 else pgr


def tdmBothTensors(ks):
    """True when the TDM moves both tensors, which the per-tensor levels need.

    TDMInst is a per-tensor bitmask, bit 0 for A and bit 1 for B, read that way
    rather than compared against 3 so this does not depend on the separate
    reject that pins the parameter to 0 or 3.

    A block count is a prefetch depth only where nothing stages the tile in
    VGPRs first, which is why this is the precondition for the whole feature and
    not just for the shapes that misbuild without it.
    """
    tdmInst = ks.get("TDMInst", 0)
    return bool(tdmInst & 0x01) and bool(tdmInst & 0x02)


def decouplePgrBlocks(ks):
    """(decoupled, numLdsBlkA, numLdsBlkB).

    Derived on demand rather than stored in solution state, so nothing derived
    reaches the serialized library.
    """
    decoupled, pgrA, pgrB = pgrLevelsForTensors(ks)
    return decoupled, ldsBlocksForPgrLevel(pgrA), ldsBlocksForPgrLevel(pgrB)


def decoupledSingleBuffered(ks):
    """True when exactly one tensor is left on a single LDS block.

    That tensor has nowhere to put its next tile except on top of the copy the
    current iteration is still reading, so the write-after-read barriers have to
    fire even though the scalar PrefetchGlobalRead is nonzero.
    """
    decoupled, numLdsBlkA, numLdsBlkB = decouplePgrBlocks(ks)
    return decoupled and min(numLdsBlkA, numLdsBlkB) == 1 and max(numLdsBlkA, numLdsBlkB) > 1


def tdmDealiasAB(ks):
    """True when A and B get their own TDM descriptor sets instead of sharing one.

    Costs 12 SGPRs -- Group0 is 4 and Group1 is 8, the fixed tuple widths of
    tensor_load_to_lds -- against an architectural ceiling of 106, paid for by
    closing runtime StaggerU in _disableUnsupportedRuntimeStaggerU.

    Never derived, only selected by TDMFuse=6, so that 0 stays inert. Equal
    block counts keep the alias, because their byte-identity with a legacy
    configuration is the evidence the feature rests on. MXSA/MXSB stay
    parity-aliased; de-aliasing all four costs another 24 SGPRs. TDMSplit keeps
    the alias, because its multi-wave increment recomputes one parity-selected
    split stride for one shared descriptor.
    """
    if ks.get("TDMFuse") != 6:
        return False
    decoupled, numLdsBlkA, numLdsBlkB = decouplePgrBlocks(ks)
    if not (decoupled and numLdsBlkA != numLdsBlkB):
        return False
    if not tdmBothTensors(ks):
        return False
    if ks.get("TDMSplit"):
        return False
    return ks.get("NumWaves", 1) > 1 and not ks.get("UseSubtileImpl")


def decoupledOneBlockBoth(ks):
    """True when both tensors are on a single LDS block inside a prefetch loop.

    Same emit shape as 1LDSBuffer=1 but reached from the per-tensor block counts,
    which is what lets it exist at every ScheduleIterAlg -- 1LDSBuffer=1 is
    rejected outside SIA 2 and 3 -- and what avoids needing a per-tensor
    1LDSBufferA/B alongside PrefetchGlobalReadA/B.

    PrefetchGlobalRead must be nonzero: at level 0 the no-prefetch branch already
    allocates a single block and NumLdsBlk stays at the 2 that legacy
    PrefetchGlobalRead=0 also reports.
    """
    decoupled, numLdsBlkA, numLdsBlkB = decouplePgrBlocks(ks)
    return decoupled and max(numLdsBlkA, numLdsBlkB) == 1 and bool(ks["PrefetchGlobalRead"])
