"""Reporter design of the Gosai 2024 episomal MPRA library.

Same shape as the ``*Library`` presets in :mod:`alphagenome_encoder_ft.constructs` — the
fixed pieces of one published reporter, plus the ``Construct`` that assembles them around an
insert — but kept with the scripts that use it rather than in the package.

The insert sits directly upstream of a minimal promoter driving GFP (Gosai et al. 2024,
*Nature*, Methods "MPRA library construction"):

    ... resulting into the 200 bp oligo sequence positioned directly upstream of the
    promoter and the 20 bp barcode falling in the 3' UTR of GFP.

The barcode is therefore far past any window worth feeding the encoder. Dissecting boda2's
``MPRA_DOWNSTREAM`` against that quote gives the cassette:

    offset    0- 27   vector spacer
             28- 78   polylinker (KpnI 28, SacI 35, NheI 41, XhoI 47, EcoRV 53, BglII 60, SfiI 66)
             79- 84   HindIII
             85-127   minP, TATA box at 98
            128-154   5' UTR + Kozak (GCCACC)
            155-299   EGFP coding sequence

The regulatory unit ends at offset 155, which is where ``REGULATORY_DOWNSTREAM_LENGTH``
stops: all of the promoter, none of the reporter ORF.
"""

from __future__ import annotations

from alphagenome_encoder_ft import Construct

# Reporter pieces, 5' to 3'. Concatenated they are boda2's MPRA_UPSTREAM / MPRA_DOWNSTREAM
# (`boda/common/constants.py`) verbatim; offsets below are within each 300 bp flank.

# Upstream flank: vector backbone ending in the SfiI cloning junction. Nothing regulatory.
_BACKBONE = (  # 0-278
    "ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAG"
    "GATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGT"
    "TGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGG"
    "CTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCT"
)
_SFII_UPSTREAM = "GGCCTAACTGGCC"  # 279-291, SfiI site (GGCCNNNNNGGCC)
_UPSTREAM_TAIL = "GCTTGACG"  # 292-299, abuts the insert

# Downstream flank: insert -> polylinker -> minP -> 5' UTR/Kozak -> EGFP.
_SPACER = "CACTGCGGCTCCTGCGATCTAACTGGCC"  # 0-27
_POLYLINKER = (  # 28-78: KpnI 28, SacI 35, NheI 41, XhoI 47, EcoRV 53, BglII 60, SfiI 66
    "GGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCC"
)
_HINDIII = "AAGCTT"  # 79-84
_MINP = "AGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGG"  # 85-127, TATA box at 98
_UTR_KOZAK = "CAATCCGGTACTGTTGGTAAAGCCACC"  # 128-154, Kozak GCCACC at 149
_EGFP_CDS = (  # 155-299, first 145 bp of the EGFP ORF
    "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAG"
    "TTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT"
)


class EpisomalMPRAGosai2024Library:
    """Reporter design of the episomal MPRA library in Gosai et al. 2024 (K562, HepG2, SK-N-SH)."""

    UPSTREAM = _BACKBONE + _SFII_UPSTREAM + _UPSTREAM_TAIL
    # Everything from the insert up to the GFP start codon: the regulatory unit.
    REGULATORY_DOWNSTREAM = _SPACER + _POLYLINKER + _HINDIII + _MINP + _UTR_KOZAK
    DOWNSTREAM = REGULATORY_DOWNSTREAM + _EGFP_CDS

    INSERT_LENGTH = 200
    REGULATORY_DOWNSTREAM_LENGTH = len(REGULATORY_DOWNSTREAM)  # 155
    INPUT_LENGTH = 384
    MALINOIS_INPUT_LENGTH = 600

    @classmethod
    def upstream_length(cls) -> int:
        """Vector backbone left over once the insert and the regulatory block are placed: 29 bp."""

        return cls.INPUT_LENGTH - cls.INSERT_LENGTH - cls.REGULATORY_DOWNSTREAM_LENGTH

    @classmethod
    def construct(cls) -> Construct:
        """``upstream vector + insert + polylinker/minP/5'UTR-Kozak`` -> 384 bp.

        Both 300 bp flanks are carried, and ``window_start`` fixes the window 29 bp before
        the insert. For a 200 bp insert the window is exactly 29 + 200 + 155, ending at the
        GFP start codon. Jitter slides the window over real vector upstream and EGFP
        downstream instead of padding with N. A shorter insert keeps the same 29 bp upstream
        and the window takes that much more of the downstream flank (EGFP), so it is never
        padded either. 384 bp pools to 3 encoder positions, giving a 3 x 1536 = 4608-wide
        flatten head.
        """

        return Construct(
            prefix=cls.UPSTREAM,
            suffix=cls.DOWNSTREAM,
            length=cls.INPUT_LENGTH,
            window_start=len(cls.UPSTREAM) - cls.upstream_length(),
        )

    @classmethod
    def malinois_construct(cls) -> Construct:
        """The 600 bp window Malinois trained on: both full flanks, the insert centred.

        Identical to boda2's rule for a 200 bp insert (200 bp of each flank kept). For a
        shorter insert the two differ by one base, because ``Construct`` takes the odd base
        of the trim off the downstream flank and boda2 takes it off the upstream one.
        """

        return Construct(
            prefix=cls.UPSTREAM,
            suffix=cls.DOWNSTREAM,
            length=cls.MALINOIS_INPUT_LENGTH,
        )


# Landmarks the cassette dissection above depends on: a cheap guard against an edited constant.
_LIB = EpisomalMPRAGosai2024Library
assert len(_LIB.UPSTREAM) == 300, len(_LIB.UPSTREAM)
assert len(_LIB.DOWNSTREAM) == 300, len(_LIB.DOWNSTREAM)
assert _LIB.DOWNSTREAM.find("TATATAAT") == 98, "TATA box moved - re-derive REGULATORY_DOWNSTREAM_LENGTH"
assert _LIB.DOWNSTREAM.find("GCCACC") == 149, "Kozak moved - re-derive REGULATORY_DOWNSTREAM_LENGTH"
assert _LIB.DOWNSTREAM[155:176] == "ATGGTGAGCAAGGGCGAGGAG", "GFP start moved"
assert _LIB.upstream_length() > 0, "INPUT_LENGTH leaves no room for the upstream vector"
