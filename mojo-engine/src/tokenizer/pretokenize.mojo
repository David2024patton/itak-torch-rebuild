# pretokenize.mojo - the pre-tokenizer: a hand-written scanner
# implementing the llama4 / GPT-4o split pattern, a direct port
# of crates/tokenizer/src/pretokenize.rs (see that file and
# crates/tokenizer/src/lib.rs for the pattern text). BPE runs per
# piece, never across piece boundaries. The split is lossless:
# concatenating the pieces reproduces the input exactly.
#
# The pattern, as the Rust crate documents it:
#
#   [^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'s|'t|'re|'ve|'m|'ll|'d)?
#   | [^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'s|'t|'re|'ve|'m|'ll|'d)?
#   | \\p{N}{1,3}
#   |  ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*
#   | \\s*[\\r\\n]+
#   | \\s+(?!\S)
#   | \\s+
#
# The two word alternatives span the same maximal letter run, so
# the scanner implements them as one letter run plus the optional
# contraction suffix. There is no standalone contraction
# alternative: a contraction that does not follow a word (the 've
# in you'd've, the 's in 90's) is picked up by the prefix-char
# word branch instead.
#
# Deviations, mirroring the Rust crate's stated ones and adding
# this port's own:
# - Rust approximates \\p{L} by char::is_alphabetic and \\p{N} by
#   char::is_numeric, and \\s by char::is_whitespace (the exact
#   Unicode White_Space property, reproduced exactly below).
# - Mojo has no Unicode property tables, so is_alphabetic /
#   is_numeric below are curated codepoint range tables covering
#   the major scripts (Latin incl. extensions, Greek, Cyrillic,
#   Armenian, Hebrew, Arabic, Indic, Thai, CJK, kana, Hangul,
#   Ethiopic, Georgian, and the other blocks listed in the
#   ranges). Combining marks U+0300..=U+036F count as letters,
#   matching the pattern's \\p{M} inside word runs. Digits are
#   classified before letters, so a digit block inside a broad
#   script range still classifies as a number. Rare scripts and
#   newer Unicode additions can classify as symbols where Rust
#   says letter (or the reverse at block edges); splitting is
#   always lossless either way. The 16-prompt Glimmer parity set
#   (Latin, accented Latin, CJK, kana, digits, emoji, symbols)
#   sits entirely inside the covered ranges.

from tokenizer.byte_level import codepoints_of


def is_whitespace_cp(cp: Int) -> Bool:
    """The Unicode White_Space property, exactly (Rust's
    char::is_whitespace set)."""
    if 0x09 <= cp and cp <= 0x0D:
        return True
    if cp == 0x20 or cp == 0x85 or cp == 0xA0 or cp == 0x1680:
        return True
    if 0x2000 <= cp and cp <= 0x200A:
        return True
    if cp == 0x2028 or cp == 0x2029 or cp == 0x202F or cp == 0x205F:
        return True
    return cp == 0x3000


def is_numeric_cp(cp: Int) -> Bool:
    """Approximation of \\p{N} (Nd, Nl, No): Rust's
    char::is_numeric. Decimal blocks for the major scripts, the
    letter-number blocks, and the common other-number forms."""
    # Nd: ASCII and major script decimal digits.
    if 0x30 <= cp and cp <= 0x39:
        return True
    if 0x660 <= cp and cp <= 0x669:
        return True
    if 0x6F0 <= cp and cp <= 0x6F9:
        return True
    if 0x7C0 <= cp and cp <= 0x7C9:
        return True
    if 0x966 <= cp and cp <= 0x96F:
        return True
    if 0x9E6 <= cp and cp <= 0x9EF:
        return True
    if 0xA66 <= cp and cp <= 0xA6F:
        return True
    if 0xAE6 <= cp and cp <= 0xAEF:
        return True
    if 0xB66 <= cp and cp <= 0xB6F:
        return True
    if 0xBE6 <= cp and cp <= 0xBEF:
        return True
    if 0xC66 <= cp and cp <= 0xC6F:
        return True
    if 0xCE6 <= cp and cp <= 0xCEF:
        return True
    if 0xD66 <= cp and cp <= 0xD6F:
        return True
    if 0xDE6 <= cp and cp <= 0xDEF:
        return True
    if 0xE50 <= cp and cp <= 0xE59:
        return True
    if 0xED0 <= cp and cp <= 0xED9:
        return True
    if 0xF20 <= cp and cp <= 0xF29:
        return True
    if 0x1040 <= cp and cp <= 0x1049:
        return True
    if 0x1090 <= cp and cp <= 0x1099:
        return True
    if 0x17E0 <= cp and cp <= 0x17E9:
        return True
    if 0x1810 <= cp and cp <= 0x1819:
        return True
    if 0x1946 <= cp and cp <= 0x194F:
        return True
    if 0x19D0 <= cp and cp <= 0x19D9:
        return True
    if 0x1A80 <= cp and cp <= 0x1A89:
        return True
    if 0x1A90 <= cp and cp <= 0x1A99:
        return True
    if 0x1B50 <= cp and cp <= 0x1B59:
        return True
    if 0x1BB0 <= cp and cp <= 0x1BB9:
        return True
    if 0x1C40 <= cp and cp <= 0x1C49:
        return True
    if 0x1C50 <= cp and cp <= 0x1C59:
        return True
    if 0xA620 <= cp and cp <= 0xA629:
        return True
    if 0xA8D0 <= cp and cp <= 0xA8D9:
        return True
    if 0xA900 <= cp and cp <= 0xA909:
        return True
    if 0xA9D0 <= cp and cp <= 0xA9D9:
        return True
    if 0xA9F0 <= cp and cp <= 0xA9F9:
        return True
    if 0xAA50 <= cp and cp <= 0xAA59:
        return True
    if 0xABF0 <= cp and cp <= 0xABF9:
        return True
    if 0xFF10 <= cp and cp <= 0xFF19:
        return True
    if 0x1D7CE <= cp and cp <= 0x1D7FF:
        return True
    # Nl: letter numbers (Roman numerals and kin).
    if 0x2160 <= cp and cp <= 0x2188:
        return True
    if 0x3007 == cp or cp == 0x3021 or cp == 0x3022 or cp == 0x3023:
        return True
    if 0x3024 <= cp and cp <= 0x3029:
        return True
    if cp == 0x3038 or cp == 0x3039 or cp == 0x303A:
        return True
    # No: other numbers (fractions, superscripts, circled runs).
    if cp == 0xB2 or cp == 0xB3 or cp == 0xB9:
        return True
    if cp == 0xBC or cp == 0xBD or cp == 0xBE:
        return True
    if 0x2070 <= cp and cp <= 0x2079 and cp != 0x2071:
        # U+2071 is a letter (superscript n); 0x2070, 0x2074..79
        # are No. U+2072/U+2073 are unassigned.
        if cp == 0x2070 or (0x2074 <= cp and cp <= 0x2079):
            return True
        return False
    if 0x2080 <= cp and cp <= 0x2089:
        return True
    if 0x2460 <= cp and cp <= 0x249B:
        return True
    if 0x24EA == cp:
        return True
    if 0x00BD == cp:
        return True
    return False


def is_letter_cp(cp: Int) -> Bool:
    """Approximation of the Alphabetic property: Rust's
    char::is_alphabetic. Curated ranges; see the module header
    for the deviation note. Digits are checked before this in
    classify(), so broad script blocks may span digit subranges.
    """
    if 0x41 <= cp and cp <= 0x5A:
        return True
    if 0x61 <= cp and cp <= 0x7A:
        return True
    if cp == 0xAA or cp == 0xB5 or cp == 0xBA:
        return True
    if 0xC0 <= cp and cp <= 0xFF and cp != 0xD7 and cp != 0xF7:
        return True
    if 0x100 <= cp and cp <= 0x24F:
        return True  # Latin Extended-A/B
    if 0x250 <= cp and cp <= 0x2AF:
        return True  # IPA extensions (letters)
    if 0x2B0 <= cp and cp <= 0x2C1:
        return True  # spacing modifier letters
    if 0x300 <= cp and cp <= 0x36F:
        return True  # combining marks: the pattern's \\p{M}
    if 0x370 <= cp and cp <= 0x373:
        return True
    if 0x375 <= cp and cp <= 0x377:
        return True
    if 0x37B <= cp and cp <= 0x37D:
        return True
    if cp == 0x37F:
        return True
    if 0x386 == cp:
        return True
    if 0x388 <= cp and cp <= 0x38A:
        return True
    if cp == 0x38C:
        return True
    if 0x38E <= cp and cp <= 0x3A1:
        return True
    if 0x3A3 <= cp and cp <= 0x3F5:
        return True
    if 0x3F7 <= cp and cp <= 0x481:
        return True  # Greek remainder + Cyrillic start
    if 0x48A <= cp and cp <= 0x52F:
        return True
    if 0x531 <= cp and cp <= 0x556:
        return True
    if cp == 0x559:
        return True
    if 0x561 <= cp and cp <= 0x587:
        return True  # Armenian
    if 0x5D0 <= cp and cp <= 0x5EA:
        return True
    if 0x5F0 <= cp and cp <= 0x5F2:
        return True  # Hebrew letters (points classify as symbols)
    if 0x621 <= cp and cp <= 0x64A:
        return True
    if 0x66E <= cp and cp <= 0x6D3:
        return True
    if 0x6D5 <= cp and cp <= 0x6DC:
        return True
    if 0x6E5 <= cp and cp <= 0x6E6:
        return True
    if 0x6EE <= cp and cp <= 0x6FC:
        return True
    if 0x6FF == cp:
        return True
    if 0x710 <= cp and cp <= 0x74A:
        return True  # Syriac
    if 0x74D <= cp and cp <= 0x7B1:
        return True
    if 0x7C0 <= cp and cp <= 0x7EA:
        return True  # Nko (digits checked first)
    if 0x800 <= cp and cp <= 0x815:
        return True  # Samaritan
    if 0x840 <= cp and cp <= 0x858:
        return True  # Mandaic
    if 0x860 <= cp and cp <= 0x86A:
        return True
    if 0x8A0 <= cp and cp <= 0x8B4:
        return True  # Arabic extended
    if 0x904 <= cp and cp <= 0x939:
        return True
    if 0x93D == cp:
        return True
    if 0x950 == cp:
        return True
    if 0x958 <= cp and cp <= 0x961:
        return True
    if 0x971 <= cp and cp <= 0x980:
        return True
    if 0x985 <= cp and cp <= 0x98C:
        return True
    if 0x98F <= cp and cp <= 0x990:
        return True
    if 0x993 <= cp and cp <= 0x9A8:
        return True
    if 0x9AA <= cp and cp <= 0x9B0:
        return True
    if cp == 0x9B2:
        return True
    if 0x9B6 <= cp and cp <= 0x9B9:
        return True  # Bengali letters (vowel signs are symbols)
    if 0xA05 <= cp and cp <= 0xA0A:
        return True
    if 0xA0F <= cp and cp <= 0xA10:
        return True
    if 0xA13 <= cp and cp <= 0xA28:
        return True
    if 0xA2A <= cp and cp <= 0xA30:
        return True
    if 0xA32 <= cp and cp <= 0xA33:
        return True
    if 0xA35 <= cp and cp <= 0xA36:
        return True
    if 0xA38 <= cp and cp <= 0xA39:
        return True
    if 0xA59 <= cp and cp <= 0xA5C:
        return True
    if cp == 0xA5E:
        return True
    if 0xA72 <= cp and cp <= 0xA74:
        return True  # Gurmukhi / Gujarati heads
    if 0xA85 <= cp and cp <= 0xA8D:
        return True
    if 0xA8F <= cp and cp <= 0xA91:
        return True
    if 0xA93 <= cp and cp <= 0xAA8:
        return True
    if 0xAAA <= cp and cp <= 0xAB0:
        return True
    if 0xAB2 <= cp and cp <= 0xAB9:
        return True  # Gujarati
    if 0xB05 <= cp and cp <= 0xB0C:
        return True
    if 0xB0F <= cp and cp <= 0xB10:
        return True
    if 0xB13 <= cp and cp <= 0xB28:
        return True
    if 0xB2A <= cp and cp <= 0xB30:
        return True
    if 0xB32 <= cp and cp <= 0xB33:
        return True
    if 0xB35 <= cp and cp <= 0xB39:
        return True
    if 0xB5C <= cp and cp <= 0xB5D:
        return True
    if 0xB5F <= cp and cp <= 0xB61:
        return True  # Oriya
    if 0xB85 <= cp and cp <= 0xB8A:
        return True
    if 0xB8E <= cp and cp <= 0xB90:
        return True
    if 0xB92 <= cp and cp <= 0xB95:
        return True
    if 0xB99 <= cp and cp <= 0xB9A:
        return True
    if cp == 0xB9C:
        return True
    if 0xB9E <= cp and cp <= 0xB9F:
        return True
    if 0xBA3 <= cp and cp <= 0xBA4:
        return True
    if 0xBA8 <= cp and cp <= 0xBAA:
        return True
    if 0xBAE <= cp and cp <= 0xBB9:
        return True  # Tamil
    if 0xC05 <= cp and cp <= 0xC0C:
        return True
    if 0xC0E <= cp and cp <= 0xC10:
        return True
    if 0xC12 <= cp and cp <= 0xC28:
        return True
    if 0xC2A <= cp and cp <= 0xC33:
        return True
    if 0xC35 <= cp and cp <= 0xC39:
        return True
    if 0xC60 <= cp and cp <= 0xC61:
        return True  # Telugu
    if 0xC85 <= cp and cp <= 0xC8C:
        return True
    if 0xC8E <= cp and cp <= 0xC90:
        return True
    if 0xC92 <= cp and cp <= 0xCA8:
        return True
    if 0xCAA <= cp and cp <= 0xCB3:
        return True
    if 0xCB5 <= cp and cp <= 0xCB9:
        return True
    if 0xCDE == cp:
        return True
    if 0xCE0 <= cp and cp <= 0xCE1:
        return True  # Kannada
    if 0xD05 <= cp and cp <= 0xD0C:
        return True
    if 0xD0E <= cp and cp <= 0xD10:
        return True
    if 0xD12 <= cp and cp <= 0xD3A:
        return True
    if 0xD60 <= cp and cp <= 0xD61:
        return True  # Malayalam
    if 0xD81 <= cp and cp <= 0xD83:
        return True
    if 0xD85 <= cp and cp <= 0xD96:
        return True
    if 0xD9A <= cp and cp <= 0xDB1:
        return True
    if 0xDB3 <= cp and cp <= 0xDBB:
        return True
    if cp == 0xDBD:
        return True
    if 0xDC0 <= cp and cp <= 0xDC6:
        return True  # Sinhala
    if 0xE01 <= cp and cp <= 0xE30:
        return True
    if 0xE32 <= cp and cp <= 0xE33:
        return True
    if 0xE40 <= cp and cp <= 0xE46:
        return True  # Thai letters (digits checked first)
    if 0xE81 <= cp and cp <= 0xE82:
        return True
    if cp == 0xE84:
        return True
    if 0xE87 <= cp and cp <= 0xE88:
        return True
    if cp == 0xE8A:
        return True
    if cp == 0xE8D:
        return True
    if 0xE94 <= cp and cp <= 0xE97:
        return True
    if 0xE99 <= cp and cp <= 0xE9F:
        return True
    if 0xEA1 <= cp and cp <= 0xEA3:
        return True
    if cp == 0xEA5:
        return True
    if cp == 0xEA7:
        return True
    if 0xEAA <= cp and cp <= 0xEAB:
        return True
    if 0xEAD <= cp and cp <= 0xEB0:
        return True
    if 0xEB2 <= cp and cp <= 0xEB3:
        return True
    if cp == 0xEBD:
        return True
    if 0xEC0 <= cp and cp <= 0xEC4:
        return True
    if cp == 0xEC6:
        return True  # Lao
    if 0xF00 == cp:
        return True
    if 0xF40 <= cp and cp <= 0xF47:
        return True
    if 0xF49 <= cp and cp <= 0xF6C:
        return True
    if 0xF88 <= cp and cp <= 0xF8C:
        return True  # Tibetan letters
    if 0x1000 <= cp and cp <= 0x102A:
        return True
    if 0x103F == cp:
        return True
    if 0x1050 <= cp and cp <= 0x1055:
        return True
    if 0x105A <= cp and cp <= 0x105D:
        return True
    if 0x1061 == cp:
        return True
    if 0x1065 <= cp and cp <= 0x1066:
        return True
    if 0x106E <= cp and cp <= 0x1070:
        return True
    if 0x1075 <= cp and cp <= 0x1081:
        return True
    if cp == 0x108E:
        return True  # Myanmar (digits checked first)
    if 0x10A0 <= cp and cp <= 0x10C5:
        return True
    if 0x10C7 == cp or cp == 0x10CD:
        return True
    if 0x10D0 <= cp and cp <= 0x10FA:
        return True
    if cp == 0x10FC:
        return True
    if 0x10FF == cp:
        return True  # Georgian
    if 0x1100 <= cp and cp <= 0x1248:
        return True  # Hangul Jamo + Ethiopic head
    if 0x124A <= cp and cp <= 0x124D:
        return True
    if 0x1250 <= cp and cp <= 0x1256:
        return True
    if cp == 0x1258:
        return True
    if 0x125A <= cp and cp <= 0x125D:
        return True
    if 0x1260 <= cp and cp <= 0x1288:
        return True
    if 0x128A <= cp and cp <= 0x128D:
        return True
    if 0x1290 <= cp and cp <= 0x12B0:
        return True
    if 0x12B2 <= cp and cp <= 0x12B5:
        return True
    if 0x12B8 <= cp and cp <= 0x12BE:
        return True
    if cp == 0x12C0:
        return True
    if 0x12C2 <= cp and cp <= 0x12C5:
        return True
    if 0x12C8 <= cp and cp <= 0x12D6:
        return True
    if 0x12D8 <= cp and cp <= 0x1310:
        return True
    if 0x1312 <= cp and cp <= 0x1315:
        return True
    if 0x1318 <= cp and cp <= 0x135A:
        return True  # Ethiopic
    if 0x1380 <= cp and cp <= 0x138F:
        return True
    if 0x13A0 <= cp and cp <= 0x13F4:
        return True  # Cherokee
    if 0x1401 <= cp and cp <= 0x166C:
        return True
    if 0x166F <= cp and cp <= 0x167F:
        return True  # Canadian Aboriginal
    if 0x1681 <= cp and cp <= 0x169A:
        return True  # Ogham
    if 0x16A0 <= cp and cp <= 0x16EA:
        return True  # Runic
    if 0x16EE <= cp and cp <= 0x16F8:
        return True  # Runic numbers are Nl: numeric check ran first
    if 0x1700 <= cp and cp <= 0x170C:
        return True
    if 0x170E <= cp and cp <= 0x1711:
        return True  # Tagalog
    if 0x1720 <= cp and cp <= 0x1731:
        return True  # Hanunoo
    if 0x1740 <= cp and cp <= 0x1751:
        return True  # Buhid
    if 0x1760 <= cp and cp <= 0x176C:
        return True
    if 0x176E <= cp and cp <= 0x1770:
        return True  # Tagbanwa
    if 0x1780 <= cp and cp <= 0x17B3:
        return True  # Khmer (digits checked first)
    if 0x1820 <= cp and cp <= 0x1877:
        return True
    if 0x1880 <= cp and cp <= 0x18A8:
        return True  # Mongolian
    if 0x18AA == cp:
        return True
    if 0x18B0 <= cp and cp <= 0x18F5:
        return True  # Limbu / Tai Le / New Tai Lue letters
    if 0x1900 <= cp and cp <= 0x191E:
        return True
    if 0x1950 <= cp and cp <= 0x196D:
        return True
    if 0x1970 <= cp and cp <= 0x1974:
        return True  # Tai Tham / Tai Le
    if 0x1980 <= cp and cp <= 0x19AB:
        return True
    if 0x19C1 <= cp and cp <= 0x19C7:
        return True  # New Tai Lue / Tai Tham
    if 0x1A20 <= cp and cp <= 0x1A5E:
        return True  # Tai Tham letters
    if 0x1AA7 == cp:
        return True
    if 0x1B05 <= cp and cp <= 0x1B33:
        return True
    if 0x1B45 <= cp and cp <= 0x1B4B:
        return True  # Balinese (digits checked first)
    if 0x1B83 <= cp and cp <= 0x1BA0:
        return True
    if 0x1BA2 <= cp and cp <= 0x1BA7:
        return True
    if 0x1BAA == cp:
        return True
    if 0x1BC0 <= cp and cp <= 0x1BE5:
        return True  # Sundanese / Batak
    if 0x1C00 <= cp and cp <= 0x1C23:
        return True
    if 0x1C4D <= cp and cp <= 0x1C4F:
        return True  # Lepcha
    if 0x1C5A <= cp and cp <= 0x1C7D:
        return True  # Ol Chiki (digits checked first)
    if 0x1CE9 <= cp and cp <= 0x1CEC:
        return True
    if 0x1CEE <= cp and cp <= 0x1CF1:
        return True
    if cp == 0x1CF4:
        return True
    if 0x1CF8 <= cp and cp <= 0x1CF9:
        return True  # Vedic extensions that are letters
    if 0x2135 <= cp and cp <= 0x2138:
        return True  # letterlike symbols that are alphabetic
    if 0x2C00 <= cp and cp <= 0x2C2E:
        return True  # Glagolitic
    if 0x2C30 <= cp and cp <= 0x2C5E:
        return True
    if 0x2C60 <= cp and cp <= 0x2C7B:
        return True
    if 0x2C80 <= cp and cp <= 0x2CE4:
        return True  # Coptic
    if 0x2D00 <= cp and cp <= 0x2D25:
        return True
    if cp == 0x2D27:
        return True
    if cp == 0x2D2D:
        return True  # Georgian supplement
    if 0x2D30 <= cp and cp <= 0x2D67:
        return True
    if 0x2D6F == cp:
        return True
    if 0x2D80 <= cp and cp <= 0x2D96:
        return True
    if 0x2DA0 <= cp and cp <= 0x2DA6:
        return True
    if 0x2DA8 <= cp and cp <= 0x2DAE:
        return True
    if 0x2DB0 <= cp and cp <= 0x2DB6:
        return True
    if 0x2DB8 <= cp and cp <= 0x2DBE:
        return True
    if 0x2DC0 <= cp and cp <= 0x2DC6:
        return True
    if 0x2DC8 <= cp and cp <= 0x2DCE:
        return True
    if 0x2DD0 <= cp and cp <= 0x2DD6:
        return True
    if 0x2DD8 <= cp and cp <= 0x2DDE:
        return True  # Ethiopic extended
    if 0x3005 <= cp and cp <= 0x3007:
        return True  # CJK iteration marks / ideographic zero (Nl first)
    if cp == 0x3029 or cp == 0x302A:
        return True
    if 0x3031 <= cp and cp <= 0x3035:
        return True
    if 0x3038 <= cp and cp <= 0x303C:
        return True
    if 0x3041 <= cp and cp <= 0x3096:
        return True
    if 0x309D <= cp and cp <= 0x309F:
        return True  # Hiragana
    if 0x30A1 <= cp and cp <= 0x30FA:
        return True
    if 0x30FC <= cp and cp <= 0x30FF:
        return True  # Katakana
    if 0x3105 <= cp and cp <= 0x312D:
        return True  # Bopomofo
    if 0x3131 <= cp and cp <= 0x318E:
        return True  # Hangul compatibility Jamo
    if 0x31A0 <= cp and cp <= 0x31BA:
        return True  # Bopomofo extended
    if 0x31F0 <= cp and cp <= 0x31FF:
        return True  # Katakana phonetic extensions
    if 0x3400 <= cp and cp <= 0x4DB5:
        return True  # CJK Ext A
    if 0x4E00 <= cp and cp <= 0x9FD5:
        return True  # CJK Unified
    if 0xA000 <= cp and cp <= 0xA48C:
        return True  # Yi syllables
    if 0xA717 <= cp and cp <= 0xA71F:
        return True
    if cp == 0xA788:
        return True
    if 0xA7F8 <= cp and cp <= 0xA801:
        return True
    if 0xA803 == cp:
        return True
    if 0xA807 == cp:
        return True
    if 0xA80C == cp:
        return True
    if 0xA815 <= cp and cp <= 0xA81B:
        return True
    if 0xA81E <= cp and cp <= 0xA822:
        return True  # Latin/Syloti heads
    if 0xA840 <= cp and cp <= 0xA873:
        return True  # Phags-pa
    if 0xA882 <= cp and cp <= 0xA8B3:
        return True  # Saurashtra
    if 0xA8F2 <= cp and cp <= 0xA8F7:
        return True
    if cp == 0xA8FB:
        return True
    if 0xA90A <= cp and cp <= 0xA925:
        return True  # Kayah Li (digits checked first)
    if 0xA930 <= cp and cp <= 0xA946:
        return True  # Rejang
    if 0xA960 <= cp and cp <= 0xA97C:
        return True  # Hangul Jamo Ext-A
    if 0xA984 <= cp and cp <= 0xA9B2:
        return True  # Javanese
    if 0xA9E0 <= cp and cp <= 0xA9E4:
        return True
    if 0xA9E6 <= cp and cp <= 0xA9EF:
        return True
    if 0xA9FA <= cp and cp <= 0xA9FE:
        return True  # Myanmar extended
    if 0xAA00 <= cp and cp <= 0xAA28:
        return True
    if 0xAA2A <= cp and cp <= 0xAA2E:
        return True
    if 0xAA30 <= cp and cp <= 0xAA33:
        return True
    if 0xAA35 <= cp and cp <= 0xAA36:
        return True
    if 0xAA40 <= cp and cp <= 0xAA42:
        return True
    if cp == 0xAA44:
        return True
    if 0xAA47 <= cp and cp <= 0xAA4B:
        return True  # Cham
    if 0xAA60 <= cp and cp <= 0xAA76:
        return True
    if cp == 0xAA7A:
        return True
    if 0xAA80 <= cp and cp <= 0xAAAF:
        return True
    if 0xAAB1 == cp:
        return True
    if 0xAAB5 <= cp and cp <= 0xAAB6:
        return True
    if cp == 0xAAB9 or cp == 0xAABD:
        return True
    if 0xAABF <= cp and cp <= 0xAAC0:
        return True
    if cp == 0xAAC2:
        return True  # Tai Viet
    if 0xAAC0 <= cp and cp <= 0xAAC2:
        return True
    if 0xAADB <= cp and cp <= 0xAADD:
        return True
    if 0xABC0 <= cp and cp <= 0xABE2:
        return True  # Meetei Mayek
    if 0xAC00 <= cp and cp <= 0xD7A3:
        return True  # Hangul syllables
    if 0xD7B0 <= cp and cp <= 0xD7C6:
        return True
    if 0xD7CB <= cp and cp <= 0xD7FB:
        return True  # Hangul Jamo Ext-B
    if 0xF900 <= cp and cp <= 0xFA6D:
        return True
    if 0xFA70 <= cp and cp <= 0xFAD9:
        return True  # CJK compatibility ideographs
    if 0xFB00 <= cp and cp <= 0xFB06:
        return True
    if 0xFB13 <= cp and cp <= 0xFB17:
        return True
    if cp == 0xFB1D:
        return True
    if 0xFB1F <= cp and cp <= 0xFB28:
        return True
    if 0xFB2A <= cp and cp <= 0xFB36:
        return True
    if 0xFB38 <= cp and cp <= 0xFB3C:
        return True
    if cp == 0xFB3E:
        return True
    if 0xFB40 <= cp and cp <= 0xFB41:
        return True
    if 0xFB43 <= cp and cp <= 0xFB44:
        return True
    if 0xFB46 <= cp and cp <= 0xFBB1:
        return True
    if 0xFBD3 <= cp and cp <= 0xFD3D:
        return True
    if 0xFD50 <= cp and cp <= 0xFD8F:
        return True
    if 0xFD92 <= cp and cp <= 0xFDC7:
        return True
    if 0xFDF0 <= cp and cp <= 0xFDFB:
        return True
    if 0xFE70 <= cp and cp <= 0xFE74:
        return True
    if 0xFE76 <= cp and cp <= 0xFEFC:
        return True  # Alphabetic presentation forms
    if 0xFF21 <= cp and cp <= 0xFF3A:
        return True
    if 0xFF41 <= cp and cp <= 0xFF5A:
        return True
    if 0xFF66 <= cp and cp <= 0xFFBE:
        return True
    if 0xFFC2 <= cp and cp <= 0xFFC7:
        return True
    if 0xFFCA <= cp and cp <= 0xFFCF:
        return True
    if 0xFFD2 <= cp and cp <= 0xFFD7:
        return True
    if 0xFFDA <= cp and cp <= 0xFFDC:
        return True  # Fullwidth / halfwidth letters
    if 0x10000 <= cp and cp <= 0x100FA:
        return True  # Linear B
    if 0x10280 <= cp and cp <= 0x1029C:
        return True  # Lycian
    if 0x102A0 <= cp and cp <= 0x102D0:
        return True  # Carian
    if 0x10300 <= cp and cp <= 0x10323:
        return True  # Old Italic
    if 0x10330 <= cp and cp <= 0x1034A:
        return True  # Gothic
    if 0x10380 <= cp and cp <= 0x1039D:
        return True  # Ugaritic
    if 0x103A0 <= cp and cp <= 0x103C3:
        return True
    if 0x103C8 <= cp and cp <= 0x103CF:
        return True
    if 0x103D1 <= cp and cp <= 0x103D5:
        return True  # Old Persian
    if 0x10400 <= cp and cp <= 0x1049D:
        return True  # Deseret + Shavian + Osmanya heads
    if 0x104A0 <= cp and cp <= 0x104A9:
        return False  # Osmanya digits: numeric check ran first
    if 0x104B0 <= cp and cp <= 0x104D8:
        return True
    if 0x104DC <= cp and cp <= 0x104FB:
        return True  # Osage
    if 0x10530 <= cp and cp <= 0x10563:
        return True  # Elbasan / Caucasian Albanian / Vithkuqi heads
    if 0x10570 <= cp and cp <= 0x1057A:
        return True
    if 0x1057C <= cp and cp <= 0x1058A:
        return True
    if 0x1058C <= cp and cp <= 0x10592:
        return True
    if 0x10594 <= cp and cp <= 0x10595:
        return True
    if 0x10600 <= cp and cp <= 0x10736:
        return True
    if 0x10740 <= cp and cp <= 0x10755:
        return True
    if 0x10760 <= cp and cp <= 0x10767:
        return True
    if 0x10780 <= cp and cp <= 0x107BA:
        return True  # Latin extended astral
    if 0x10A60 <= cp and cp <= 0x10A7C:
        return True
    if 0x10A7E == cp:
        return True  # Old South Arabian
    if 0x10B40 <= cp and cp <= 0x10B55:
        return True
    if 0x10B58 <= cp and cp <= 0x10B72:
        return True
    if 0x10B78 <= cp and cp <= 0x10B7F:
        return True  # Avestan / Inscriptional
    if 0x10C80 <= cp and cp <= 0x10CB2:
        return True
    if 0x10CC0 <= cp and cp <= 0x10CF2:
        return True
    if 0x10CF9 <= cp and cp <= 0x10CFF:
        return True  # Old Hungarian / Old Turkic
    if 0x10D40 <= cp and cp <= 0x10D85:
        return True  # Hanifi Rohingya (digits checked first)
    if 0x10E80 <= cp and cp <= 0x10EA9:
        return True
    if 0x10EAB <= cp and cp <= 0x10EAC:
        return True
    if 0x10EAD == cp:
        return True  # Yezidi
    if 0x10F00 <= cp and cp <= 0x10F27:
        return True  # Old Georgian / Nuskhuri
    if 0x10F70 <= cp and cp <= 0x10F81:
        return True  # Old Uyghur
    if 0x11003 <= cp and cp <= 0x11037:
        return True
    if 0x11071 == cp:
        return True
    if 0x11075 <= cp and cp <= 0x11079:
        return True
    if 0x1107F <= cp and cp <= 0x11081:
        return True
    if 0x11083 <= cp and cp <= 0x110AF:
        return True  # Brahmi / Kaithi heads (numbers checked first)
    if 0x110D0 <= cp and cp <= 0x110E8:
        return True  # Sora Sompeng
    if 0x11103 <= cp and cp <= 0x11126:
        return True
    if 0x1112C <= cp and cp <= 0x11134:
        return True  # Chakma
    if 0x11147 <= cp and cp <= 0x11172:
        return True  # Sharada
    if 0x11176 <= cp and cp <= 0x111DA:
        return True
    if 0x111DC == cp:
        return True  # Sharada / Sinhala archaic
    if 0x111E1 <= cp and cp <= 0x111F4:
        return True
    if 0x11200 <= cp and cp <= 0x11211:
        return True
    if 0x11213 <= cp and cp <= 0x1122B:
        return True  # Khojki
    if 0x11280 <= cp and cp <= 0x11286:
        return True
    if 0x11288 == cp:
        return True
    if 0x1128A <= cp and cp <= 0x1128D:
        return True
    if 0x1128F <= cp and cp <= 0x1129D:
        return True
    if 0x1129F <= cp and cp <= 0x112A8:
        return True  # Multani
    if 0x112B0 <= cp and cp <= 0x112DE:
        return True  # Khudawadi
    if 0x11305 <= cp and cp <= 0x1130C:
        return True
    if 0x1130F <= cp and cp <= 0x11310:
        return True
    if 0x11313 <= cp and cp <= 0x11328:
        return True
    if 0x1132A <= cp and cp <= 0x11330:
        return True
    if 0x11332 <= cp and cp <= 0x11333:
        return True
    if 0x11335 <= cp and cp <= 0x11339:
        return True
    if cp == 0x1133D:
        return True
    if 0x11350 <= cp and cp <= 0x1134D:
        return True  # Grantha
    if 0x11400 <= cp and cp <= 0x11434:
        return True
    if 0x11447 <= cp and cp <= 0x1144A:
        return True  # Newa
    if 0x11480 <= cp and cp <= 0x114AF:
        return True
    if 0x114C4 <= cp and cp <= 0x114C5:
        return True
    if cp == 0x114C7:
        return True  # Tirhuta
    if 0x11580 <= cp and cp <= 0x115AE:
        return True
    if 0x115D8 <= cp and cp <= 0x115DB:
        return True  # Siddham
    if 0x11600 <= cp and cp <= 0x1162F:
        return True
    if 0x11633 <= cp and cp <= 0x1163A:
        return True
    if cp == 0x1163D:
        return True
    if 0x1163F <= cp and cp <= 0x11640:
        return True
    if 0x11644 == cp:
        return True  # Modi
    if 0x11680 <= cp and cp <= 0x116B2:
        return True  # Takri
    if 0x11700 <= cp and cp <= 0x1171A:
        return True  # Ahom
    if 0x11800 <= cp and cp <= 0x1182B:
        return True
    if 0x1182C <= cp and cp <= 0x1183A:
        return True  # Dogra
    if 0x118A0 <= cp and cp <= 0x118F2:
        return True
    if 0x118FF <= cp and cp <= 0x11908:
        return True  # Warang Citi / Dives Akuru heads
    if 0x11909 == cp:
        return True
    if 0x1190C <= cp and cp <= 0x11913:
        return True
    if 0x11915 <= cp and cp <= 0x11916:
        return True
    if 0x11918 <= cp and cp <= 0x11935:
        return True
    if 0x11937 <= cp and cp <= 0x11938:
        return True
    if 0x1193B <= cp and cp <= 0x1193C:
        return True
    if cp == 0x1193E:
        return True
    if 0x11940 == cp:
        return True
    if 0x11942 <= cp and cp <= 0x11943:
        return True  # Dives Akuru
    if 0x119A0 <= cp and cp <= 0x119A7:
        return True
    if 0x119AA <= cp and cp <= 0x119D0:
        return True
    if 0x119D4 <= cp and cp <= 0x119D7:
        return True
    if 0x119DA <= cp and cp <= 0x119DB:
        return True
    if 0x119E0 == cp:
        return True  # Nandinagari
    if 0x11A00 <= cp and cp <= 0x11A0A:
        return True  # Zanabazar
    if 0x11A47 <= cp and cp <= 0x11A56:
        return True
    if 0x11A59 <= cp and cp <= 0x11A82:
        return True  # Soyombo
    if 0x11ABF <= cp and cp <= 0x11AC0:
        return True
    if 0x11AC2 <= cp and cp <= 0x11ACE:
        return True  # Devanagari extended astral
    if 0x11B00 <= cp and cp <= 0x11B05:
        return True
    if 0x11B09 == cp:
        return True  # Devanagari Extended-A
    if 0x11C08 <= cp and cp <= 0x11C0B:
        return True
    if 0x11C0D == cp:
        return True
    if 0x11C10 <= cp and cp <= 0x11C13:
        return True
    if 0x11C16 <= cp and cp <= 0x11C22:
        return True
    if 0x11C24 <= cp and cp <= 0x11C2B:
        return True
    if 0x11C2E <= cp and cp <= 0x11C34:
        return True
    if 0x11C38 <= cp and cp <= 0x11C3D:
        return True
    if cp == 0x11C3F:
        return True
    if 0x11C40 == cp:
        return True
    if 0x11C72 <= cp and cp <= 0x11C8E:
        return True  # Bhaiksuki / Marchen
    if 0x11C92 <= cp and cp <= 0x11CA7:
        return True
    if 0x11CA9 <= cp and cp <= 0x11CB0:
        return True
    if 0x11CB2 <= cp and cp <= 0x11CB3:
        return True
    if 0x11CB5 <= cp and cp <= 0x11CB6:
        return True  # Masaram Gondi
    if 0x11D08 <= cp and cp <= 0x11D09:
        return True
    if 0x11D0B <= cp and cp <= 0x11D30:
        return True
    if 0x11D46 <= cp and cp <= 0x11D47:
        return True  # Gunjala Gondi / Makasar heads
    if 0x11D60 <= cp and cp <= 0x11D65:
        return True
    if 0x11D67 <= cp and cp <= 0x11D68:
        return True
    if 0x11D6A <= cp and cp <= 0x11D89:
        return True  # Kawi
    if 0x11EE0 <= cp and cp <= 0x11EF2:
        return True  # Arabic astral (presentation forms covered above)
    if 0x11F00 <= cp and cp <= 0x11F10:
        return True
    if 0x11F12 <= cp and cp <= 0x11F33:
        return True
    if 0x11F3E <= cp and cp <= 0x11F4F:
        return True  # Kawi letters
    if 0x1342F <= cp and cp <= 0x1343B:
        return True
    if 0x13460 <= cp and cp <= 0x143FA:
        return True  # Egyptian hieroglyphs (Lo in Unicode)
    if 0x16E40 <= cp and cp <= 0x16E7F:
        return True  # Medefaidrin
    if 0x16A40 <= cp and cp <= 0x16A5E:
        return True  # Miao
    if 0x16B00 <= cp and cp <= 0x16B2F:
        return True
    if 0x16B40 <= cp and cp <= 0x16B43:
        return True  # Pahawh Hmong
    if 0x16C80 <= cp and cp <= 0x16CAD:
        return True  # Tangut components are not letters; skip
    if 0x18800 <= cp and cp <= 0x18CD5:
        return True  # Tangut (Lo)
    if 0x1AFF0 <= cp and cp <= 0x1AFFB:
        return True
    if 0x1AFFD <= cp and cp <= 0x1AFFE:
        return True
    if 0x1B000 <= cp and cp <= 0x1B122:
        return True  # Kana extended (mostly Lo)
    if 0x1B150 <= cp and cp <= 0x1B152:
        return True
    if 0x1B164 <= cp and cp <= 0x1B167:
        return True
    if 0x1B170 <= cp and cp <= 0x1B2FB:
        return True  # Nushu
    if 0x1BC70 <= cp and cp <= 0x1BC88:
        return True  # Duployan
    if 0x1BFD0 <= cp and cp <= 0x1BFEF:
        return False
    if 0x1D400 <= cp and cp <= 0x1D454:
        return True
    if 0x1D456 <= cp and cp <= 0x1D49C:
        return True
    if 0x1D49E <= cp and cp <= 0x1D4A9:
        return True
    if 0x1D4AC == cp:
        return True
    if 0x1D4AE <= cp and cp <= 0x1D4B9:
        return True
    if 0x1D4BB == cp:
        return True
    if 0x1D4BD <= cp and cp <= 0x1D4C3:
        return True
    if 0x1D4C5 <= cp and cp <= 0x1D505:
        return True
    if 0x1D507 <= cp and cp <= 0x1D50A:
        return True
    if 0x1D50D <= cp and cp <= 0x1D514:
        return True
    if 0x1D516 <= cp and cp <= 0x1D51C:
        return True
    if 0x1D51E <= cp and cp <= 0x1D539:
        return True
    if 0x1D53B <= cp and cp <= 0x1D53E:
        return True
    if 0x1D540 <= cp and cp <= 0x1D544:
        return True
    if cp == 0x1D546:
        return True
    if 0x1D54A <= cp and cp <= 0x1D550:
        return True
    if 0x1D552 <= cp and cp <= 0x1D6A5:
        return True  # Mathematical alphanumeric (letters)
    if 0x1D6A8 <= cp and cp <= 0x1D7CB:
        return True  # Mathematical Greek letters
    if 0x1D800 <= cp and cp <= 0x1D9FF:
        return False  # Sutton SignWriting is not alphabetic
    if 0x1DF00 <= cp and cp <= 0x1DF1E:
        return True  # Latin extended astral (newer)
    if 0x1E030 <= cp and cp <= 0x1E06F:
        return True  # Cyrillic extended astral
    if 0x1E100 <= cp and cp <= 0x1E12A:
        return True  # Nyiakeng Puachue Hmong
    if 0x1E2C0 <= cp and cp <= 0x1E2EB:
        return True  # Wancho
    if 0x1E4D0 <= cp and cp <= 0x1E4F9:
        return True  # Nag Mundari
    if 0x20000 <= cp and cp <= 0x2A6DF:
        return True  # CJK Ext B and beyond
    if 0x2A700 <= cp and cp <= 0x2EBE0:
        return True  # CJK Ext C..I heads (approx)
    if 0x2F800 <= cp and cp <= 0x2FA1D:
        return True  # CJK compatibility supplement
    if 0x30000 <= cp and cp <= 0x3134A:
        return True  # CJK Ext G/H
    return False


def is_letter(cps: List[Int], idx: Int) -> Bool:
    return is_letter_cp(cps[idx])


def is_number(cps: List[Int], idx: Int) -> Bool:
    return is_numeric_cp(cps[idx])


def is_symbol_cp(cp: Int) -> Bool:
    """A character the symbol branch consumes: not whitespace,
    not a letter, not a number (Rust's is_symbol)."""
    return (
        not is_whitespace_cp(cp)
        and not is_letter_cp(cp)
        and not is_numeric_cp(cp)
    )


def _ascii_lower(cp: Int) -> Int:
    if 0x41 <= cp and cp <= 0x5A:
        return cp + 32
    return cp


def contraction_len(cps: List[Int], i: Int) -> Int:
    """If a contraction starts at `i` (cps[i] is `'`), returns
    its length in codepoints including the apostrophe, else 0.
    Suffixes, longest first: re, ve, ll, then s, t, m, d; ASCII
    case-insensitive (port of the Rust helper)."""
    var c1 = -1
    var c2 = -1
    if i + 1 < len(cps):
        c1 = _ascii_lower(cps[i + 1])
    if i + 2 < len(cps):
        c2 = _ascii_lower(cps[i + 2])
    # 'r' 'e' / 'v' 'e' / 'l' 'l'
    if (
        (c1 == 0x72 and c2 == 0x65)
        or (c1 == 0x76 and c2 == 0x65)
        or (c1 == 0x6C and c2 == 0x6C)
    ):
        return 3
    if c1 == 0x73 or c1 == 0x74 or c1 == 0x6D or c1 == 0x64:  # s t m d
        return 2
    return 0


def letter_run(cps: List[Int], start: Int) -> Int:
    """Consumes a run of letters starting at `start`; returns
    the end index."""
    var i = start
    while i < len(cps) and is_letter_cp(cps[i]):
        i += 1
    return i


def word_end(cps: List[Int], end: Int) -> Int:
    """The end of a word piece whose letters end at `end`:
    absorbs one trailing contraction when one starts there."""
    if end < len(cps) and cps[end] == 0x27:  # '\''
        var clen = contraction_len(cps, end)
        if clen > 0:
            return end + clen
    return end


def symbol_run(cps: List[Int], start: Int) -> Int:
    """Consumes a symbol run starting at `start`, plus any
    \\r/\\n run that immediately follows it (the pattern's
    [\\r\\n]* tail); returns the end index."""
    var i = start
    while i < len(cps) and is_symbol_cp(cps[i]):
        i += 1
    while i < len(cps) and (cps[i] == 0x0D or cps[i] == 0x0A):
        i += 1
    return i


def scan_whitespace(cps: List[Int], i: Int) -> Int:
    """The whitespace branches: \\s*[\\r\\n]+, then \\s+(?!\\S),
    then \\s+. Port of the Rust helper."""
    var n = len(cps)
    var run_end = i
    while run_end < n and is_whitespace_cp(cps[run_end]):
        run_end += 1
    # Run contains a newline: take the run up to the end of its
    # last newline; spaces after that newline are rescanned.
    var last_newline = -1
    for k in range(i, run_end):
        if cps[k] == 0x0D or cps[k] == 0x0A:
            last_newline = k
    if last_newline >= 0:
        return last_newline + 1
    # Run reaches the end of the text: \\s+(?!\S) matches all.
    if run_end == n:
        return run_end
    # Run of 2+ before non-whitespace: \\s+(?!\S) backtracks by
    # one, leaving the final space to prefix the next piece.
    if run_end - i >= 2:
        return run_end - 1
    # A single whitespace char before non-whitespace prefixes a
    # letter or symbol piece; before a digit it stands alone.
    var next_cp = cps[run_end]
    if is_letter_cp(next_cp):
        return word_end(cps, letter_run(cps, run_end))
    if is_symbol_cp(next_cp):
        return symbol_run(cps, run_end)
    return run_end


def scan_piece(cps: List[Int], i: Int) -> Int:
    """Scans one piece starting at codepoint index `i` and
    returns the index just past its end. Mirrors the alternation
    order of the pattern. There is deliberately no standalone
    contraction branch: see word_end."""
    var n = len(cps)
    var c = cps[i]
    # 1. Letter branch, bare.
    if is_letter_cp(c):
        return word_end(cps, letter_run(cps, i))
    # 2. Letter branch with one prefix char. Covers " word",
    #    "-word", and a contraction left over after a word piece.
    if (
        not is_numeric_cp(c)
        and c != 0x0D
        and c != 0x0A
        and i + 1 < n
        and is_letter_cp(cps[i + 1])
    ):
        return word_end(cps, letter_run(cps, i + 1))
    # 3. Number branch: \\p{N}{1,3}.
    if is_numeric_cp(c):
        var j = i
        while j < n and j - i < 3 and is_numeric_cp(cps[j]):
            j += 1
        return j
    # 4. Whitespace branches.
    if is_whitespace_cp(c):
        return scan_whitespace(cps, i)
    # 5. Symbol branch.
    return symbol_run(cps, i)


def pretokenize(cps: List[Int]) -> List[List[Int]]:
    """Splits codepoints into the pieces BPE will encode
    independently. Lossless: the pieces concatenate to the
    input."""
    var pieces = List[List[Int]]()
    var i = 0
    while i < len(cps):
        var end = scan_piece(cps, i)
        var piece = List[Int]()
        for k in range(i, end):
            piece.append(cps[k])
        pieces.append(piece^)
        i = end
    return pieces^


def pretokenize_text(text: String) -> List[String]:
    """String convenience wrapper over pretokenize, matching
    the Rust signature pretokenize(&str) -> Vec<&str> in
    content."""
    from tokenizer.byte_level import string_from_codepoints

    var cps = codepoints_of(text)
    var pieces = pretokenize(cps)
    var out = List[String]()
    for piece in pieces:
        out.append(string_from_codepoints(piece, 0, len(piece)))
    return out^
