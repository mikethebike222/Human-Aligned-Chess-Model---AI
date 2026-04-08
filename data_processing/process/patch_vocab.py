"""
patch_vocab.py — Patch ALLIE's 2022.bin to work with our expanded vocabulary.

WHY THIS IS NEEDED
------------------
ALLIE's tokenizer had 1997 tokens total. Our tokenizer adds 10 rapid time
control tokens, bringing the total to 2007.

The new vocab layout is:
    [0    – 1967]  chess moves        (unchanged)
    [1968 – 1991]  blitz time ctrls   (unchanged)
    [1992 – 2001]  rapid time ctrls   (NEW — these 10 slots shift everything after)
    [2002]         <RESIGNED-OR-CHECKMATED>  (was 1992, now +10)
    [2003]         <BOS>                     (was 1993, now +10)
    [2004]         <EOS>                     (was 1994, now +10)
    [2005]         <UNK>                     (was 1995, now +10)
    [2006]         <PAD>                     (was 1996, now +10)

ELO tokens (stored as elo + vocab_size) also shift:
    old: elo + 1997
    new: elo + 2007

So any token in 2022.bin where bits 13-0 >= 1992 needs to have 10 added
to those bits. Chess moves (0-1967) and blitz time controls (1968-1991)
are completely unaffected.

HOW THE PATCH WORKS
-------------------
For every uint32 in the file:
    lower_14 = token & 0x3FFF          # extract bits 13-0
    if lower_14 >= OLD_SPECIAL_START:  # 1992 (first token that shifts)
        lower_14 += R                  # shift by number of new rapid tokens
    token = (token & ~0x3FFF) | lower_14  # put bits back

This is a fast vectorized numpy operation — takes seconds even on large files.

Usage:
    python patch_vocab.py \\
        --input  ~/data/lichess-2022-blitz-train/2022.bin \\
        --output ~/data/lichess-2022-blitz-train/2022-patched.bin \\
        --n_new_tokens 10

Always keep the original 2022.bin as a backup until you've verified the patch.
"""

import argparse

import numpy as np


# The first token ID that shifts when we insert rapid tokens.
# Everything from here up (special tokens + ELO offsets) needs to move.
# Chess moves (0–1967) and blitz time controls (1968–1991) are unaffected.
OLD_SPECIAL_START = 1992


def patch_bin(input_path: str, output_path: str, n_new_tokens: int) -> None:
    """
    Load a .bin memmap, shift token IDs >= OLD_SPECIAL_START by n_new_tokens,
    and write the result to output_path.

    Processes in chunks to avoid loading the full file into RAM.

    Args:
        input_path:    path to the original ALLIE .bin file
        output_path:   path to write the patched file
        n_new_tokens:  how many new tokens were inserted (10 for our rapid tokens)
    """
    CHUNK = 50_000_000  # 50M tokens per chunk (~200MB) — safe for any instance size

    print(f"Loading {input_path} ...")
    data = np.memmap(input_path, dtype=np.uint32, mode="r")
    n = len(data)
    print(f"  {n:,} tokens ({n * 4 / 1e9:.2f} GB)")

    # Pre-allocate output memmap (same size as input)
    print(f"Allocating output file {output_path} ...")
    out = np.memmap(output_path, dtype=np.uint32, mode="w+", shape=(n,))

    n_shifted_total = 0

    print(f"Patching in chunks of {CHUNK:,} tokens ...")
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        chunk = data[start:end].copy()  # copy chunk into RAM

        lower = (chunk & np.uint32(0x3FFF)).astype(np.int32)
        upper = chunk & ~np.uint32(0x3FFF)

        needs_shift = lower >= OLD_SPECIAL_START
        n_shifted_total += int(needs_shift.sum())
        lower[needs_shift] += n_new_tokens

        out[start:end] = upper | lower.astype(np.uint32)

        pct = end / n * 100
        print(f"  {end:,} / {n:,} tokens ({pct:.1f}%)", end="\r")

    out.flush()
    print(f"\n  {n_shifted_total:,} tokens shifted ({n_shifted_total / n * 100:.1f}%)")
    print("Done!")
    print(f"  Original vocab boundary: {OLD_SPECIAL_START}")
    print(f"  New vocab boundary:      {OLD_SPECIAL_START + n_new_tokens}")
    print(f"  ELO offset (old):        1997")
    print(f"  ELO offset (new):        {1997 + n_new_tokens}")


def verify_patch(original_path: str, patched_path: str, n_new_tokens: int, n_samples: int = 10) -> None:
    """
    Quick sanity check: sample a few tokens and verify the patch looks correct.
    Prints before/after for tokens that were shifted.
    """
    print("\nVerifying patch on sample tokens...")
    orig    = np.memmap(original_path,  dtype=np.uint32, mode="r")
    patched = np.memmap(patched_path,   dtype=np.uint32, mode="r")

    assert len(orig) == len(patched), "File sizes don't match!"

    # Find some tokens that should have been shifted (sample first 10M only)
    sample = orig[:10_000_000]
    orig_lower = sample & np.uint32(0x3FFF)
    shifted_indices = np.where(orig_lower >= OLD_SPECIAL_START)[0]

    if len(shifted_indices) == 0:
        print("  No tokens needed shifting — is the file already patched?")
        return

    sample_idx = shifted_indices[:n_samples]
    for i in sample_idx:
        o = int(orig[i] & np.uint32(0x3FFF))
        p = int(patched[i] & np.uint32(0x3FFF))
        shift_ok = (p == o + n_new_tokens)
        status = "✓" if shift_ok else "✗ ERROR"
        print(f"  [{status}] index {i}: {o} → {p} (expected {o + n_new_tokens})")

    # Also verify that chess move tokens (0–1967) were NOT shifted
    move_indices = np.where(orig_lower < 1968)[0][:n_samples]
    for i in move_indices:
        o = int(orig[i] & np.uint32(0x3FFF))
        p = int(patched[i] & np.uint32(0x3FFF))
        unchanged = (o == p)
        status = "✓" if unchanged else "✗ ERROR"
        print(f"  [{status}] move token at index {i}: {o} unchanged → {p}")

    print("Verification complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Patch ALLIE's 2022.bin to work with an expanded vocabulary"
    )
    parser.add_argument(
        "--input",  required=True,
        help="Path to original 2022.bin (ALLIE's tokenized blitz data)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the patched .bin file",
    )
    parser.add_argument(
        "--n_new_tokens", type=int, default=10,
        help="Number of new tokens inserted into the vocab (default: 10 rapid tokens)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run a quick sanity check after patching",
    )
    args = parser.parse_args()

    patch_bin(args.input, args.output, args.n_new_tokens)

    if args.verify:
        verify_patch(args.input, args.output, args.n_new_tokens)


if __name__ == "__main__":
    main()
