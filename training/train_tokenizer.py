# -*- coding: utf-8 -*-
"""آموزش توکنایزر SentencePiece روی متن مجموعه‌داده + تبدیل به tokenizer.json.

    python -m training.train_tokenizer \
        --corpus data/transcripts.txt          # هر خط: یک جمله
        --vocab-size 4005 \
        --out-dir weights

خروجی: tokenizer.model (SentencePiece unigram) و tokenizer.json (همان واژگان،
برای رانتایم وب) به‌همراه گزارش تطابق شناسه‌ها. عدد ۴۰۰۵ با n_bins در
config/fa.yaml باید برابر بماند (جدول embed یک ردیف بیشتر برای padding دارد).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="آموزش توکنایزر SentencePiece فارسی")
    ap.add_argument("--corpus", required=True, help="فایل متن، هر خط یک جمله")
    ap.add_argument("--vocab-size", type=int, default=4005)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "weights"))
    ap.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    ap.add_argument("--character-coverage", type=float, default=0.9995)
    ap.add_argument("--check-samples", type=int, default=200,
                    help="تعداد جمله برای تست تطابق شناسهٔ SP و tokenizers")
    args = ap.parse_args()

    import sentencepiece as spm

    corpus = Path(args.corpus)
    if not corpus.exists():
        sys.exit(f"فایل متن «{corpus}» پیدا نشد.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = out_dir / "tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
        normalization_rule_name="nmt_nfkc",
        pad_id=-1,  # the LUT embed owns padding; the tokenizer itself needs none
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
    )
    print(f"tokenizer.model ساخته شد: {model_prefix}.model")

    sp = spm.SentencePieceProcessor(str(model_prefix.with_suffix(".model")))
    assert sp.vocab_size() == args.vocab_size, (
        f"vocab {sp.vocab_size()} != {args.vocab_size} — vocab_size را تنظیم کنید"
    )

    # Convert to a HuggingFace tokenizers JSON with identical ids.
    from tokenizers.implementations import SentencePieceUnigramTokenizer

    tok = SentencePieceUnigramTokenizer.from_spm(str(model_prefix.with_suffix(".model")))
    tok.save(str(model_prefix.with_suffix(".json")))
    print(f"tokenizer.json ساخته شد: {model_prefix}.json")

    # Verify the ids agree on real text.
    corpus_lines = [
        line.strip() for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rng = random.Random(0)
    samples = rng.sample(corpus_lines, min(args.check_samples, len(corpus_lines)))
    mismatches = 0
    for text in samples:
        if sp.encode(text, out_type=int) != tok.encode(text).ids:
            mismatches += 1
    status = "OK" if mismatches == 0 else f"{mismatches}/{len(samples)} ناهماهنگ"
    print(f"تست تطابق شناسه‌ها روی {len(samples)} جمله: {status}")
    if mismatches:
        sys.exit("شناسه‌های tokenizer.json با SentencePiece یکی نیست؛ از tokenizer.model استفاده کنید.")


if __name__ == "__main__":
    main()
