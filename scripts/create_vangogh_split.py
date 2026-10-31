"""Build a group-disjoint manifest from an image audit and reviewed pairs.

This manifest uses paths relative to an external dataset root. It is not
compatible with the legacy repository-root debug manifest loader.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(audit, review, candidates, seed):
    if audit["errors"]:
        raise ValueError("Audit contains unreadable images.")
    by_path = {}
    by_hash = defaultdict(list)
    for record in audit["records"]:
        path = record["path"]
        parts = PurePosixPath(path).parts
        if (len(parts) != 2 or parts[0] not in
                {"trainA", "testA", "trainB", "testB"}
                or record["split"] != parts[0] or "\\" in path
                or parts[1] in {".", ".."}):
            raise ValueError(f"Invalid audited path: {path}")
        if path in by_path:
            raise ValueError(f"Repeated audited path: {path}")
        by_path[path] = record
        by_hash[record["pixel_sha256"]].append(path)
    if not by_hash:
        raise ValueError("Audit is empty.")

    parent = {digest: digest for digest in by_hash}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    decisions = review["decisions"]
    if len(decisions) != len(candidates["candidates"]):
        raise ValueError("Review does not cover every candidate.")
    for number, (decision, candidate) in enumerate(
            zip(decisions, candidates["candidates"]), 1):
        if (decision["pair_id"] != number or
                decision["decision"] != "keep_in_same_split_group"):
            raise ValueError("Unexpected review decision.")
        for key in ("left", "right", "left_aliases", "right_aliases"):
            if decision[key] != candidate[key]:
                raise ValueError("Review and candidate list differ.")
        hashes = []
        for side in ("left", "right"):
            digest = by_path[decision[side]]["pixel_sha256"]
            if set(decision[side + "_aliases"]) != set(by_hash[digest]):
                raise ValueError("Reviewed aliases differ from image audit.")
            hashes.append(digest)
        parent[find(hashes[1])] = find(hashes[0])

    components = defaultdict(list)
    for digest in sorted(by_hash):
        components[find(digest)].append(digest)
    domains = {"content": [], "style": []}
    excluded = []
    for digests in components.values():
        paths = sorted(p for digest in digests for p in by_hash[digest])
        letters = {PurePosixPath(p).parts[0][-1] for p in paths}
        if len(letters) != 1:
            raise ValueError("A reviewed group crosses photo/art domains.")
        domain = "style" if letters == {"A"} else "content"
        group_id = hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest()
        is_photo_test = domain == "content" and any(
            p.startswith("testB/") for p in paths)
        entries = []
        for digest in sorted(digests):
            aliases = sorted(by_hash[digest])
            eligible = [p for p in aliases if not is_photo_test or
                        p.startswith("testB/")]
            if is_photo_test:
                for path in aliases:
                    if path.startswith("trainB/"):
                        excluded.append({"path": path, "group_id": group_id,
                                         "reason": "related_to_photo_test_group"})
            if not eligible:
                continue
            # Keep one file per exact decoded image. Preserve all source aliases.
            representative = min(eligible, key=lambda p: (
                not p.startswith("train"), p))
            record = by_path[representative]
            entries.append({"path": representative,
                            "pixel_sha256": digest, "group_id": group_id,
                            "width": record["width"], "height": record["height"],
                            "source_aliases": aliases})
        domains[domain].append((group_id, entries, is_photo_test))

    splits = {name: {"content": [], "style": []}
              for name in ("train", "validation", "test")}

    def rank(group):
        return hashlib.sha256(f"{seed}:{group[0]}".encode()).hexdigest()

    for domain, groups in domains.items():
        pool = sorted([g for g in groups if not g[2]], key=rank)
        if len(pool) < 10:
            raise ValueError(f"Too few independent {domain} groups.")
        n_val = max(1, round(len(pool) * 0.1))
        n_test = max(1, round(len(pool) * 0.1)) if domain == "style" else 0
        for index, (_, entries, _) in enumerate(pool):
            split = ("validation" if index < n_val else
                     "test" if index < n_val + n_test else "train")
            splits[split][domain].extend(entries)
        for _, entries, is_test in groups:
            if is_test:
                splits["test"][domain].extend(entries)

    owners = {}
    hashes_seen = set()
    for split, domains_in_split in splits.items():
        for domain, entries in domains_in_split.items():
            if not entries:
                raise ValueError(f"Empty split: {split}/{domain}")
            entries.sort(key=lambda entry: entry["path"])
            for entry in entries:
                digest, group_id = entry["pixel_sha256"], entry["group_id"]
                if digest in hashes_seen:
                    raise ValueError("An exact image was selected more than once.")
                hashes_seen.add(digest)
                if owners.setdefault(group_id, split) != split:
                    raise ValueError("A related-image group crosses splits.")
    excluded_hashes = {by_path[e["path"]]["pixel_sha256"] for e in excluded}
    if hashes_seen | excluded_hashes != set(by_hash):
        raise ValueError("Some audited image groups were not accounted for.")
    return splits, sorted(excluded, key=lambda e: e["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    audit, candidates, review = map(
        read_json, (args.audit, args.candidates, args.review))
    if review["candidate_report_sha256"] != sha256(args.candidates):
        raise ValueError("Candidate report hash does not match review.")
    if Path(candidates["dataset_root"]).resolve() != Path(audit["dataset_root"]).resolve():
        raise ValueError("Audit and candidates refer to different datasets.")
    if candidates["unique_images_checked"] != len({
            r["pixel_sha256"] for r in audit["records"]}):
        raise ValueError("Audit and candidates have different unique-image counts.")
    splits, excluded = build(audit, review, candidates, args.seed)
    result = {
        "schema_version": 2,
        "dataset_id": "vangogh2photo_grouped_v1",
        "path_base": "external_dataset_root",
        "seed": args.seed,
        "pixel_hash_method": "SHA256(width + 'x' + height + ':' + EXIF-transposed RGB bytes)",
        "source_sha256": {"audit": sha256(args.audit),
                          "candidates": sha256(args.candidates),
                          "review": sha256(args.review)},
        "policy": {
            "style": "Approximately 80/10/10 by related-image group",
            "content": "Original test groups retained; 10% remaining groups for validation",
            "exact_duplicates": "One representative per decoded pixel hash",
            "near_duplicates": "Retain variants together; exclude trainB members related to testB",
        },
        "limitations": ["Only the 21 reviewed candidate pairs were screened visually.",
                        "Undetected near duplicates may remain.",
                        "This is a custom split, not the original benchmark split.",
                        "Source images must be revalidated before training."],
        "splits": splits,
        "excluded": excluded,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for split, domains in splits.items():
        print(f"{split}: {len(domains['content'])} content, {len(domains['style'])} style")
    print("Excluded trainB files related to testB:", len(excluded))
    print("Cross-split exact-image overlap: 0")
    print("Cross-split reviewed-group overlap: 0")
    print("Saved:", args.output.resolve())
    print("Manifest checks passed; source images were not modified.")
    print("This schema needs an external-root loader before training.")


if __name__ == "__main__":
    main()
