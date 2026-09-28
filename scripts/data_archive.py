"""Create and restore a verified ZIP archive of the persistent data shards."""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile


SHARD_RE = re.compile(r"shard_\d{5}\.parquet")
MANIFEST_VERSION = 1


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def shard_paths(source_dir):
    return sorted(
        os.path.join(source_dir, name)
        for name in os.listdir(source_dir)
        if SHARD_RE.fullmatch(name)
    )


def inspect_zip(archive_path, expected_names, expected_sizes=None):
    expected_names = set(expected_names)
    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        if set(names) != expected_names:
            missing = sorted(expected_names - set(names))
            extra = sorted(set(names) - expected_names)
            raise ValueError(f"ZIP member mismatch; missing={missing}, extra={extra}")
        for info in infos:
            if not SHARD_RE.fullmatch(info.filename):
                raise ValueError(f"Unsafe or unexpected ZIP member: {info.filename!r}")
            if expected_sizes is not None and info.file_size != expected_sizes[info.filename]:
                raise ValueError(f"Wrong uncompressed size for {info.filename}")
        bad_member = zf.testzip()
        if bad_member is not None:
            raise IOError(f"ZIP CRC verification failed for {bad_member}")
        return {
            info.filename: {"size": info.file_size, "crc32": f"{info.CRC:08x}"}
            for info in infos
        }


def atomic_json_save(payload, path):
    temp_path = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def validate_published_archive(archive_path, manifest_path, expected_shards, verify_members=True):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported archive manifest: {manifest_path}")
    members = manifest.get("members", {})
    if len(members) != expected_shards:
        raise ValueError(f"Expected {expected_shards} manifest members, found {len(members)}")
    if os.path.getsize(archive_path) != manifest.get("archive_size"):
        raise IOError("Archive size does not match its manifest")
    if sha256(archive_path) != manifest.get("archive_sha256"):
        raise IOError("Archive SHA-256 does not match its manifest")
    if verify_members:
        inspected = inspect_zip(archive_path, members, {name: data["size"] for name, data in members.items()})
        if inspected != members:
            raise IOError("ZIP member metadata does not match its manifest")
    return manifest


def pack(args):
    paths = shard_paths(args.source_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.archive)), exist_ok=True)
    if os.path.isfile(args.archive) and os.path.isfile(args.manifest):
        try:
            validate_published_archive(args.archive, args.manifest, args.expected_shards)
            print(f"Existing archive is already complete and verified: {args.archive}", flush=True)
            if args.delete_source:
                print("Archive is durable and verified; deleting original shards...", flush=True)
                for path in paths:
                    os.remove(path)
                print(f"Deleted {len(paths)} original shards from {args.source_dir}", flush=True)
            return
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"Existing archive is invalid and will be replaced: {exc}", flush=True)

    if len(paths) != args.expected_shards:
        raise SystemExit(
            f"Expected exactly {args.expected_shards} shards in {args.source_dir}; found {len(paths)}"
        )
    names = [os.path.basename(path) for path in paths]
    sizes = {os.path.basename(path): os.path.getsize(path) for path in paths}

    os.makedirs(args.staging_dir, exist_ok=True)
    local_staging_dir = tempfile.mkdtemp(prefix="nanollm-data-archive-", dir=args.staging_dir)
    local_archive = os.path.join(local_staging_dir, os.path.basename(args.archive))
    uploaded_archive = f"{args.archive}.uploading-{uuid.uuid4().hex}"
    try:
        with zipfile.ZipFile(
            local_archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=args.compression_level,
            allowZip64=True,
        ) as zf:
            for index, path in enumerate(paths, start=1):
                print(f"Compressing {index}/{len(paths)}: {os.path.basename(path)}", flush=True)
                zf.write(path, arcname=os.path.basename(path))

        print("Verifying every ZIP member and CRC before publication...", flush=True)
        members = inspect_zip(local_archive, names, sizes)
        local_size = os.path.getsize(local_archive)
        local_sha256 = sha256(local_archive)
        manifest = {
            "format_version": MANIFEST_VERSION,
            "archive": os.path.basename(args.archive),
            "archive_size": local_size,
            "archive_sha256": local_sha256,
            "compression": "deflate",
            "compression_level": args.compression_level,
            "members": members,
        }
        print("Uploading verified archive to persistent storage...", flush=True)
        shutil.copyfile(local_archive, uploaded_archive)
        if os.path.getsize(uploaded_archive) != local_size or sha256(uploaded_archive) != local_sha256:
            raise IOError("Uploaded archive does not match the verified local archive")
        os.replace(uploaded_archive, args.archive)
        atomic_json_save(manifest, args.manifest)
        if os.path.getsize(args.archive) != local_size:
            raise IOError("Published archive size changed during atomic rename")
        print(f"Published and verified: {args.archive}", flush=True)
    finally:
        if os.path.exists(uploaded_archive):
            os.remove(uploaded_archive)
        shutil.rmtree(local_staging_dir, ignore_errors=True)

    if args.delete_source:
        print("Archive is durable and verified; deleting original shards...", flush=True)
        for path in paths:
            os.remove(path)
        print(f"Deleted {len(paths)} original shards from {args.source_dir}", flush=True)


def extract(args):
    manifest = validate_published_archive(args.archive, args.manifest, args.expected_shards)
    os.makedirs(args.output_dir, exist_ok=True)
    expected_names = set(manifest["members"])
    with zipfile.ZipFile(args.archive, "r") as zf:
        for index, info in enumerate(zf.infolist(), start=1):
            if info.filename not in expected_names or not SHARD_RE.fullmatch(info.filename):
                raise ValueError(f"Unsafe or unexpected ZIP member: {info.filename!r}")
            print(f"Extracting {index}/{len(expected_names)}: {info.filename}", flush=True)
            zf.extract(info, args.output_dir)

    extracted = shard_paths(args.output_dir)
    if len(extracted) != args.expected_shards:
        raise IOError(f"Expected {args.expected_shards} extracted shards, found {len(extracted)}")
    for path in extracted:
        name = os.path.basename(path)
        if os.path.getsize(path) != manifest["members"][name]["size"]:
            raise IOError(f"Extracted shard has the wrong size: {path}")
    print(f"Extracted and verified {len(extracted)} shards in {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="create, verify, and publish the archive")
    pack_parser.add_argument("--source-dir", required=True)
    pack_parser.add_argument("--archive", required=True)
    pack_parser.add_argument("--manifest", required=True)
    pack_parser.add_argument("--expected-shards", type=int, required=True)
    pack_parser.add_argument("--compression-level", type=int, default=1, choices=range(0, 10))
    pack_parser.add_argument("--staging-dir", default="/content", help="fast local directory used before upload")
    pack_parser.add_argument("--delete-source", action="store_true")
    pack_parser.set_defaults(func=pack)

    extract_parser = subparsers.add_parser("extract", help="verify and extract the archive")
    extract_parser.add_argument("--archive", required=True)
    extract_parser.add_argument("--manifest", required=True)
    extract_parser.add_argument("--output-dir", required=True)
    extract_parser.add_argument("--expected-shards", type=int, required=True)
    extract_parser.set_defaults(func=extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
