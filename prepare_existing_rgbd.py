"""Link an existing RGB-D dataset into the layout expected by Samba.

The source images stay in place. Only directory symlinks and text manifests are
created under ``rgbd_dataset``.
"""

from pathlib import Path
import argparse


def link_split(source: Path, target: Path, manifest: str) -> int:
    target.mkdir(parents=True, exist_ok=True)
    image_files = {path.stem: path for path in (source / "RGB").iterdir() if path.is_file()}
    gt_files = {path.stem: path for path in (source / "GT").iterdir() if path.is_file()}
    depth_files = {path.stem: path for path in (source / "depth").iterdir() if path.is_file()}
    if not image_files or image_files.keys() != gt_files.keys() or image_files.keys() != depth_files.keys():
        raise ValueError(f"RGB, GT, and depth files must have matching nonempty stems: {source}")

    for directory in ("RGB", "GT", "depth"):
        link = target / directory
        expected = (source / directory).resolve()
        if link.is_symlink() and link.resolve() == expected:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Refusing to replace existing path: {link}")
        link.symlink_to(expected, target_is_directory=True)

    lines = (
        f"/RGB/{image_files[stem].name} /GT/{gt_files[stem].name} /depth/{depth_files[stem].name}\n"
        for stem in sorted(image_files)
    )
    (target / manifest).write_text("".join(lines), encoding="utf-8")
    return len(image_files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/home/wmy/project/RGB-D dataset"))
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent / "rgbd_dataset"
    counts = {"train_DUT": link_split(args.source / "train_data", destination / "train_DUT", "train.txt")}
    for target_name, source_name in (
        ("DUT-RGBD", "DUT"),
        ("NJU2K", "NJU2K"),
        ("NLPR", "NLPR"),
        ("SIP", "SIP"),
        ("STERE", "STERE"),
    ):
        counts[target_name] = link_split(
            args.source / "test_data" / source_name,
            destination / target_name,
            "test.txt",
        )
    for name, count in counts.items():
        print(f"{name}: {count} triplets -> {destination / name}")


if __name__ == "__main__":
    main()
