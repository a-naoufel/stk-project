"""Build a flat upload zip locally; does not upload or deploy anything."""

import zipfile

from .submission_files import AGENT_DIR, ROOT, collect_files


def main():
    files = collect_files(AGENT_DIR)
    output = ROOT / "dist/stk-agent.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    print(f"Local package: {output}\nFiles: {', '.join(p.name for p in files)}")


if __name__ == "__main__":
    main()
