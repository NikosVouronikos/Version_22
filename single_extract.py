import sys
import subprocess


def main():

    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "python single_extract.py <attacked_image> <code>"
        )
        return

    image_path = sys.argv[1]
    code = sys.argv[2]

    cmd = [
        sys.executable,
        "validator.py",
        image_path,
        code
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)


if __name__ == "__main__":
    main()